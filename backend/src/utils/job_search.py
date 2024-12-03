import json
import torch
import sys
import os
import re

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(backend_dir)

from logging import Logger
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from llama_index.core import Settings, PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# Local imports (change relative imports to absolute)
from src.models import UserProfile, JobRecommendation, ChatHistory, CareerRecommendation
from src.utils.job_retriever import JobSearchRetriever, SearchStrategy
from src.utils.profile_questioner import ProfileQuestioner
from src.prompts.job_search_prompts import SYSTEM_PROMPT, FOLLOWUP_PROMPT
from src.prompts.search_prompts import SEARCH_ENHANCEMENT
from src.evaluation.persona_agent import CareerPersonaAgent
from src.evaluation.metrics import RecommendationMetrics
from src.evaluation.performance_tracker import ModelPerformanceTracker
from src.config import get_settings
from src.crud import create_chat_message, create_job_recommendation

class JobSearchService:
    def __init__(self, retriever, settings, db: Session):
        print("\n========== JOB SEARCH SERVICE INIT DEBUG ==========")
        print("1. Initializing JobSearchService")
        self.retriever = retriever
        self.db = db
        self.store = {}
        
        # Initialize OpenAI chat correctly
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4-turbo-preview",
            api_key=settings.openai_api_key
        )
        print("2. OpenAI chat initialized")
        
        # Initialize profile questioner
        self.profile_questioner = ProfileQuestioner(settings)
        print("3. ProfileQuestioner initialized")
        
        # Initialize performance tracker
        self.performance_tracker = ModelPerformanceTracker()
        print("4. Performance tracker initialized")
        
        # Add a state tracker for question-answer flow
        self.qa_state = {}
        print("5. JobSearchService initialization complete")
        
        # Your system prompt
        self.system_prompt = SYSTEM_PROMPT
        
        # Your conversation prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Your follow-up prompt
        self.followup_prompt = PromptTemplate(FOLLOWUP_PROMPT)
        
        # Set up search components
        self.cohere_rerank = CohereRerank(api_key=settings.cohere_api_key, top_n=3)
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7),
                self.cohere_rerank,
            ]
        )
        
        # Initialize chat engine with both prompts
        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=self.query_engine,
            condense_question_prompt=self.followup_prompt,
            chat_prompt=self.conversation_prompt,
            verbose=True
        )

    def get_session_history(self, chat_session_id: str) -> ChatMessageHistory:
        if chat_session_id not in self.store:
            self.store[chat_session_id] = ChatMessageHistory()
        return self.store[chat_session_id]

    def chat(self, message: str, chat_session_id: str = "default") -> Dict:
        try:
            print("\n========== CHAT METHOD DEBUG ==========")
            print(f"1. Processing message for session: {chat_session_id}")
            
            state = self.qa_state.get(chat_session_id)
            print(f"2. Current state: {state}")
            
            # Get user profile once at the beginning
            user_profile = self.db.query(UserProfile).filter(
                UserProfile.user_session_id == chat_session_id
            ).first()
            
            if not user_profile:
                return {
                    "response": "Please create a profile first",
                    "chat_session_id": chat_session_id
                }

            # Initialize state if first message
            if not state:
                print("3. Initializing new state...")
                
                profile_data = {
                    'core_values': user_profile.core_values or [],
                    'work_culture': user_profile.work_culture or [],
                    'skills': user_profile.skills or [],
                    'additional_interests': user_profile.additional_interests or '',
                    'chat_session_id': chat_session_id  # Required for career recommendations
                }
                
                # Get initial recommendations using sparse search
                print("4. Getting initial job recommendations...")
                initial_recommendations = self.get_initial_recommendations(profile_data)
                if not initial_recommendations:
                    print("Error: No initial recommendations found")
                    return {
                        "response": "No matching jobs found. Please adjust your profile criteria.",
                        "chat_session_id": chat_session_id
                    }
                print(f"5. Found {len(initial_recommendations)} initial recommendations")
                
                # Analyze job patterns
                print("6. Analyzing job patterns...")
                job_patterns = self._analyze_job_patterns(initial_recommendations)
                print(f"7. Pattern analysis complete")
                
                # Get first question using both job and career recommendations
                print("8. Generating initial question...")
                enhanced_profile = {
                    **profile_data,
                    'initial_jobs': initial_recommendations[:5],  # Top 5 job matches
                    'job_patterns': job_patterns
                }
                print(enhanced_profile['job_patterns']);
                
                initial_question = self.profile_questioner.generate_questions(enhanced_profile)[0]
                print(f"9. Generated first question: {initial_question}")
                
                state = {
                    "current_question": initial_question,
                    "previous_qa": [],
                    "complete": False,
                    "profile_data": enhanced_profile,
                    "initial_recommendations": initial_recommendations,
                    "job_patterns": job_patterns,
                    "required_fields": {
                        "required_skills": False,
                        "work_environment": False,
                        "team_size": False,
                        "industry": False,
                        "experience_level": False,
                        "preferences": False
                    }
                }
                self.qa_state[chat_session_id] = state
                
                return {
                    "response": initial_question,
                    "chat_session_id": chat_session_id
                }

            # Handle ongoing Q&A
            if not state.get("complete"):
                # Store the current Q&A pair
                current_qa = {
                    "question": state["current_question"],
                    "response": message
                }
                state["previous_qa"].append(current_qa)
                print(f"6. Stored Q&A pair: {json.dumps(current_qa, indent=2)}")
                
                # Process the response
                processed_response = self.profile_questioner.process_response(
                    current_qa["question"], 
                    current_qa["response"]
                )
                print(f"7. Processed response: {json.dumps(processed_response, indent=2)}")
                
                # Update which fields we've collected
                for field, value in processed_response.items():
                    if value:  # If we got a non-empty value
                        state["required_fields"][field] = True
                
                # Check if we have all required information
                missing_fields = [
                    field for field, collected in state["required_fields"].items() 
                    if not collected and field in ['required_skills', 'work_environment']  # Only check mandatory fields
                ]
                print(f"8. Missing fields: {missing_fields}")
                
                # If we have 3+ Q&A pairs and still missing optional fields, set defaults
                if len(state["previous_qa"]) >= 3:
                    for field in ['team_size', 'experience_level']:
                        if not state["required_fields"].get(field):
                            print(f"Setting default for optional field: {field}")
                            state["required_fields"][field] = True
                
                if not missing_fields:
                    print("9. All required information collected, processing final recommendations...")
                    # Process all Q&A responses for final recommendations
                    refined_results = self.refine_recommendations(
                        state.get("initial_recommendations", []),
                        [self.profile_questioner.process_response(qa["question"], qa["response"]) 
                         for qa in state["previous_qa"]],
                        state["profile_data"]
                    )
                    
                    if refined_results and refined_results.get('recommendations'):
                        top_jobs = refined_results['recommendations'][:5]
                        stored_recs = self.store_recommendations(
                            top_jobs,
                            chat_session_id,
                            user_profile.id,
                            'refined'
                        )
                        
                        state["complete"] = True
                        self.qa_state[chat_session_id] = state
                        
                        return {
                            "response": "Great! I've gathered all the information I need. Let me show you the recommendations that best match your preferences.",
                            "chat_session_id": chat_session_id,
                            "qa_complete": True,
                            "recommendations": stored_recs,
                            "evaluation": refined_results.get('evaluation', {})
                        }
                
                # Generate next question
                next_question = self.profile_questioner.generate_questions(
                    state["profile_data"], 
                    state["previous_qa"]
                )[0]
                print(f"10. Generated next question: {next_question}")
                
                state["current_question"] = next_question
                return {
                    "response": next_question,
                    "chat_session_id": chat_session_id
                }

            # Regular chat after Q&A
            print("11. Chat session complete")
            return {
                "response": "Your job recommendations are ready! Let me know if you'd like to start a new search.",
                "chat_session_id": chat_session_id
            }
                    
        except Exception as e:
            print(f"Error in chat method: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return {
                "response": "I encountered an error. Let's try a different approach. What specific job aspects are most important to you?",
                "chat_session_id": chat_session_id
            }

    def _process_qa_answers(self, answers: List[Dict]) -> Dict:
        """Process Q&A answers into search parameters"""
        search_params = {
            "technologies": [],
            "role_focus": "",
            "interests": [],
            "project_type": ""
        }
        
        for qa in answers:
            question = qa["question"].lower()
            answer = qa["answer"].lower()
            
            if "programming languages" in question:
                search_params["technologies"] = [tech.strip() for tech in answer.split(",")]
            elif "type of development role" in question:
                search_params["role_focus"] = answer
            elif "aspects of software development" in question:
                search_params["interests"] = [interest.strip() for interest in answer.split(",")]
            elif "projects" in question:
                search_params["project_type"] = answer
        
        return search_params

    def _build_search_query(self, params: Dict) -> str:
        """Build search query from parameters"""
        query_parts = []
        
        if params["technologies"]:
            query_parts.append(f"technologies: {', '.join(params['technologies'])}")
        if params["role_focus"]:
            query_parts.append(f"role: {params['role_focus']}")
        if params["interests"]:
            query_parts.append(f"focus: {', '.join(params['interests'])}")
        if params["project_type"]:
            query_parts.append(f"projects: {params['project_type']}")
        
        return " AND ".join(query_parts)

    def search(self, query: str, db: Session, user_session_id: str = None) -> Dict:
        """Main search method"""
        print("\n=== Search Debug ===")
        print(f"Query: {query}")
        print(f"Session ID: {user_session_id}")  # This is actually a chat session ID
        
        try:
            # Get or create chat history - this is correct as is
            history = self.get_session_history(user_session_id)
            
            # Get user profile using the same session ID
            user_profile = db.query(UserProfile).filter(
                UserProfile.user_session_id == user_session_id  # Correct - matches chat_session_id
            ).first()
            
            if not user_profile:
                raise ValueError("User profile not found")
            
            # Store chat message - correct as is
            history.add_user_message(query)
            create_chat_message(
                db=db,
                chat_session_id=user_session_id,  # Correct - this is the chat_session_id
                message=query,
                is_user=True
            )
            
            # Get job recommendations using graph-enhanced search
            self.retriever.strategy = SearchStrategy.GRAPH
            results = self.retriever.search_jobs(query)
            print(f"Search returned {len(results)} results")

            # Get user profile once for both filtering and personalization
            user_profile = db.query(UserProfile).filter(
                UserProfile.user_session_id == user_session_id
            ).first()
            
            # Filter and process results
            filtered_results = self.filter_irrelevant_jobs(results, query, user_profile)
            formatted_results = self._format_results(filtered_results)
            
            # Add personalization context if profile exists
            user_context = ""
            if user_profile:
                user_context = f"""
                User Preferences:
                - Skills: {', '.join(user_profile.skills or [])}
                - Values: {', '.join(user_profile.core_values or [])}
                - Work Culture: {', '.join(user_profile.work_culture or [])}
                - Goals: {user_profile.goals}
                """
            
            # Add evaluation if user profile exists
            if user_profile:
                evaluation_results = self.evaluate_recommendations(
                    recommendations=formatted_results,
                    user_profile={
                        'skills': user_profile.skills,
                        'core_values': user_profile.core_values,
                        'work_culture': user_profile.work_culture,
                        'additional_interests': user_profile.additional_interests
                    }
                )
                return {
                    'jobs': formatted_results,
                    'chat_session_id': user_session_id,  # Correct - maintaining the same ID
                    'user_context': user_context,
                    'evaluation': evaluation_results
                }
            
            return {
                "jobs": formatted_results,
                "chat_session_id": user_session_id,  # Correct - maintaining the same ID
                "user_context": user_context if user_profile else None,
                "evaluation_results": {}
            }
                
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {
                "error": str(e),
                "jobs": [],
                "chat_session_id": user_session_id  # Add this to maintain consistency
            }

    def filtered_search(self, **filters):
        try:
            print(f"Applying filters: {filters}")  # Debug log
            
            # Extract filter values
            min_salary = filters.get('min_salary')
            required_skills = filters.get('required_skills', [])
            work_environment = filters.get('work_environment')
            experience_years = filters.get('experience_years')
            education_level = filters.get('education_level')
            
            print(f"Extracted filters - Skills: {required_skills}, Environment: {work_environment}")  # Debug log
            
            # Create filter dict for the retriever
            filters = {
                'min_salary': min_salary,
                'required_skills': required_skills,
                'work_environment': work_environment,
                'experience_years': experience_years,
                'education_level': education_level
            }
            
            # Use semantic search with filters
            self.retriever.strategy = SearchStrategy.SEMANTIC
            results = self.retriever.search_jobs(
                query="",  # Empty query since we're filtering
                filters=filters
            )
            
            # Format results
            formatted_results = self._format_results(results)
            
            return {
                "jobs": formatted_results,
                "filters_applied": {k: v for k, v in filters.items() if v is not None}
            }
            
        except Exception as e:
            print(f"Error in filtered search: {str(e)}")
            return {
                "error": str(e),
                "jobs": []
            }

    def verify_and_rank_results(self, results: List[Dict], user_profile: UserProfile) -> List[Dict]:
        """Verify and rank job recommendations based on user profile"""
        verified_results = []
        
        for job in results:
            structured_desc = job.get('structured_description', {})
            
            # Calculate various match scores
            skill_match = self._calculate_skill_match(
                user_profile.skills,
                structured_desc.get('required_skills', [])
            )
            
            # Add verification data
            job['match_data'] = {
                'skill_match': skill_match,
                'skill_level': self._determine_skill_level(structured_desc),
                'recommended_skills': self._get_skill_gaps(
                    user_profile.skills,
                    structured_desc.get('required_skills', [])
                )
            }
            
            verified_results.append(job)
        
        # Sort by match score
        return sorted(verified_results, key=lambda x: x['match_data']['skill_match'], reverse=True)

    def _calculate_skill_match(self, user_skills: List[str], required_skills: List[str]) -> float:
        """Calculate semantic skill match score"""
        if not user_skills or not required_skills:
            return 0.0
        
        total_score = 0
        for user_skill in user_skills:
            # Get best match score for this skill
            skill_scores = [
                self._get_skill_similarity(user_skill, req_skill)
                for req_skill in required_skills
            ]
            total_score += max(skill_scores) if skill_scores else 0
        
        return total_score / len(user_skills)  # Normalize by user skills

    def _get_skill_similarity(self, skill: str, text: str) -> float:
        """Calculate semantic similarity between a skill and text"""
        try:
            print(f"\n=== Skill Similarity Debug ===")
            print(f"Comparing: '{skill}' with text length: {len(text)}")
            
            # Clean inputs
            skill = skill.lower().strip()
            text = text.lower().strip()
            
            # Get embeddings
            skill_embedding = self.retriever.embed_model.encode(skill)
            text_embedding = self.retriever.embed_model.encode(text)
            
            # Calculate similarity
            similarity = cosine_similarity(
                [skill_embedding],
                [text_embedding]
            )[0][0]
            
            print(f"Similarity Score: {similarity}")
            return float(similarity)
            
        except Exception as e:
            print(f"Error in skill similarity: {str(e)}")
            print(f"Skill: {skill}")
            print(f"Text snippet: {text[:100]}...")  # Print first 100 chars of text
            return 0.0

    def filter_irrelevant_jobs(self, jobs: List[Dict], query: str, user_profile: Optional[UserProfile] = None) -> List[Dict]:
        """Filter out irrelevant jobs based on query and user preferences"""
        filtered_jobs = []
        query_embedding = self.retriever.embed_model.encode([query])[0]
        print(f"Filtering {len(jobs)} jobs")

        for job in jobs:
            relevance_score = 0.0
            # 1. Title relevance (30% weight)
            title_relevance = self._check_title_relevance(job.get('title', ''), query)
            relevance_score += title_relevance * 0.3
            
            # 2. Skill match if user profile exists (30% weight)
            if user_profile and user_profile.skills:
                required_skills = set(s.lower() for s in job.get('required_skills', []))
                user_skills = set(s.lower() for s in user_profile.skills)
                exact_matches = len(required_skills.intersection(user_skills))
                
                if exact_matches > 0:
                    relevance_score += 0.3  # Boost score for exact matches
                
                # Also consider semantic skill matching
                skill_match = self._calculate_skill_match(
                    user_profile.skills,
                    job.get('required_skills', [])
                )
                relevance_score += skill_match * 0.2
            
            # 3. Semantic relevance (40% weight)
            if 'description_embedding' in job:
                semantic_score = cosine_similarity(
                    [query_embedding],
                    [job['description_embedding']]
                )[0][0]
                relevance_score += semantic_score * 0.2
            
            print(f"Job {job.get('title', 'Unknown')}: score={relevance_score}")  # Debug print
            
            # Include job if it meets minimum relevance or has exact matches
            if relevance_score > 0.3 or len(filtered_jobs) < 5:  # Ensure at least 5 results
                job['relevance_score'] = relevance_score
                filtered_jobs.append(job)
        
        return sorted(filtered_jobs, key=lambda x: x['relevance_score'], reverse=True)

    def _get_related_titles(self, query: str) -> List[str]:
        """Get related job titles based on the query"""
        try:
            # Get base results
            results = self.retriever.search_jobs(query, {"limit": 5})
            
            # Extract unique titles
            titles = set()
            for job in results:
                # Add the main title
                titles.add(job.title)
                
                # Add related titles from structured description if available
                if hasattr(job, 'structured_description'):
                    struct_desc = job.structured_description
                    if isinstance(struct_desc, dict) and 'related_titles' in struct_desc:
                        titles.update(struct_desc['related_titles'])
            
            return list(titles)[:5]  # Return top 5 related titles
            
        except Exception as e:
            print(f"Error getting related titles: {str(e)}")
            return []

    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """Format job search results for output"""
        formatted = []
        for job in results:
            # Convert to dict if it's a SQLAlchemy model
            job_dict = dict(job) if hasattr(job, '_asdict') else job
            
            # Remove large embedding data
            job_dict.pop('description_embedding', None)
            
            # Format scores as floats
            if 'semantic_score' in job_dict:
                job_dict['relevance_score'] = float(job_dict['semantic_score'])
            if 'text_score' in job_dict:
                job_dict['text_match_score'] = float(job_dict['text_score'])
            
            formatted.append(job_dict)
        
        return formatted

    def _check_title_relevance(self, title: str, query: str) -> float:
        """Check how relevant a job title is to the search query"""
        try:
            # Clean and normalize strings
            title = title.lower()
            query = query.lower()
            
            # Split into words
            title_words = set(title.split())
            query_words = set(query.split())
            
            # Calculate overlap
            common_words = title_words.intersection(query_words)
            if not query_words:
                return 0.0
            
            # Return percentage of query words found in title
            return len(common_words) / len(query_words)
            
        except Exception as e:
            print(f"Error checking title relevance: {str(e)}")
            return 0.0

    async def profile_based_search(self, profile_data: Dict, query: str, db: Session) -> Dict:
        """Perform a search incorporating user profile data"""
        try:
            # Generate profile-specific search parameters
            search_params = self.profile_questioner.generate_search_params(profile_data)
            
            # Combine with query parameters
            combined_query = self._enhance_query_with_profile(query, search_params)
            
            # Perform search with enhanced parameters
            results = self.search(combined_query, db)
            
            # Rank results based on profile alignment
            ranked_results = self._rank_by_profile_match(results, profile_data)
            
            return {
                "results": ranked_results,
                "profile_match_details": search_params,
                "enhanced_query": combined_query
            }
        except Exception as e:
            print(f"Error in profile-based search: {str(e)}")
            return {"error": str(e), "results": []}

    def _enhance_query_with_profile(self, query: str, profile_params: Dict) -> str:
        """Enhance search query with profile parameters"""
        enhanced_query = query
        
        # Add relevant skills
        if profile_params.get('skills'):
            skills_str = " OR ".join(profile_params['skills'][:3])  # Top 3 skills
            enhanced_query += f" AND ({skills_str})"
            
        # Add work environment preferences
        if profile_params.get('work_environment'):
            enhanced_query += f" AND {profile_params['work_environment']}"
            
        return enhanced_query

    def _rank_by_profile_match(self, results: List[Dict], profile_data: Dict) -> List[Dict]:
        """Rank search results based on profile match"""
        ranked_results = []
        
        for result in results:
            print(f"\nScoring job: {result.get('title')}")
            
            # Calculate match scores with semantic similarity
            skill_match = self._calculate_skill_match(
                profile_data.get('skills', []),
                result.get('required_skills', [])
            )
            print(f"- Skill match: {skill_match:.2f}")
            
            # Extract culture from description if not explicitly provided
            job_culture = result.get('company_culture', [])
            if not job_culture:
                description = result.get('description', '')
                # Look for culture keywords in description
                culture_keywords = ['culture', 'environment', 'workplace', 'team']
                job_culture = [word for word in description.lower().split() 
                              if any(keyword in word for keyword in culture_keywords)]
            
            culture_match = self._calculate_culture_match(
                profile_data.get('work_culture', []),
                job_culture
            )
            print(f"- Culture match: {culture_match:.2f}")
            
            # Use skill similarity for core values (since we already have this method)
            values_match = 0.0
            description = result.get('description', '')
            if description and profile_data.get('core_values'):
                values_scores = [
                    self._get_skill_similarity(value, description)
                    for value in profile_data.get('core_values', [])
                ]
                values_match = sum(score for score in values_scores if score > 0.3) / len(values_scores) if values_scores else 0.0
            print(f"- Values match: {values_match:.2f}")
            
            # Weighted scoring
            match_score = (
                skill_match * 0.5 +          # 50% weight on skills
                culture_match * 0.3 +        # 30% weight on culture
                values_match * 0.2           # 20% weight on values
            )
            print(f"- Overall score: {match_score:.2f}")
            
            # Add match data to result
            result['profile_match'] = {
                'overall_score': match_score,
                'skill_match': skill_match,
                'culture_match': culture_match,
                'values_match': values_match
            }
            
            ranked_results.append(result)
        
        # Sort by match score
        return sorted(ranked_results, key=lambda x: x['profile_match']['overall_score'], reverse=True)

    def _calculate_culture_match(self, profile_culture: List[str], job_culture: List[str]) -> float:
        """Calculate culture match score"""
        if not profile_culture or not job_culture:
            return 0.0
            
        profile_culture = set(c.lower() for c in profile_culture)
        job_culture = set(c.lower() for c in job_culture)
        
        matches = len(profile_culture.intersection(job_culture))
        return matches / len(job_culture) if job_culture else 0.0

    def evaluate_recommendations(self, recommendations: List[Dict], user_profile: Dict) -> Dict:
        """Evaluate recommendations using persona-based evaluation"""
        print("\nEvaluating recommendations:")
        evaluation_results = []
        
        for rec in recommendations:
            try:
                print(f"\nEvaluating job: {rec.get('title', 'Unknown Title')}")
                
                # Create proper job_details dictionary with required fields
                job_details = {
                    'job_id': rec.get('job_id'),
                    'title': rec.get('title', ''),
                    'company_name': rec.get('company_name', ''),
                    'description': rec.get('description', ''),
                    'location': rec.get('location', ''),
                    'salary_range': f"{rec.get('min_salary', 0)}-{rec.get('max_salary', 0)}",
                    'required_skills': rec.get('match_data', {}).get('recommended_skills', []),
                    'culture_values': [],  # Add empty defaults for optional fields
                    'work_environment': []
                }
                
                # Ensure all required fields exist
                if not all(k in job_details for k in ['title', 'description']):
                    print(f"Missing required job details fields")
                    continue
                    
                # Create persona agent with validated job details
                persona = CareerPersonaAgent(job_details=job_details, llm=self.llm)
                
                # Get system score from initial matching
                system_score = rec.get('match_data', {}).get('skill_match', 0)
                
                # Get persona-based evaluation with proper user profile format
                try:
                    formatted_profile = {
                        'skills': user_profile.get('skills', []),
                        'core_values': user_profile.get('core_values', []),
                        'work_culture': user_profile.get('work_culture', []),
                        'interests': user_profile.get('additional_interests', '')
                    }
                    evaluation = persona.evaluate_profile_match(formatted_profile)
                except Exception as e:
                    print(f"Error in evaluation, using default: {str(e)}")
                    evaluation = persona._get_default_evaluation()
                
                # Calculate persona score
                persona_score = (
                    evaluation.get('skills_alignment', 0) * 0.4 +
                    evaluation.get('values_compatibility', 0) * 0.3 +
                    evaluation.get('culture_fit', 0) * 0.2 +
                    evaluation.get('growth_potential', 0) * 0.1
                ) / 10
                
                evaluation_results.append({
                    'job_id': rec['job_id'],
                    'title': rec['title'],
                    'system_score': system_score,
                    'persona_score': persona_score,
                    'detailed_evaluation': evaluation,
                    'score_delta': persona_score - system_score,
                'performance_metrics': self.performance_tracker.metrics_history
                })
                
            except Exception as e:
                print(f"Error evaluating recommendation: {str(e)}")
                continue
        
        return {
            'evaluations': evaluation_results,
            'metrics': RecommendationMetrics.aggregate_scores(
                [e['detailed_evaluation'] for e in evaluation_results]
            ) if evaluation_results else {},
            'performance_trends': self.performance_tracker.metrics_history
        }

    def get_initial_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Get initial recommendations based on hard constraints"""
        try:
            print("\n========== INITIAL RECOMMENDATIONS DEBUG ==========")
            print("1. Starting initial recommendations search")
            
            # Create search parameters from profile for sparse search
            required_terms = []
            required_terms.extend(user_profile.get('skills', []))
            required_terms.extend(user_profile.get('work_culture', []))
            required_terms.extend(user_profile.get('core_values', []))
            
            print(f"2. Using hard constraints (required terms):")
            print(f"- Skills: {user_profile.get('skills', [])}")
            print(f"- Work Culture: {user_profile.get('work_culture', [])}")
            print(f"- Core Values: {user_profile.get('core_values', [])}")
            
            # Use sparse search to filter by hard constraints
            print("\n3. Performing sparse search with hard constraints...")
            sparse_results = self.retriever.sparse_search(
                query=' '.join(required_terms),
                filters={
                    'skills': user_profile.get('skills', []),
                    'work_culture': user_profile.get('work_culture', []),
                    'core_values': user_profile.get('core_values', [])
                }
            )
            
            if not sparse_results:
                print("No results found after sparse search")
                return []
            
            print(f"4. Sparse search found {len(sparse_results)} matching jobs")
            
            # Then do semantic search on the filtered results
            print("\n5. Performing semantic search on filtered results...")
            semantic_results = self.retriever.semantic_search(
                query=' '.join(required_terms),
                filters={
                    'job_ids': [r['job_id'] for r in sparse_results]  # Use dict access
                }
            )
            
            if not semantic_results:
                print("No results found after semantic search")
                return sparse_results[:100]  # Return up to 100 sparse results if semantic fails
            
            print(f"6. Semantic search refined to {len(semantic_results)} jobs")
            
            # Rank results by profile match
            print("\n7. Ranking results by profile match...")
            ranked_results = self._rank_by_profile_match(
                semantic_results,
                {
                    'skills': user_profile.get('skills', []),
                    'work_culture': user_profile.get('work_culture', []),
                    'core_values': user_profile.get('core_values', [])
                }
            )
            
            print(f"8. Final recommendations count: {len(ranked_results)}")
            print("9. Top recommendations:")
            for i, job in enumerate(ranked_results[:5], 1):
                # Access the correct score path
                score = job.get('profile_match', {}).get('overall_score', 0)
                print(f"   {i}. {job.get('title')} - Score: {score:.2f}")
            print("=" * 50)
            
            return ranked_results
            
        except Exception as e:
            print(f"Error in initial recommendations: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return []

    def refine_recommendations(self, initial_results: List[Dict], qa_responses: List[Dict], user_profile: Dict) -> Dict:
        try:
            print("\n========== REFINING RECOMMENDATIONS ==========")
            print(f"1. Processing {len(qa_responses)} Q&A responses")
            
            # Update profile with soft constraints from Q&A
            enhanced_profile = user_profile.copy()
            for response in qa_responses:
                if response.get('required_skills'):
                    enhanced_profile.setdefault('skills', []).extend(response['required_skills'])
                if response.get('work_environment'):
                    enhanced_profile.setdefault('work_culture', []).append(response['work_environment'])
                if response.get('preferences'):
                    enhanced_profile.setdefault('preferences', []).extend(response['preferences'])
                if response.get('industry'):
                    enhanced_profile.setdefault('industries', []).append(response['industry'])

            print(f"2. Enhanced profile: {json.dumps(enhanced_profile, indent=2)}")
            
            # First use calculate_match_score for initial scoring
            scored_results = []
            for job in initial_results[:5]:  # Process top 5
                try:
                    match_score, matching_skills, matching_culture, matching_values = self.calculate_match_score(
                        job, 
                        enhanced_profile
                    )
                    
                    # Structure the job with required fields
                    scored_job = {
                        'job_id': job.get('job_id'),
                        'title': job.get('title'),
                        'company_name': job.get('company_name'),
                        'location': job.get('location', ''),
                        'description': job.get('description', ''),
                        'match_score': float(match_score),  # Ensure float type
                        'matching_skills': matching_skills,
                        'matching_culture': matching_culture,
                        'matching_values': matching_values,
                        'user_id': enhanced_profile.get('user_id'),  # Required field
                        'recommendation_type': 'refined',
                        'preference_version': 1
                    }
                    scored_results.append(scored_job)
                    
                except Exception as e:
                    print(f"Error scoring job {job.get('title')}: {str(e)}")
                    continue
            
            # Use evaluate_recommendations for additional insights
            evaluation_results = self.evaluate_recommendations(scored_results, enhanced_profile)
            print(f"3. Evaluation complete: {len(evaluation_results.get('evaluations', []))} evaluations")
            
            # Combine scores and evaluation data
            final_results = []
            for job, evaluation in zip(scored_results, evaluation_results.get('evaluations', [])):
                try:
                    detailed_eval = evaluation['detailed_evaluation']
                    
                    # Update job with evaluation data while maintaining required fields
                    recommendation = {
                        'job_id': job['job_id'],
                        'title': job['title'],
                        'company_name': job['company_name'],
                        'location': job['location'],
                        'description': job['description'],
                        'match_score': float((job['match_score'] + evaluation['persona_score']) / 2),  # Average both scores
                        'matching_skills': job['matching_skills'],
                        'matching_culture': job['matching_culture'],
                        'user_id': job['user_id'],
                        'recommendation_type': 'refined',
                        'preference_version': 1,
                        'evaluation_data': {
                            'skills_alignment': detailed_eval.get('skills_alignment', 0),
                            'values_compatibility': detailed_eval.get('values_compatibility', 0),
                            'culture_fit': detailed_eval.get('culture_fit', 0),
                            'growth_potential': detailed_eval.get('growth_potential', 0),
                            'reasoning': detailed_eval.get('reasoning', {}),
                            'skill_gaps': detailed_eval.get('skill_gaps', []),
                            'culture_fit_details': detailed_eval.get('culture_fit_details', []),
                            'system_score': job['match_score'],
                            'persona_score': evaluation['persona_score'],
                            'score_delta': evaluation['score_delta']
                        }
                    }
                    final_results.append(recommendation)
                    
                except Exception as e:
                    print(f"Error processing evaluation for {job['title']}: {str(e)}")
                    continue
            
            print(f"\n4. Processed {len(final_results)} final recommendations")
            
            # Generate and save performance plots
            performance_data = {
                'metrics_history': self.performance_tracker.metrics_history,
                'timestamp': self.performance_tracker.timestamp
            }
            
            return {
                'recommendations': final_results,
                'evaluation': {
                    'qa_responses': qa_responses,
                    'enhanced_profile': enhanced_profile,
                    'metrics': evaluation_results.get('metrics', {}),
                    'performance_trends': evaluation_results.get('performance_trends', []),
                    'performance_data': performance_data
                }
            }
                    
        except Exception as e:
            print(f"Error in refining recommendations: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'recommendations': [],
                'evaluation': {}
            }
    
    def store_recommendations(self, recommendations: List[Dict], chat_session_id: str, user_id: int, recommendation_type: str = 'initial'):
        """Store job recommendations for a user session"""
        try:
            print("\n=== STORE RECOMMENDATIONS DEBUG ===")
            print(f"Processing {len(recommendations)} recommendations for session {chat_session_id}")
            
            # First verify the user exists
            user_profile = self.db.query(UserProfile).filter(UserProfile.id == user_id).first()
            if not user_profile:
                print(f"Error: No user profile found for user_id {user_id}")
                return []
            
            stored_recs = []
            for rec in recommendations:
                try:
                    print(f"\nProcessing recommendation for job: {rec.get('title', 'Unknown Title')}")
                    
                    # Create a dictionary with only the fields needed by create_job_recommendation
                    job_data = {
                        # Remove db from job_data as it's a separate parameter
                        'user_id': user_id,
                        'chat_session_id': chat_session_id,
                        'job_id': rec.get('job_id'),
                        'title': rec.get('title'),
                        'company_name': rec.get('company_name'),
                        'location': rec.get('location'),
                        'match_score': float(rec.get('match_score', 0.0)),
                        'matching_skills': rec.get('matching_skills', []),
                        'matching_culture': rec.get('matching_culture', []),
                        'evaluation_data': rec.get('evaluation_data', {}),
                        'recommendation_type': recommendation_type,
                        'preference_version': 1
                    }
                    
                    print(f"Creating job recommendation with data: {json.dumps(job_data, indent=2)}")
                    
                    # Pass db separately from job_data
                    job_rec = create_job_recommendation(
                        db=self.db,  # Pass db separately
                        **job_data   # Unpack the rest of the fields
                    )

                    stored_rec = {
                        'id': job_rec.id,
                        'job_id': job_rec.job_id,
                        'title': job_rec.title,
                        'company_name': job_rec.company_name,
                        'location': job_rec.location,
                        'match_score': float(job_rec.match_score) if job_rec.match_score else 0.0,
                        'matching_skills': job_rec.matching_skills,
                        'matching_culture': job_rec.matching_culture,
                        'evaluation_data': job_rec.evaluation_data,
                        'recommendation_type': job_rec.recommendation_type,
                        'preference_version': job_rec.preference_version,
                        'created_at': str(job_rec.created_at),
                        'user_id': job_rec.user_id,
                        'chat_session_id': job_rec.chat_session_id
                    }
                
                    stored_recs.append(stored_rec)
                    print(f"Successfully stored recommendation for job: {job_data['title']}")

                except Exception as e:
                    print(f"Error processing recommendation: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    continue

            print(f"\nFinished processing: Successfully stored {len(stored_recs)} of {len(recommendations)} recommendations")
            return stored_recs
                        
        except Exception as e:
            print(f"Error in store_recommendations: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.db.rollback()
            return []
        
    def get_user_recommendations(self, user_id: int, limit: int = 10):
        """Get a user's recommendations across all sessions"""
        return self.db.query(JobRecommendation)\
            .filter(JobRecommendation.user_id == user_id)\
            .order_by(JobRecommendation.created_at.desc())\
            .limit(limit)\
            .all()

    def get_session_recommendations(self, chat_session_id: str):
        """Get recommendations for a specific chat session"""
        return self.db.query(JobRecommendation)\
            .filter(JobRecommendation.chat_session_id == chat_session_id)\
            .order_by(JobRecommendation.created_at.desc())\
            .all()

    def _analyze_job_patterns(self, job_recommendations: List[Dict]) -> Dict:
        """Analyze job recommendations to identify patterns and trends"""
        try:
            print("\n=== JOB PATTERN ANALYSIS DEBUG ===")
            print(f"Analyzing {len(job_recommendations)} job recommendations...")
            
            # Initialize pattern tracking
            industries = {}
            roles = {}
            skill_counts = {}
            companies = {}
            locations = {}
            
            for idx, job in enumerate(job_recommendations):
                description = job.get('description', '')
                title = job.get('title', 'No Title')
                company = job.get('company_name', 'No Company')
                location = job.get('location', 'Unknown')

                # Extract industry and skills using semantic analysis
                extracted_industry = self._extract_industry_semantic(description)
                extracted_skills = self._extract_skills_semantic(description)
                
                # Track patterns
                industries[extracted_industry] = industries.get(extracted_industry, 0) + 1
                for skill in extracted_skills:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
                companies[company] = companies.get(company, 0) + 1
                locations[location] = locations.get(location, 0) + 1
                
                print(f"\nProcessing job {idx + 1}:")
                print(f"- Title: {title}")
                print(f"- Company: {company}")
                print(f"- Industry: {extracted_industry}")
                print(f"- Location: {location}")
                print(f"- Extracted Skills: {extracted_skills}")

            return {
                'top_industries': self._get_top_items(industries, 5),
                'top_roles': self._get_top_items(roles, 10),
                'common_skills': self._get_top_items(skill_counts, 10),
                'top_companies': self._get_top_items(companies, 5),
                'top_locations': self._get_top_items(locations, 5),
                'total_jobs': len(job_recommendations)
            }

        except Exception as e:
            print(f"Error in job pattern analysis: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {}

    def _extract_industry_semantic(self, description: str) -> str:
        """Extract industry using semantic analysis of job description"""
        # Common industry signals in descriptions
        industry_signals = {
            'Healthcare': ['patient', 'clinical', 'medical', 'health', 'hospital', 'care'],
            'Technology': ['software', 'data', 'digital', 'tech', 'IT', 'computer', 'programming'],
            'Manufacturing': ['manufacturing', 'production', 'assembly', 'factory', 'industrial'],
            'Finance': ['financial', 'banking', 'investment', 'trading', 'finance'],
            'Education': ['education', 'teaching', 'academic', 'school', 'university', 'learning'],
            'Retail': ['retail', 'store', 'merchandising', 'e-commerce', 'consumer'],
            'Environmental': ['environmental', 'sustainability', 'renewable', 'recycling', 'waste'],
            'Aerospace': ['aerospace', 'aviation', 'aircraft', 'flight', 'space'],
            'Marketing': ['marketing', 'advertising', 'brand', 'campaign', 'media'],
            'Consulting': ['consulting', 'advisory', 'professional services', 'solutions']
        }
        
        description_lower = description.lower()
        industry_scores = {}
        
        for industry, signals in industry_signals.items():
            score = sum(signal.lower() in description_lower for signal in signals)
            if score > 0:
                industry_scores[industry] = score
        
        if industry_scores:
            return max(industry_scores.items(), key=lambda x: x[1])[0]
        return "Unknown"

    def _extract_skills_semantic(self, description: str) -> List[str]:
        """Extract skills using semantic analysis of job description"""
        # Common skill keywords and phrases
        skill_patterns = [
            # Technical Skills
            r'\b(python|java|javascript|sql|aws|azure|cloud)\b',
            r'\b(machine learning|ai|artificial intelligence|data science)\b',
            r'\b(agile|scrum|waterfall|kanban)\b',
            
            # Business Skills
            r'\b(project management|program management|team leadership)\b',
            r'\b(strategic planning|business development|analytics)\b',
            r'\b(marketing automation|digital marketing|content strategy)\b',
            
            # Soft Skills
            r'\b(communication skills|leadership|problem solving)\b',
            r'\b(team collaboration|interpersonal skills|presentation skills)\b',
            
            # Tools & Platforms
            r'\b(salesforce|jira|confluence|git|microsoft office)\b',
            r'\b(adobe|photoshop|illustrator|indesign)\b'
        ]
        
        extracted_skills = set()
        description_lower = description.lower()
        
        import re
        for pattern in skill_patterns:
            matches = re.findall(pattern, description_lower)
            extracted_skills.update(matches)
        
        # Look for skills in requirements/qualifications section
        req_section = self._extract_requirements_section(description)
        if req_section:
            for pattern in skill_patterns:
                matches = re.findall(pattern, req_section.lower())
                extracted_skills.update(matches)
        
        return list(extracted_skills)

    def _extract_requirements_section(self, description: str) -> str:
        """Extract the requirements/qualifications section from job description"""
        requirement_headers = [
            'requirements', 'qualifications', 'what you need', 
            'what we\'re looking for', 'skills', 'experience required'
        ]
        
        description_lower = description.lower()
        for header in requirement_headers:
            if header in description_lower:
                start_idx = description_lower.find(header)
                # Try to find the next section header or end of text
                next_section = float('inf')
                for next_header in ['responsibilities', 'about us', 'what we offer']:
                    idx = description_lower.find(next_header, start_idx + len(header))
                    if idx != -1:
                        next_section = min(next_section, idx)
                
                if next_section == float('inf'):
                    return description[start_idx:]
                return description[start_idx:next_section]
        
        return ""

    def _get_top_items(self, items_dict: Dict, limit: int) -> List[str]:
        """Helper to get top items from a frequency dictionary"""
        return [item for item, _ in sorted(items_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:limit]]

    def calculate_match_score(self, job: Dict, user_profile: Dict) -> Tuple[float, List[str], List[str], List[str]]:
        """Calculate comprehensive match score including skills, culture and values"""
        print("\n=== Calculate Match Score Debug ===")
        print(f"Job Title: {job.get('title')}")
        print(f"User Skills: {user_profile.get('skills', [])}")
        print(f"User Culture: {user_profile.get('work_culture', [])}")
        print(f"User Values: {user_profile.get('core_values', [])}")
        
        # Calculate skill match
        skill_match = self._calculate_skill_match(
            user_profile.get('skills', []),
            job.get('required_skills', [])
        )
        print(f"Initial Skill Match Score: {skill_match}")
        
        # Find matching elements with similarity threshold
        matching_skills = [
            skill for skill in user_profile.get('skills', [])
            if self._get_skill_similarity(skill, job.get('description', '')) > 0.3
        ]
        print(f"Matching Skills Found: {matching_skills}")
        
        matching_culture = [
            culture for culture in user_profile.get('work_culture', [])
            if self._get_skill_similarity(culture, job.get('description', '')) > 0.3
        ]
        print(f"Matching Culture Found: {matching_culture}")
        
        matching_values = [
            value for value in user_profile.get('core_values', [])
            if self._get_skill_similarity(value, job.get('description', '')) > 0.3
        ]
        print(f"Matching Values Found: {matching_values}")
        
        # Calculate component scores
        user_skills = user_profile.get('skills', [])
        user_culture = user_profile.get('work_culture', [])
        user_values = user_profile.get('core_values', [])
        
        skills_score = skill_match if skill_match > 0 else (len(matching_skills) / len(user_skills) if user_skills else 0)
        culture_score = len(matching_culture) / len(user_culture) if user_culture else 0
        values_score = len(matching_values) / len(user_values) if user_values else 0
        
        print(f"Component Scores:")
        print(f"- Skills Score: {skills_score}")
        print(f"- Culture Score: {culture_score}")
        print(f"- Values Score: {values_score}")
        
        # Final weighted score
        final_score = (
            skills_score * 0.5 +      # 50% skills
            culture_score * 0.3 +     # 30% culture
            values_score * 0.2        # 20% values
        )
        print(f"Final Score: {final_score}")
        print("=" * 50)
        
        return final_score, matching_skills, matching_culture, matching_values

    def extract_skills_from_description(self, description: str) -> List[str]:
        """Extract likely skills from job description"""
        # Split description into sentences
        sentences = description.lower().split('.')
        
        # Common skill-related phrases
        skill_indicators = ['experience with', 'knowledge of', 'proficiency in', 
                           'skills in', 'expertise in', 'familiar with']
        
        extracted_skills = []
        for sentence in sentences:
            for indicator in skill_indicators:
                if indicator in sentence:
                    # Extract the part after the indicator
                    skills_part = sentence.split(indicator)[1]
                    # Split by common separators
                    skills = [s.strip() for s in skills_part.split(',')]
                    extracted_skills.extend(skills)
        
        return list(set(extracted_skills))  # Remove duplicates

    def format_job_recommendation(self, job: Dict) -> str:
        """Format a single job recommendation with evaluation insights"""
        match_data = job.get('profile_match', {})
        eval_details = match_data.get('evaluation_details', {})
        
        return f"""
Job: {job.get('title')}
Company: {job.get('company_name')}
Location: {job.get('location')}
Match Score: {match_data.get('overall_score', 0):.2f}

Fit Analysis:
- Skills Match: {match_data.get('skill_match', 0):.2f}
- Culture Match: {match_data.get('culture_match', 0):.2f}

Key Insights:
{eval_details.get('reasoning', {}).get('skills', '')}

Growth Potential:
{eval_details.get('reasoning', {}).get('growth', '')}

Areas for Development:
- {', '.join(eval_details.get('skill_gaps', ['None identified']))}

------------------------------
"""

    def _extract_skills_from_description(self, description: str) -> List[str]:
        """Extract skills from job description"""
        skills = []
        
        # Split description into sections
        sections = description.split('\n')
        in_skills_section = False
        current_section = ""
        
        # Common section headers that indicate skills
        skill_headers = [
            "minimum qualifications",
            "qualifications",
            "requirements",
            "skills",
            "experience",
            "technical/soft skills"
        ]
        
        # End markers for skills sections
        end_markers = [
            "preferred qualifications",
            "additional notes",
            "benefits",
            "salary range",
            "about"
        ]
        
        for line in sections:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Check if we're entering a skills section
            if any(header in line_lower for header in skill_headers):
                in_skills_section = True
                continue
            
            # Check if we're leaving a skills section
            if any(marker in line_lower for marker in end_markers):
                in_skills_section = False
                continue
            
            if in_skills_section and line:
                # Remove bullet points and numbers
                skill = re.sub(r'^[\u2022\-\*\d.]+\s*', '', line)
                
                # Split on periods if the line contains multiple sentences
                for part in skill.split('.'):
                    part = part.strip()
                    if len(part) <= 3:
                        continue
                    
                    # Clean up common prefixes
                    part = re.sub(r'^(Experience|Knowledge of|Ability to|Strong|Advanced|Proven|Demonstrated)\s+', '', part, flags=re.IGNORECASE)
                    
                    # Split on common separators
                    for subpart in re.split(r'[,;]|\sand\s', part):
                        subpart = subpart.strip()
                        if len(subpart) > 3:
                            skills.append(subpart)
        
        # Remove duplicates while preserving order
        seen = set()
        cleaned_skills = [x for x in skills if not (x in seen or seen.add(x))]
        
        return cleaned_skills

    def _extract_industry_from_description(self, description: str, company_name: str) -> str:
        """Extract industry from job description and company name"""
        industry_map = {
            "healthcare": "Healthcare",
            "cancer": "Healthcare",
            "medical": "Healthcare",
            "manufacturing": "Manufacturing",
            "materials science": "Manufacturing",
            "chemical": "Manufacturing",
            "technology": "Technology",
            "software": "Technology",
            "financial": "Financial Services",
            "accounting": "Financial Services",
            "tax": "Financial Services"
        }
        
        # First check company description (usually in first paragraph)
        first_para = description.lower().split('\n\n')[0]
        for keyword, industry in industry_map.items():
            if keyword in first_para:
                return industry
        
        # Then check full description
        desc_lower = description.lower()
        for keyword, industry in industry_map.items():
            if keyword in desc_lower:
                return industry
            
        return "Unknown"

# @lru_cache()
# def get_search_service(db: Session):
#     """Initialize and return a JobSearchService instance"""
#     settings = get_settings()
    
#     # Initialize the model using the name from settings
#     embed_model = SentenceTransformer(settings.embed_model_name)
    
#     # Initialize the retriever with the model instance
#     retriever = JobSearchRetriever(
#         db=db,
#         embed_model=embed_model,  # Pass the initialized model
#         strategy=SearchStrategy.SEMANTIC
#     )
    
#     # Create and return the search service
#     return JobSearchService(retriever=retriever, settings=settings, db=db)