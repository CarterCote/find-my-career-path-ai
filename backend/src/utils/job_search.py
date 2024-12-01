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
from ..crud import create_chat_message, create_job_recommendation
from ..models import UserProfile, JobRecommendation, ChatHistory, CareerRecommendation
from .job_retriever import JobSearchRetriever, SearchStrategy
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from ..config import get_settings
from sentence_transformers import SentenceTransformer
from .profile_questioner import ProfileQuestioner
from ..prompts.job_search_prompts import SYSTEM_PROMPT, FOLLOWUP_PROMPT
from ..prompts.search_prompts import SEARCH_ENHANCEMENT
from ..evaluation.persona_agent import CareerPersonaAgent
from ..evaluation.metrics import RecommendationMetrics

import json
import torch

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
        
        # Add a state tracker for question-answer flow
        self.qa_state = {}
        print("4. JobSearchService initialization complete")
        
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
        """Handle chat interactions including Q&A flow"""
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
                print("4. Getting initial recommendations...")
                initial_recommendations = self.get_initial_recommendations({
                    'core_values': user_profile.core_values or [],
                    'work_culture': user_profile.work_culture or [],
                    'skills': user_profile.skills or []
                })
                
                if not initial_recommendations:
                    print("Error: No initial recommendations found")
                    return {
                        "response": "No matching jobs found. Please adjust your profile criteria.",
                        "chat_session_id": chat_session_id
                    }
                
                print(f"5. Found {len(initial_recommendations)} initial recommendations")
                print("6. Generating questions...")
                
                # Generate questions based on profile and initial recommendations
                try:
                    questions = self.profile_questioner.generate_questions({
                        'core_values': user_profile.core_values or [],
                        'work_culture': user_profile.work_culture or [],
                        'skills': user_profile.skills or [],
                        'additional_interests': user_profile.additional_interests or '',
                        'initial_recommendations': initial_recommendations[:5]  # Use top 5 for question generation
                    })
                    
                    state = {
                        "questions": questions,
                        "answers": [],
                        "current_index": 0,
                        "complete": False,
                        "initial_recommendations": initial_recommendations
                    }
                    self.qa_state[chat_session_id] = state
                    
                    print(f"7. Generated {len(questions)} questions")
                    return {
                        "response": questions[0],
                        "chat_session_id": chat_session_id
                    }
                    
                except Exception as e:
                    print(f"Error generating questions: {str(e)}")
                    return {
                        "response": "I'm having trouble generating questions. Let's start with: What type of work environment do you prefer?",
                        "chat_session_id": chat_session_id
                    }

            # Handle Q&A completion and responses
            if state and not state.get("complete"):
                # Store the current answer
                if state.get("current_index") is not None:
                    state["answers"].append({
                        "question": state["questions"][state["current_index"]],
                        "answer": message
                    })
                    
                # Check if we've completed all questions
                if len(state["answers"]) >= len(state["questions"]):
                    print("8. Q&A complete, refining recommendations...")
                    
                    # Process all Q&A responses
                    processed_responses = []
                    print("\nStarting Q&A Processing...")
                    for qa in state["answers"]:
                        print(f"\nProcessing Q&A pair:")
                        print(f"Question: {qa['question']}")
                        print(f"Answer: {qa['answer']}")
                        try:
                            processed = self.profile_questioner.process_response(
                                qa["question"], 
                                qa["answer"]
                            )
                            if processed:
                                processed_responses.append(processed)
                                print(f"Successfully added processed response: {json.dumps(processed, indent=2)}")
                            else:
                                print("Warning: Received empty processed response")
                        except Exception as e:
                            print(f"Error processing response: {str(e)}")
                            print(f"Error type: {type(e)}")
                            import traceback
                            print(f"Traceback: {traceback.format_exc()}")

                    print(f"\nFinal processed responses: {json.dumps(processed_responses, indent=2)}")
                    
                    # Get refined recommendations
                    refined_results = self.refine_recommendations(
                        state["initial_recommendations"],
                        processed_responses,
                        {
                            'core_values': user_profile.core_values or [],
                            'work_culture': user_profile.work_culture or [],
                            'skills': user_profile.skills or [],
                            'additional_interests': user_profile.additional_interests or ''
                        }
                    )
                    
                    # Store recommendations and mark state as complete
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
                            "response": "Q&A session complete! I've analyzed your answers and found some great matches. Let me show you the recommendations.",
                            "chat_session_id": chat_session_id,
                            "qa_complete": True,
                            "recommendations": top_jobs,
                            "evaluation": refined_results.get('evaluation', {})
                        }
                
                # If not complete, move to next question
                else:
                    state["current_index"] += 1
                    next_question = state["questions"][state["current_index"]]
                    return {
                        "response": next_question,
                        "chat_session_id": chat_session_id
                    }

            # Regular chat after Q&A
            print("10. Regular chat mode")
            return self._handle_regular_chat(message, chat_session_id)
                
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
            # Calculate match scores
            skill_match = self._calculate_skill_match(
                profile_data.get('skills', []),
                result.get('required_skills', [])
            )
            
            culture_match = self._calculate_culture_match(
                profile_data.get('work_culture', []),
                result.get('company_culture', [])
            )
            
            # Combine scores
            match_score = (skill_match * 0.6) + (culture_match * 0.4)
            
            # Add match data to result
            result['profile_match'] = {
                'overall_score': match_score,
                'skill_match': skill_match,
                'culture_match': culture_match
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
                    'score_delta': persona_score - system_score
                })
                
            except Exception as e:
                print(f"Error evaluating recommendation: {str(e)}")
                continue
        
        return {
            'evaluations': evaluation_results,
            'metrics': RecommendationMetrics.aggregate_scores(
                [e['detailed_evaluation'] for e in evaluation_results]
            ) if evaluation_results else {}
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
                print(f"   {i}. {job.get('title')} - Score: {job.get('match_score', 0):.2f}")
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
                for key, value in response.items():
                    if key in enhanced_profile and isinstance(enhanced_profile[key], list):
                        if isinstance(value, list):
                            enhanced_profile[key].extend(value)
                        else:
                            enhanced_profile[key].append(value)
                    else:
                        enhanced_profile[key] = value

            # Re-rank initial results with enhanced profile
            refined_results = self._rank_by_profile_match(
                initial_results,
                enhanced_profile
            )
            
            # Select top 5 BEFORE evaluation
            top_5_results = refined_results[:5]
            
            # Only evaluate the top 5
            evaluation_results = self.evaluate_recommendations(
                top_5_results,  # Only evaluate top 5
                enhanced_profile
            )
            
            # Format recommendations with evaluation insights
            formatted_recommendations = [
                self.format_job_recommendation(rec) 
                for rec in top_5_results
            ]
            
            return {
                'recommendations': top_5_results,
                'formatted_recommendations': formatted_recommendations,
                'evaluation': evaluation_results
            }
                
        except Exception as e:
            print(f"Error in refining recommendations: {str(e)}")
            return {
                'recommendations': initial_results[:5],
                'evaluation': {}
            }

    def store_recommendations(self, recommendations: List[Dict], chat_session_id: str, user_id: int, recommendation_type: str = 'refined'):
        """Store job recommendations for a user session"""
        try:
            # First verify the user exists (KEEP THIS GUARDRAIL)
            user_profile = self.db.query(UserProfile).filter(UserProfile.id == user_id).first()
            if not user_profile:
                print(f"Error: No user profile found for user_id {user_id}")
                return []

            stored_recs = []
            for rec in recommendations:
                try:
                    # KEEP THE SOPHISTICATED MATCH SCORE LOGIC
                    match_score = (
                        rec.get('profile_match', {}).get('overall_score', 0.0)
                        or rec.get('match_data', {}).get('skill_match', 0.0)
                        or rec.get('relevance_score', 0.0)
                        or rec.get('match_score', 0.0)
                    )
                    
                    # KEEP THE EVALUATION DATA PROCESSING
                    profile_match = rec.get('profile_match', {})
                    evaluation_data = {
                        'skills_alignment': profile_match.get('skill_match', 0.0) * 10,
                        'values_compatibility': profile_match.get('culture_match', 0.0) * 10,
                        'culture_fit': profile_match.get('culture_match', 0.0) * 10,
                        'growth_potential': profile_match.get('growth_potential', 0.0) * 10,
                        'skill_gaps': profile_match.get('evaluation_details', {}).get('skill_gaps', []),
                        'culture_fit_details': profile_match.get('evaluation_details', {}).get('culture_fit_details', []),
                        'reasoning': profile_match.get('evaluation_details', {}).get('reasoning', {})
                    }

                    try:
                        # Use CRUD but within our error handling
                        job_rec = create_job_recommendation(
                            db=self.db,
                            user_id=user_id,
                            chat_session_id=chat_session_id,
                            job_id=rec.get('job_id'),
                            title=rec.get('title'),
                            company_name=rec.get('company_name'),
                            match_score=float(match_score),
                            matching_skills=profile_match.get('evaluation_details', {}).get('skill_gaps', []),
                            matching_culture=profile_match.get('evaluation_details', {}).get('culture_fit_details', []),
                            location=rec.get('location'),
                            recommendation_type=recommendation_type,
                            evaluation_data=evaluation_data,
                            preference_version=1
                        )
                        stored_recs.append(job_rec)
                    except Exception as e:
                        print(f"Error creating individual recommendation: {str(e)}")
                        # Continue with next recommendation instead of failing entire batch
                        continue

                except Exception as e:
                    print(f"Error processing recommendation data: {str(e)}")
                    continue

            if stored_recs:
                print(f"Successfully stored {len(stored_recs)} recommendations for user {user_id}, chat_session {chat_session_id}")
            else:
                print("Warning: No recommendations were successfully stored")
                
            return stored_recs
                    
        except Exception as e:
            print(f"Error in store_recommendations: {str(e)}")
            self.db.rollback()  # Rollback any partial changes
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

    def generate_career_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Generate career path recommendations based on profile and job clusters"""
        try:
            # Get job recommendations first
            job_recs = self.get_initial_recommendations(user_profile)
            
            # Cluster similar jobs to identify career paths
            career_clusters = self._cluster_jobs_into_careers(job_recs)
            
            # Generate career recommendations
            career_recs = []
            for cluster in career_clusters:
                career_rec = CareerRecommendation(
                    chat_session_id=user_profile['chat_session_id'],  # Changed from session_id
                    career_title=cluster['career_title'],
                    career_field=cluster['field'],
                    reasoning=self._generate_career_reasoning(cluster, user_profile),
                    skills_required=cluster['common_skills'],
                    growth_potential=cluster['progression_path'],
                    match_score=cluster['match_score']
                )
                # Link example jobs
                career_rec.example_jobs = self._get_top_examples(cluster['jobs'], 3)
                career_recs.append(career_rec)
            
            return career_recs
        except Exception as e:
            print(f"Error generating career recommendations: {str(e)}")
            return []

    def calculate_match_score(self, job: Dict, user_profile: UserProfile) -> Tuple[float, List[str], List[str], List[str]]:
        """Calculate comprehensive match score including skills, culture and values"""
        print("\n=== Calculate Match Score Debug ===")
        print(f"Job Title: {job.get('title')}")
        print(f"User Skills: {user_profile.skills}")
        print(f"User Culture: {user_profile.work_culture}")
        print(f"User Values: {user_profile.core_values}")
        
        # Calculate skill match
        skill_match = self._calculate_skill_match(
            user_profile.skills,
            job.get('required_skills', [])
        )
        print(f"Initial Skill Match Score: {skill_match}")
        
        # Find matching elements with similarity threshold
        matching_skills = [
            skill for skill in user_profile.skills 
            if self._get_skill_similarity(skill, job.get('description', '')) > 0.3
        ]
        print(f"Matching Skills Found: {matching_skills}")
        
        matching_culture = [
            culture for culture in user_profile.work_culture 
            if self._get_skill_similarity(culture, job.get('description', '')) > 0.3
        ]
        print(f"Matching Culture Found: {matching_culture}")
        
        matching_values = [
            value for value in user_profile.core_values 
            if self._get_skill_similarity(value, job.get('description', '')) > 0.3
        ]
        print(f"Matching Values Found: {matching_values}")
        
        # Calculate component scores
        skills_score = skill_match if skill_match > 0 else len(matching_skills) / len(user_profile.skills)
        culture_score = len(matching_culture) / len(user_profile.work_culture) if user_profile.work_culture else 0
        values_score = len(matching_values) / len(user_profile.core_values) if user_profile.core_values else 0
        
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