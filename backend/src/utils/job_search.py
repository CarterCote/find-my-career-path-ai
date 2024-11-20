from logging import Logger
from typing import Dict, List, Optional
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
from ..crud import create_chat_message
from ..models import UserProfile
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

class JobSearchService:
    def __init__(self, retriever, settings, db: Session):
        print("\n========== JOB SEARCH SERVICE INIT DEBUG ==========")
        print("1. Initializing JobSearchService")
        self.retriever = retriever
        self.db = db
        self.store = {}
        
        # Initialize OpenAI chat
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

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def chat(self, message: str, session_id: str = "default") -> Dict:
        """Handle chat interactions including Q&A flow"""
        try:
            print("\n========== JOB SEARCH SERVICE CHAT DEBUG ==========")
            print(f"1. Chat method called with message: {message}")
            
            state = self.qa_state.get(session_id)
            
            # Initialize state if first message
            if not state:
                user_profile = self.db.query(UserProfile).filter(
                    UserProfile.session_id == session_id
                ).first()
                
                if not user_profile:
                    return {
                        "response": "Please create a profile first",
                        "session_id": session_id
                    }
                
                # Generate questions based on profile
                questions = self.profile_questioner.generate_questions({
                    'core_values': user_profile.core_values or [],
                    'work_culture': user_profile.work_culture or [],
                    'skills': user_profile.skills or [],
                    'additional_interests': user_profile.additional_interests or ''
                })
                
                state = {
                    "questions": questions,
                    "answers": [],
                    "current_index": 0,
                    "complete": False
                }
                self.qa_state[session_id] = state
                
                print("Debug - Initialized new Q&A session")
                print(f"Debug - First question: {questions[0]}")
                
                return {
                    "response": questions[0],
                    "session_id": session_id
                }
            
            # Handle ongoing Q&A
            if not state["complete"]:
                # Store answer
                state["answers"].append({
                    "question": state["questions"][state["current_index"]],
                    "answer": message
                })
                print(f"Debug - Stored answer {len(state['answers'])} of {len(state['questions'])}")
                
                state["current_index"] += 1
                
                # More questions to ask
                if state["current_index"] < len(state["questions"]):
                    next_question = state["questions"][state["current_index"]]
                    print(f"Debug - Asking question {state['current_index'] + 1} of {len(state['questions'])}")
                    return {
                        "response": next_question,
                        "session_id": session_id
                    }
                
                # Q&A complete, update profile
                state["complete"] = True
                print("Debug - Q&A complete, updating profile")
                
                qa_summary = "; ".join([
                    f"Q: {qa['question']} A: {qa['answer']}"
                    for qa in state["answers"]
                ])
                
                try:
                    # Update user profile
                    user_profile = self.db.query(UserProfile).filter(
                        UserProfile.session_id == session_id
                    ).first()
                    user_profile.additional_interests = qa_summary
                    self.db.commit()
                    print("Debug - Updated user profile with Q&A summary")
                    
                    return {
                        "response": "Thank you for answering all questions! You can now get personalized job recommendations using /users/recommendations/{session_id}",
                        "session_id": session_id,
                        "qa_complete": True
                    }
                    
                except Exception as e:
                    print(f"Error updating profile: {str(e)}")
                    return {
                        "response": "Error updating profile with your answers.",
                        "session_id": session_id
                    }
            
            # Regular chat after Q&A
            return self._handle_regular_chat(message, session_id)
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return {
                "response": "I encountered an error. Please try again.",
                "session_id": session_id
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

    def search(self, query: str, db: Session, session_id: str = "default") -> Dict:
        try:
            # Get or create chat history
            history = self.get_session_history(session_id)
            
            # Store user message in both memory and database
            history.add_user_message(query)
            create_chat_message(db, session_id, query, is_user=True)
            
            # Get job recommendations using graph-enhanced search
            self.retriever.strategy = SearchStrategy.GRAPH
            results = self.retriever.search_jobs(query)
            
            # Get user profile once for both filtering and personalization
            user_profile = db.query(UserProfile).filter(
                UserProfile.session_id == session_id
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
                    'session_id': session_id,
                    'user_context': user_context,
                    'evaluation': evaluation_results
                }
            
            return {
                "jobs": formatted_results,
                "session_id": session_id,
                "user_context": user_context if user_profile else None,
                "evaluation_results": {}
            }
                
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {
                "error": str(e),
                "jobs": []
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
        """Calculate match score between user skills and job requirements"""
        if not user_skills or not required_skills:
            return 0.0
            
        # Convert to sets for comparison
        user_set = {skill.lower() for skill in user_skills}
        required_set = {skill.lower() for skill in required_skills}
        
        # Calculate direct matches
        direct_matches = len(user_set.intersection(required_set))
        
        # Calculate semantic similarity for non-exact matches
        semantic_matches = 0
        for user_skill in user_set:
            if user_skill not in required_set:
                # Use embedding similarity to find related skills
                similarity = self._get_skill_similarity(user_skill, required_set)
                semantic_matches += similarity
        
        total_score = (direct_matches + semantic_matches) / len(required_set)
        return min(1.0, total_score)  # Normalize to 0-1

    def filter_irrelevant_jobs(self, jobs: List[Dict], query: str, user_profile: Optional[UserProfile] = None) -> List[Dict]:
        """Filter out irrelevant jobs based on query and user preferences"""
        filtered_jobs = []
        
        print(f"Filtering {len(jobs)} jobs")  # Debug print
        
        # Get query embeddings for semantic matching
        query_embedding = self.retriever.embed_model.encode([query])[0]
        
        for job in jobs:
            relevance_score = 0
            
            # 1. Title relevance (30% weight)
            title_relevance = self._check_title_relevance(job.get('title', ''), query)
            relevance_score += title_relevance * 0.3
            
            # 2. Skill match if user profile exists (30% weight)
            if user_profile and user_profile.skills:
                skill_match = self._calculate_skill_match(
                    user_profile.skills,
                    job.get('required_skills', [])
                )
                relevance_score += skill_match * 0.3
            else:
                # If no user profile, increase weight of other factors
                relevance_score += 0.3  # Add default score
            
            # 3. Semantic relevance (40% weight)
            if 'description_embedding' in job:
                semantic_score = cosine_similarity(
                    [query_embedding],
                    [job['description_embedding']]
                )[0][0]
                relevance_score += semantic_score * 0.4
            else:
                # If no embedding, increase weight of other factors
                relevance_score += 0.4  # Add default score
            
            print(f"Job {job.get('title', 'Unknown')}: score={relevance_score}")  # Debug print
            
            # Lower threshold and always include some results
            if relevance_score > 0.3 or len(filtered_jobs) < 5:  # Ensure at least 5 results
                job['relevance_score'] = relevance_score
                filtered_jobs.append(job)
        
        print(f"Filtered to {len(filtered_jobs)} jobs")  # Debug print
        
        # Sort by relevance score
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

    def _calculate_skill_match(self, profile_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill match score"""
        if not profile_skills or not job_skills:
            return 0.0
            
        profile_skills = set(s.lower() for s in profile_skills)
        job_skills = set(s.lower() for s in job_skills)
        
        matches = len(profile_skills.intersection(job_skills))
        return matches / len(job_skills) if job_skills else 0.0

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
        evaluation_results = []
        
        for rec in recommendations:
            try:
                # Create persona agent for this job
                agent = CareerPersonaAgent(rec, self.llm)
                
                # Get evaluation from persona perspective
                evaluation = agent.evaluate_profile_match(user_profile)
                
                # Compare with system's match score
                system_score = rec.get('match_score', 0)
                
                # Calculate persona score from evaluation metrics
                scores = [
                    evaluation.get('skills_alignment', 0),
                    evaluation.get('values_compatibility', 0),
                    evaluation.get('culture_fit', 0),
                    evaluation.get('growth_potential', 0)
                ]
                persona_score = sum(scores) / (len(scores) * 10)  # Convert 0-10 scale to 0-1
                
                evaluation_results.append({
                    'job_id': rec.get('job_id', ''),
                    'title': rec.get('title', ''),
                    'system_score': system_score,
                    'persona_score': persona_score,
                    'detailed_evaluation': evaluation,
                    'score_delta': abs(system_score - persona_score)
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