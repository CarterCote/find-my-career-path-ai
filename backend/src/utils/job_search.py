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

import json

class JobSearchService:
    def __init__(self, retriever, settings):
        self.retriever = retriever

        self.store = {}
        
        # Initialize OpenAI chat
        self.llm = ChatOpenAI(
            temperature=0.7,  # Increased for more natural conversation
            model_name="gpt-4-turbo-preview",
            api_key=settings.openai_api_key
        )
        
        # Your system prompt
        self.system_prompt = """You are an intelligent job search assistant. Your role is to help users find relevant job opportunities and provide career guidance. When users describe what they're looking for, help them by:

        1. Understanding their requirements for:
           - Role/title
           - Skills and experience level
           - Salary expectations
           - Location preferences
           - Work type (remote/hybrid/onsite)
           
        2. Providing relevant job matches with key details about:
           - Job responsibilities
           - Required qualifications
           - Company information
           - Compensation and benefits
           - Location and work arrangement
           
        3. Offering additional career guidance like:
           - Skills that could make them more competitive
           - Industry insights
           - Interview preparation tips
           - Salary negotiation advice

        Keep your tone professional but friendly, and focus on being helpful and informative.

        Example Interactions:
        User: "I'm looking for a remote software engineering job that pays at least $120k"
        Assistant: Let me help you find relevant positions. Based on your requirements, I'll look for remote software engineering roles with competitive compensation...

        User: "Tell me more about the requirements for the first job"
        Assistant: Let me break down the key requirements for that position...
        """
        
        # Your conversation prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Your follow-up prompt
        self.followup_prompt = PromptTemplate("""
        Given the conversation history and follow-up message, rewrite the question 
        to include relevant context from previous messages.

        Chat History:
        {chat_history}

        Follow Up Message:
        {question}

        Standalone question:
        """)
        
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
        try:
            # Create messages list with system prompt and user message
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(role=MessageRole.USER, content=message)
            ]
            
            # Get response from ChatGPT
            response = self.llm.chat(messages)
            
            return {
                "response": response.content,
                "session_id": session_id
            }
        except Exception as e:
            print(f"Error in chat: {str(e)}")  # For debugging
            return {
                "response": "I apologize, but I'm having trouble right now. Could you try asking your question again?",
                "session_id": session_id
            }

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
            
            return {
                "jobs": formatted_results,
                "session_id": session_id,
                "user_context": user_context if user_profile else None
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

@lru_cache()
def get_search_service(db: Session):
    """Initialize and return a JobSearchService instance"""
    settings = get_settings()
    
    # Initialize the model using the name from settings
    embed_model = SentenceTransformer(settings.embed_model_name)
    
    # Initialize the retriever with the model instance
    retriever = JobSearchRetriever(
        db=db,
        embed_model=embed_model,  # Pass the initialized model
        strategy=SearchStrategy.SEMANTIC
    )
    
    # Create and return the search service
    return JobSearchService(retriever=retriever, settings=settings)