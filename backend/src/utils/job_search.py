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
from langchain_core.messages import HumanMessage
from ..crud import create_chat_message
from ..models import UserProfile
from .job_retriever import SearchStrategy

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
            related_titles = self._get_related_titles(query)
            
            # Format results for the LLM
            formatted_results = self._format_results(results)
            
            # Get user profile for personalization
            user_profile = db.query(UserProfile).filter(
                UserProfile.session_id == session_id
            ).first()
            
            # Include user preferences in the prompt if available
            user_context = ""
            if user_profile:
                user_context = f"""
                User Preferences:
                - Skills: {', '.join(user_profile.skills or [])}
                - Values: {', '.join(user_profile.core_values or [])}
                - Work Culture: {', '.join(user_profile.work_culture or [])}
                - Goals: {user_profile.goals}
                """
            
            # Create the message for the LLM with user context
            messages = [
                SystemMessage(content=self.system_prompt),
                *[ChatMessage(content=m.content, role=m.role) for m in history.messages[-5:]],  # Last 5 messages
                HumanMessage(content=f"""
                User Context: {user_context}
                
                Current Query: {query}
                
                Found Jobs: {json.dumps(formatted_results, indent=2)}
                
                Related Titles: {', '.join(related_titles)}
                
                Please provide a personalized response that:
                1. Addresses their search requirements
                2. Highlights the most relevant jobs based on their preferences
                3. Suggests related titles they might be interested in
                4. Provides relevant career advice
                """)
            ]
            
            # Get response from LLM
            response = self.llm(messages)
            
            # Store assistant response in both memory and database
            history.add_ai_message(response.content)
            create_chat_message(db, session_id, response.content, is_user=False)
            
            return {
                "response": response.content,
                "session_id": session_id,
                "jobs": formatted_results,
                "related_titles": related_titles
            }
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request.",
                "session_id": session_id,
                "error": str(e)
            }

    def filtered_search(
        self,
        min_salary: Optional[float] = None,
        required_skills: Optional[List[str]] = None,
        work_environment: Optional[str] = None,
        experience_years: Optional[int] = None,
        education_level: Optional[str] = None
    ) -> Dict:
        """Search jobs with structured filters from the enriched data"""
        try:
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