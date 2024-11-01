from logging import Logger
from typing import Dict
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

class JobSearchService:
    def __init__(self, retriever, settings):
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
            # Store user message
            create_chat_message(db, session_id, query, is_user=True)
            
            # Try to get job recommendations
            try:
                response = str(self.chat_engine.chat(query))
            except Exception as e:
                # If no job postings found or other search error
                response = """I apologize, but I don't have any specific job recommendations at this time. 
                However, I can still help answer general career questions or provide guidance about:
                - Career planning
                - Resume writing
                - Interview preparation
                - Skill development
                What would you like to know more about?"""
            
            # Store assistant response
            create_chat_message(db, session_id, response, is_user=False)
            
            return {
                "response": response,
                "session_id": session_id
            }
        except Exception as e:
            return {
                "response": "I apologize, but I'm having trouble processing your request. Please try again later.",
                "session_id": session_id,
                "error": str(e)
            }