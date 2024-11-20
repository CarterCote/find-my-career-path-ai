from typing import List, Dict
from llama_index.core import PromptTemplate
from langchain_openai import ChatOpenAI
from ..models import UserProfile
from sqlalchemy.orm import Session
import json
from ..prompts.profile_prompts import QUESTION_GENERATION, RESPONSE_PROCESSING

class ProfileQuestioner:
    def __init__(self, settings):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4-turbo-preview",
            api_key=settings.openai_api_key
        )
        
        self.question_prompt = QUESTION_GENERATION
        self.processing_prompt = RESPONSE_PROCESSING

    def generate_questions(self, profile_data: Dict) -> List[str]:
        """Generate targeted questions based on user profile"""
        try:
            print(f"Debug - Profile data: {profile_data}")  # Add debug logging
            
            # Format the profile data for better prompt context
            formatted_profile = {
                'core_values': profile_data.get('core_values', []),
                'work_culture': profile_data.get('work_culture', []),
                'skills': profile_data.get('skills', []),
                'additional_interests': profile_data.get('additional_interests', '')
            }
            
            system_prompt = (
                "You are a career guidance expert. Generate personalized questions based on the user's profile. "
                "Your response must be a valid JSON array of strings containing 3-4 questions. "
                'Example format: ["question 1?", "question 2?", "question 3?"]'
            )
            
            user_prompt = f"""Based on this profile, generate specific follow-up questions:
Core Values: {', '.join(str(v) for v in formatted_profile['core_values'])}
Work Culture: {', '.join(str(v) for v in formatted_profile['work_culture'])}
Skills: {', '.join(str(v) for v in formatted_profile['skills'])}
Additional Interests: {formatted_profile['additional_interests']}

Focus your questions on:
1. Specific applications of their listed skills
2. How their values align with potential roles
3. Their work culture preferences
4. Their career interests

Return ONLY a JSON array of questions."""
            
            response = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            print(f"Debug - LLM response: {response.content}")  # Add debug logging
            
            # Clean the response content to handle potential formatting
            content = response.content.strip()
            if not content.startswith('['):
                content = content.split('[', 1)[-1]
            if not content.endswith(']'):
                content = content.split(']')[0] + ']'
            
            try:
                questions = json.loads(content)
                if isinstance(questions, list):
                    return questions
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw content: {content}")
            
            # If we get here, fall back to text parsing
            default_questions = [
                "What specific technical skills are most important for your ideal role?",
                "What type of work environment do you thrive in?",
                "What industry sectors interest you most?"
            ]
            
            # Try to extract questions from text if present
            if '?' in content:
                questions = [q.strip() for q in content.split('\n') 
                            if '?' in q and len(q.strip()) > 10]
                return questions if questions else default_questions
            
            return default_questions
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return [
                "What specific technical skills are most important for your ideal role?",
                "What type of work environment do you thrive in?",
                "What industry sectors interest you most?"
            ]

    def process_response(self, question: str, response: str) -> Dict:
        """Process user's natural language response into structured search parameters"""
        try:
            result = self.llm.chat([{
                "role": "system",
                "content": "You are a data structuring assistant. Always respond with a valid JSON object."
            }, {
                "role": "user",
                "content": f"""Convert this response into search parameters:
                    Question: {question}
                    Response: {response}
                    
                    Return EXACTLY like this example:
                    {{
                        "required_skills": ["python", "aws"],
                        "work_environment": "startup",
                        "experience_years": 3,
                        "team_size": "small",
                        "industry": "technology"
                    }}"""
            }])
            
            # Clean and parse response
            content = result.content.strip()
            if not content.startswith('{'):
                content = content.split('{', 1)[-1]
            if not content.endswith('}'):
                content = content.split('}')[0] + '}'
            
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw content: {content}")
                return {}
            
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            return {}