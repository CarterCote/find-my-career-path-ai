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
            print("\n========== PROFILE QUESTIONER DEBUG ==========")
            print(f"Function called with profile data: {profile_data}")
            
            formatted_profile = {
                'core_values': profile_data.get('core_values', []),
                'work_culture': profile_data.get('work_culture', []),
                'skills': profile_data.get('skills', []),
                'additional_interests': profile_data.get('additional_interests', '')
            }
            
            print("\nFormatted profile:")
            print("================")
            print(f"Core Values: {', '.join(formatted_profile['core_values'])}")
            print(f"Work Culture: {', '.join(formatted_profile['work_culture'])}")
            print(f"Skills: {', '.join(formatted_profile['skills'])}")
            print(f"Additional Interests: {formatted_profile['additional_interests']}")
            print("================")
            
            messages = [{
                "role": "system", 
                "content": QUESTION_GENERATION.format(
                    core_values=', '.join(formatted_profile['core_values']),
                    work_culture=', '.join(formatted_profile['work_culture']),
                    skills=', '.join(formatted_profile['skills']),
                    additional_interests=formatted_profile['additional_interests']
                )
            }]

            print(f"\nDebug - Final prompt being sent to LLM:\n{messages[0]['content']}")
            
            # Use invoke instead of chat
            response = self.llm.invoke(messages)
            print(f"\nDebug - Raw LLM response: {response}")
            
            # Extract content from the response
            content = response.content if hasattr(response, 'content') else str(response)
            content = content.strip()
            
            # Parse the numbered list into questions
            questions = [q.strip() for q in content.split('\n') if '?' in q]
            
            if len(questions) == 4:
                return questions
            
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
            messages = [
                {
                    "role": "system",
                    "content": "You are a data structuring assistant. Always respond with a valid JSON object."
                },
                {
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
                }
            ]
            
            # Use invoke instead of chat
            result = self.llm.invoke(messages)
            
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