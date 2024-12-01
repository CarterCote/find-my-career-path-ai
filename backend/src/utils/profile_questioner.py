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
        print(f"\n=== Processing Response Start ===")
        print(f"Input Question: {question}")
        print(f"Input Response: {response}")
        
        try:
            # Prepare messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a data structuring assistant. Analyze the question and response to extract relevant job search parameters.
                    Focus on these key aspects:
                    - Skills mentioned
                    - Work environment preferences
                    - Industry interests
                    - Team size preferences
                    - Experience level indicators
                    
                    Always return a valid JSON object with relevant fields."""
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}
                    Response: {response}
                    
                    Structure the response into these categories (include only relevant fields):
                    {{
                        "required_skills": ["skill1", "skill2"],
                        "work_environment": "environment type",
                        "team_size": "size preference",
                        "industry": "industry preference",
                        "experience_level": "level",
                        "preferences": ["other relevant preferences"]
                    }}"""
                }
            ]
            
            print("\nInvoking LLM...")
            result = self.llm.invoke(messages)
            content = result.content.strip()
            print(f"Raw LLM Response: {content}")
            
            try:
                # Find the JSON object in the content
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                print(f"JSON Extraction - Start Index: {start_idx}, End Index: {end_idx}")
                
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    print(f"Extracted JSON string: {json_str}")
                    
                    parsed_data = json.loads(json_str)
                    print(f"Successfully parsed JSON: {json.dumps(parsed_data, indent=2)}")
                    
                    # Validate and clean the response
                    cleaned_data = {}
                    if parsed_data.get('required_skills'):
                        cleaned_data['required_skills'] = [
                            skill.lower().strip() 
                            for skill in parsed_data['required_skills']
                        ]
                    if parsed_data.get('work_environment'):
                        cleaned_data['work_environment'] = parsed_data['work_environment'].lower().strip()
                    if parsed_data.get('team_size'):
                        cleaned_data['team_size'] = parsed_data['team_size'].lower().strip()
                    if parsed_data.get('industry'):
                        cleaned_data['industry'] = parsed_data['industry'].lower().strip()
                    if parsed_data.get('experience_level'):
                        cleaned_data['experience_level'] = parsed_data['experience_level'].lower().strip()
                    if parsed_data.get('preferences'):
                        cleaned_data['preferences'] = [
                            pref.lower().strip() 
                            for pref in parsed_data['preferences']
                        ]
                    
                    print(f"Final cleaned data: {json.dumps(cleaned_data, indent=2)}")
                    print("=== Processing Response End ===\n")
                    return cleaned_data
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw content causing error: {content}")
                print("=== Processing Response End (with JSON error) ===\n")
                return {}
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print("=== Processing Response End (with error) ===\n")
            return {}