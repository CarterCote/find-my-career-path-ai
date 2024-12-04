from typing import List, Dict
from llama_index.core import PromptTemplate
from langchain_openai import ChatOpenAI
from ..models import UserProfile
from sqlalchemy.orm import Session
import json
from ..prompts.profile_prompts import INITIAL_QUESTION_PROMPT, FOLLOW_UP_QUESTION_PROMPT, OPTIMIZER_PROMPT

class ProfileQuestioner:
    def __init__(self, settings):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o",
            api_key=settings.openai_api_key
        )
        
        self.conversation_history = []
        self.current_prompt = FOLLOW_UP_QUESTION_PROMPT
        self.response_quality_history = []

    def optimize_prompt(self, formatted_profile: Dict, previous_qa: List[Dict]) -> str:
        """Optimize the question generation prompt based on conversation history"""
        try:
            print("\n========== OPTIMIZER DEBUG ==========")
            print(f"Input Profile Data:")
            print(f"Core Values: {formatted_profile['core_values']}")
            print(f"Work Culture: {formatted_profile['work_culture']}")
            print(f"Skills: {formatted_profile['skills']}")
            
            # Analyze response quality
            positive_responses = []
            improvement_areas = []
            
            for qa in previous_qa:
                try:
                    analysis_prompt = f"""Analyze this question-answer pair for response quality:
                    Question: {qa['question']}
                    Response: {qa['response']}
                    
                    Return a JSON object with these fields:
                    {{
                        "detailed_response": boolean,  // Did the response provide specific details?
                        "needs_improvement": boolean,  // Could the question have been more specific?
                        "relevant_details": [string],  // List of relevant details provided
                        "missing_aspects": [string]    // Aspects that could have been explored
                    }}
                    
                    IMPORTANT: Respond ONLY with the JSON object, no additional text.
                    """
                    
                    messages = [{"role": "system", "content": analysis_prompt}]
                    response = self.llm.invoke(messages)
                    
                    # Clean the response content
                    content = response.content.strip()
                    # Remove any markdown formatting if present
                    if content.startswith('```json'):
                        content = content[7:-3] if content.endswith('```') else content[7:]
                    elif content.startswith('```'):
                        content = content[3:-3] if content.endswith('```') else content[3:]
                        
                    print(f"Debug - Raw analysis response: {content}")
                    
                    # Parse JSON response
                    quality_score = json.loads(content)
                    
                    if quality_score.get('detailed_response', False):
                        positive_responses.append(f"Question '{qa['question']}' elicited specific details")
                    if quality_score.get('needs_improvement', False):
                        improvement_areas.append(f"Question '{qa['question']}' could be more specific")
                        
                except Exception as e:
                    print(f"Error analyzing response: {str(e)}")
                    print(f"Raw response content: {response.content if hasattr(response, 'content') else 'No content'}")
                    continue

            optimizer_context = OPTIMIZER_PROMPT.format(
                core_values=', '.join(formatted_profile['core_values']),
                work_culture=', '.join(formatted_profile['work_culture']),
                skills=', '.join(formatted_profile['skills']),
                qa_history="\n".join([f"Q: {qa['question']}\nA: {qa['response']}" for qa in previous_qa]),
                current_prompt=self.current_prompt,
                positive_responses="\n- ".join(positive_responses) if positive_responses else "None identified",
                improvement_areas="\n- ".join(improvement_areas) if improvement_areas else "None identified"
            )

            print(f"\nFormatted Optimizer Context:")
            print(optimizer_context)
            
            messages = [{"role": "system", "content": optimizer_context}]
            response = self.llm.invoke(messages)
            
            new_prompt = response.content.strip()
            print(f"\nOptimized Prompt:\n{new_prompt}")
            
            # Verify the new prompt has the placeholders
            if "{core_values}" not in new_prompt or "{work_culture}" not in new_prompt or "{skills}" not in new_prompt:
                print("\nWarning: New prompt is missing required placeholders, falling back to current prompt")
                return self.current_prompt
                
            self.current_prompt = new_prompt
            return new_prompt
            
        except Exception as e:
            print(f"Error in optimizer: {str(e)}")
            return self.current_prompt

    def _analyze_response_quality(self, question: str, response: str) -> Dict:
        """Analyze the quality of a response"""
        try:
            analysis_prompt = f"""Analyze this question-answer pair for response quality:
            Question: {question}
            Response: {response}
            
            Return a JSON object with these fields:
            {{
                "detailed_response": boolean,  // Did the response provide specific details?
                "needs_improvement": boolean,  // Could the question have been more specific?
                "relevant_details": [string],  // List of relevant details provided
                "missing_aspects": [string]    // Aspects that could have been explored
            }}
            
            IMPORTANT: Respond ONLY with the JSON object, no additional text.
            """
            
            messages = [{"role": "system", "content": analysis_prompt}]
            response = self.llm.invoke(messages)
            
            # Clean the response content
            content = response.content.strip()
            # Remove any markdown formatting if present
            if content.startswith('```json'):
                content = content[7:-3] if content.endswith('```') else content[7:]
            elif content.startswith('```'):
                content = content[3:-3] if content.endswith('```') else content[3:]
                
            print(f"Debug - Raw analysis response: {content}")
            
            # Parse JSON response
            return json.loads(content)
            
        except Exception as e:
            print(f"Error analyzing response: {str(e)}")
            print(f"Raw response content: {response.content if hasattr(response, 'content') else 'No content'}")
            return {
                "detailed_response": False,
                "needs_improvement": True,
                "relevant_details": [],
                "missing_aspects": []
            }

    def generate_questions(self, profile_data: Dict, previous_qa: List[Dict] = None) -> List[str]:
        """Generate targeted questions based on user profile and previous Q&A"""
        try:
            print("\n========== PROFILE QUESTIONER DEBUG ==========")
            # print(f"Function called with profile data: {profile_data}")
            
            formatted_profile = self._format_profile(profile_data)
            qa_count = len(previous_qa) if previous_qa else 0
            
            print("\nFormatted profile:")
            print("================")
            print(f"Core Values: {', '.join(formatted_profile['core_values'])}")
            print(f"Work Culture: {', '.join(formatted_profile['work_culture'])}")
            print(f"Skills: {', '.join(formatted_profile['skills'])}")
            print(f"Additional Interests: {formatted_profile['additional_interests']}")
            print("================")
            
            # Determine missing fields
            missing_fields = []
            if not formatted_profile['skills']:
                missing_fields.append('skills')
            if not formatted_profile['work_culture']:
                missing_fields.append('work environment preferences')
            
            # Use INITIAL_QUESTION_PROMPT only for the first question
            if qa_count == 0:
                print(f"\nGenerating initial question...")
                context = self._build_context(formatted_profile, previous_qa, is_core_question=True)
            else:
                # Use FOLLOW_UP_QUESTION_PROMPT for subsequent questions
                print("\nGenerating follow-up question...")
                print("\nOptimizing prompt based on previous responses...")
                self.optimize_prompt(formatted_profile, previous_qa)
                context = self._build_context(formatted_profile, previous_qa, is_core_question=False)
                
                # If there are missing fields, explicitly ask about them
                if missing_fields:
                    context += f"\n\nWe still need to understand your {', '.join(missing_fields)}. Could you provide more details on these aspects?"

            messages = [{
                "role": "system",
                "content": context
            }]

            print(f"\nDebug - Final prompt being sent to LLM:\n{messages[0]['content']}")
            
            response = self.llm.invoke(messages)
            print(f"\nDebug - Raw LLM response: {response}")
            
            # Extract and return single question
            content = response.content if hasattr(response, 'content') else str(response)
            content = content.strip()
            print(f"\nDebug - Stripped content: {content}")
            
            questions = [q.strip() for q in content.split('\n') if '?' in q]
            print(f"\nDebug - Extracted questions: {questions}")
            
            if questions:
                selected_question = questions[0]
                print(f"\nDebug - Selected question: {selected_question}")
                return [selected_question]
            else:
                default_question = self._get_default_questions()[qa_count]
                print(f"\nDebug - Using default question: {default_question}")
                return [default_question]

        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return [self._get_default_questions()[0]]

    def _format_profile(self, profile_data: Dict) -> Dict:
        """Helper to format profile data"""
        return {
            'core_values': profile_data.get('core_values', []),
            'work_culture': profile_data.get('work_culture', []),
            'skills': profile_data.get('skills', []),
            'additional_interests': profile_data.get('additional_interests', ''),
            'job_patterns': profile_data.get('job_patterns', {})
        }

    def _build_context(self, formatted_profile: Dict, previous_qa: List[Dict] = None, is_core_question: bool = True) -> str:
        print("\n=== BUILDING QUESTION CONTEXT ===")
        print(f"Question Type: {'Core' if is_core_question else 'Follow-up'}")
        
        qa_history = "\n".join([
            f"Q: {qa['question']}\nA: {qa['response']}"
            for qa in (previous_qa or [])
        ])
        
        # Build comprehensive job market insights
        job_patterns = formatted_profile.get('job_patterns', {})
        print("\nJob Patterns Debug:")
        print(f"Raw job_patterns: {json.dumps(job_patterns, indent=2)}")
        
        recommendations_context = ""
        if job_patterns:
            print("\nBuilding recommendations context with patterns:")
            recommendations_context += f"\n\nTop Industries: {', '.join(job_patterns['top_industries'])}"
            recommendations_context += f"\nCommon Role Types: {', '.join(job_patterns['top_roles'][:5])}"
            recommendations_context += f"\nMost Required Skills: {', '.join(job_patterns['common_skills'][:5])}"
            recommendations_context += f"\nActive Hiring Companies: {', '.join(job_patterns['top_companies'][:3])}"
            print(f"Final recommendations_context: {recommendations_context}")
        else:
            print("No job patterns available for recommendations context")
        
        final_context = (INITIAL_QUESTION_PROMPT if is_core_question else FOLLOW_UP_QUESTION_PROMPT).format(
            core_values=', '.join(formatted_profile['core_values']),
            work_culture=', '.join(formatted_profile['work_culture']),
            skills=', '.join(formatted_profile['skills']),
            qa_history=qa_history,
            recommendations=recommendations_context
        )
        
        print("\nFinal Context Structure:")
        print("1. Profile Data")
        print("2. Job Market Insights")
        print("3. Previous Q&A History")
        print("=== END CONTEXT BUILDING ===\n")
        
        return final_context

    def _get_default_questions(self) -> List[str]:
        """Return default fallback questions"""
        return [
            "What specific technical skills are most important for your ideal role?",
            "What type of work environment do you thrive in?",
            "What industry sectors interest you most?",
            "What role would you like to have in 5 years?"
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