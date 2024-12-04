from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..prompts.evaluation_prompts import PERSONA_CREATION, PROFILE_EVALUATION
import json

class CareerPersonaAgent:
    def __init__(self, job_details: Dict, llm: ChatOpenAI):
        self.job_details = job_details
        self.llm = llm
        try:
            self.persona = self._create_persona()
        except Exception as e:
            print(f"Error creating persona for {job_details.get('title', 'Unknown Job')}: {str(e)}")
            self.persona = self._get_default_persona()

    def _clean_json_content(self, content: str) -> str:
        """Clean and normalize JSON content from LLM response"""
        # Remove any markdown formatting
        content = content.replace('```json', '').replace('```', '')
        
        # Remove all newlines and extra whitespace
        content = ' '.join(content.split())
        
        # Extract JSON object if there's extra text
        try:
            start = content.index('{')
            end = content.rindex('}') + 1
            content = content[start:end]
        except ValueError:
            return '{}'
        
        return content

    def _create_persona(self) -> Dict:
        """Create detailed persona for the job role"""
        try:
            print("\nCreating persona with job details:")
            print(f"Title: {self.job_details.get('title')}")
            print(f"Company: {self.job_details.get('company_name')}")
            print(f"Description length: {len(self.job_details.get('description', ''))}")
            
            # Format job details for prompt
            formatted_details = "\n".join([
                f"Title: {self.job_details.get('title')}",
                f"Company: {self.job_details.get('company_name')}",
                f"Location: {self.job_details.get('location')}",
                f"Description: {self.job_details.get('description', '')[:1000]}",
                f"Required Skills: {', '.join(self.job_details.get('required_skills', []))}"
            ])
            
            messages = [
                SystemMessage(content=(
                    "You are an expert at creating realistic professional personas. "
                    "Return ONLY a single-line JSON object with no line breaks, indentation, or additional text."
                )),
                HumanMessage(content=PERSONA_CREATION.format(job_details=formatted_details))
            ]
            
            response = self.llm.invoke(messages)
            content = self._clean_json_content(response.content if hasattr(response, 'content') else '')
            
            try:
                parsed = json.loads(content)
                required_fields = ['skills', 'values', 'work_preferences', 
                                 'experience_level', 'career_goals', 'communication_style']
                if not all(field in parsed for field in required_fields):
                    print("Missing required fields in persona")
                    return self._get_default_persona()
                return parsed
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Cleaned content: {content}")
                return self._get_default_persona()
            
        except Exception as e:
            print(f"Error creating persona: {str(e)}")
            return self._get_default_persona()
        
    def evaluate_profile_match(self, user_profile: Dict) -> Dict:
        """Evaluate user profile match from persona perspective"""
        try:
            print(f"\nEvaluating profile match for job: {self.job_details.get('title', 'Unknown')}")
            
            messages = [
                SystemMessage(content=(
                    "You are an experienced professional evaluating career fits. "
                    "The response must be a valid JSON object with this exact structure:\n"
                    "{\n"
                    '    "skills_alignment": 5,\n'
                    '    "values_compatibility": 5,\n'
                    '    "culture_fit": 5,\n'
                    '    "growth_potential": 5,\n'
                    '    "reasoning": {\n'
                    '        "skills": "explanation",\n'
                    '        "values": "explanation",\n'
                    '        "culture": "explanation",\n'
                    '        "growth": "explanation"\n'
                    '    },\n'
                    '    "skill_gaps": ["gap1", "gap2"],\n'
                    '    "culture_fit_details": ["detail1", "detail2"]\n'
                    "}\n"
                    "Return ONLY the JSON with no additional text or formatting."
                )),
                HumanMessage(content=PROFILE_EVALUATION.format(
                    job_title=self.job_details.get('title', 'Unknown'),
                    persona_json=json.dumps(self.persona, indent=2),
                    profile_json=json.dumps({
                        "core_values": user_profile.get("core_values", []),
                        "work_culture": user_profile.get("work_culture", []),
                        "skills": user_profile.get("skills", []),
                        "additional_interests": user_profile.get("additional_interests", "")
                    }, indent=2)
                ))
            ]
            
            response = self.llm.invoke(messages)
            if not hasattr(response, 'content'):
                print("Invalid response format from LLM")
                return self._get_default_evaluation()
            
            content = response.content.strip()
            # print(f"Raw evaluation response: {content}")
            
            if not content:
                print("Empty response from LLM")
                return self._get_default_evaluation()
            
            # Clean the response
            content = content.replace('```json', '').replace('```', '').strip()
            if not content.startswith('{'):
                content = content.split('{', 1)[1]
            if not content.endswith('}'):
                content = content.rsplit('}', 1)[0] + '}'
            
            try:
                parsed = json.loads(content)
                # Validate required fields
                required_fields = ['skills_alignment', 'values_compatibility', 
                                 'culture_fit', 'growth_potential', 'reasoning']
                if not all(field in parsed for field in required_fields):
                    print("Missing required fields in evaluation")
                    return self._get_default_evaluation()
                return parsed
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Cleaned content: {content}")
                return self._get_default_evaluation()
            
        except Exception as e:
            print(f"Error in profile evaluation: {str(e)}")
            return self._get_default_evaluation()
        
    def _get_default_persona(self) -> Dict:
        """Return a default persona when creation fails"""
        return {
            "skills": ["professional expertise", "communication"],
            "values": ["professionalism", "quality"],
            "work_preferences": ["collaborative environment"],
            "experience_level": "mid-level",
            "career_goals": ["career growth"],
            "communication_style": "professional"
        }
        
    def _get_default_evaluation(self) -> Dict:
        """Provide a default evaluation when processing fails"""
        return {
            "skills_alignment": 5,
            "values_compatibility": 5,
            "culture_fit": 5,
            "growth_potential": 5,
            "reasoning": {
                "skills": "Error processing evaluation",
                "values": "Error processing evaluation", 
                "culture": "Error processing evaluation",
                "growth": "Error processing evaluation"
            },
            "skill_gaps": [],
            "culture_fit_details": []
        }