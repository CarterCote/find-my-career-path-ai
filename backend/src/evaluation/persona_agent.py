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
            # Provide a more complete default persona
            self.persona = {
                "skills": ["professional expertise", "communication"],
                "values": ["professionalism", "quality"],
                "work_preferences": ["collaborative environment"],
                "experience_level": "mid-level",
                "career_goals": ["career growth"],
                "communication_style": "professional"
            }
        
    def _create_persona(self) -> Dict:
        """Create detailed persona for the job role"""
        try:
            print(f"\nCreating persona for job: {self.job_details.get('title', 'Unknown')}")
            messages = [
                SystemMessage(content=(
                    "You are an expert at creating realistic professional personas. "
                    "Return ONLY valid JSON without any additional text, markdown, or formatting. "
                    "The response should start with '{' and end with '}'."
                )),
                HumanMessage(content=PERSONA_CREATION.format(
                    job_details=json.dumps({
                        "title": self.job_details.get("title", ""),
                        "company_name": self.job_details.get("company_name", ""),
                        "description": self.job_details.get("description", ""),
                        "required_skills": self.job_details.get("matching_skills", []),
                        "culture": self.job_details.get("matching_culture", [])
                    }, indent=2)
                ))
            ]
            response = self.llm.invoke(messages)
            
            # Clean the response content
            content = response.content.strip()
            if not content.startswith('{'):
                content = content[content.find('{'):]
            if not content.endswith('}'):
                content = content[:content.rfind('}')+1]
            
            print(f"Raw persona response: {content}")
            return json.loads(content)
            
        except Exception as e:
            print(f"Error in persona creation: {str(e)}")
            print(f"Raw response content: {response.content if 'response' in locals() else 'No response'}")
            raise
        
    def evaluate_profile_match(self, user_profile: Dict) -> Dict:
        """Evaluate user profile match from persona perspective"""
        try:
            print(f"\nEvaluating profile match for job: {self.job_details.get('title', 'Unknown')}")
            messages = [
                SystemMessage(content=(
                    "You are an experienced professional evaluating career fits. "
                    "Return ONLY valid JSON without any additional text, markdown, or formatting. "
                    "The response should start with '{' and end with '}'."
                )),
                HumanMessage(content=PROFILE_EVALUATION.format(
                    job_title=self.job_details.get('title', 'Professional'),
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
            
            # Clean the response content
            content = response.content.strip()
            if not content.startswith('{'):
                content = content[content.find('{'):]
            if not content.endswith('}'):
                content = content[:content.rfind('}')+1]
            
            print(f"Raw evaluation response: {content}")
            return json.loads(content)
            
        except Exception as e:
            print(f"Error in profile evaluation: {str(e)}")
            print(f"Raw response content: {response.content if 'response' in locals() else 'No response'}")
            return {
                "skills_alignment": 5,
                "values_compatibility": 5,
                "culture_fit": 5,
                "growth_potential": 5,
                "reasoning": {
                    "skills": f"Error in evaluation: {str(e)}",
                    "values": f"Error in evaluation: {str(e)}",
                    "culture": f"Error in evaluation: {str(e)}",
                    "growth": f"Error in evaluation: {str(e)}"
                },
                "skill_gaps": [],
                "culture_fit_details": []
            }