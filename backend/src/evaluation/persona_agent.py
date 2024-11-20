from typing import Dict
from langchain_openai import ChatOpenAI
from ..prompts.evaluation_prompts import PERSONA_CREATION, PROFILE_EVALUATION
import json

class CareerPersonaAgent:
    def __init__(self, job_details: Dict, llm: ChatOpenAI):
        self.job_details = job_details
        self.llm = llm
        self.persona = self._create_persona()
        
    def _create_persona(self) -> Dict:
        """Create detailed persona for the job role"""
        response = self.llm.chat([
            {"role": "system", "content": "You are an expert at creating realistic professional personas."},
            {"role": "user", "content": PERSONA_CREATION.format(
                job_details=json.dumps(self.job_details, indent=2)
            )}
        ])
        return json.loads(response.content)
        
    def evaluate_profile_match(self, user_profile: Dict) -> Dict:
        """Evaluate user profile match from persona perspective"""
        response = self.llm.chat([
            {"role": "system", "content": "You are an experienced professional evaluating career fits."},
            {"role": "user", "content": PROFILE_EVALUATION.format(
                job_title=self.job_details.get('title', 'Professional'),
                persona_json=json.dumps(self.persona, indent=2),
                profile_json=json.dumps(user_profile, indent=2)
            )}
        ])
        return json.loads(response.content)