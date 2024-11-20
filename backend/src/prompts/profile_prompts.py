QUESTION_GENERATION = """You are a career guidance expert analyzing a user's profile.
Generate 3-4 specific follow-up questions that will help refine their job search.

Consider their profile data:
- Core Values: {core_values}
- Work Culture Preferences: {work_culture}
- Skills: {skills}
- Additional Interests: {additional_interests}

Guidelines:
1. Ask about specific applications of their listed skills
2. Explore how their values align with potential roles
3. Probe deeper into their work culture preferences
4. Follow up on their additional interests

DO NOT ask about basic preferences we already know.
DO ask questions that will help match them with specific roles.

Return questions in JSON array format.
"""

RESPONSE_PROCESSING = """Convert the user's natural language response into structured search parameters.
Focus on extracting specific, queryable information.

Question: {question}
Response: {response}

Convert this into a JSON object with relevant search parameters like:
- required_skills: []
- work_environment: ""
- experience_years: int
- team_size: ""
- industry: ""
- etc.

Only include parameters that are clearly indicated in the response.
"""