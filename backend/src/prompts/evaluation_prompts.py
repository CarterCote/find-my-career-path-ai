PERSONA_CREATION = """Create a detailed professional persona for someone successful in this role.

Job Details:
{job_details}

Consider:
1. Technical skills and expertise level
2. Core values and motivations
3. Preferred work environment
4. Career trajectory
5. Communication style
6. Leadership approach

Return as JSON with these fields:
{
    "skills": [],
    "values": [],
    "work_preferences": [],
    "experience_level": "",
    "career_goals": [],
    "communication_style": ""
}
"""

PROFILE_EVALUATION = """As a {job_title} with the following characteristics:
{persona_json}

Evaluate this candidate's profile for fit:
{profile_json}

Consider:
1. Skills alignment (0-10)
2. Values compatibility (0-10)
3. Culture fit (0-10)
4. Growth potential (0-10)

Provide specific reasoning for each score.
Return as JSON with these fields:
{
    "skills_alignment": 0,
    "values_compatibility": 0,
    "culture_fit": 0,
    "growth_potential": 0,
    "reasoning": {
        "skills": "",
        "values": "",
        "culture": "",
        "growth": ""
    },
    "skill_gaps": [],
    "culture_fit_details": []
}
"""

TEST_PROFILE_GENERATION = """Create 3 different user profiles that would be good candidates for this job:

Job Details:
Title: {title}
Description: {description}
Company: {company}

For each profile include:
1. Core values (list)
2. Work culture preferences (list)
3. Skills (list)
4. Additional interests (string)

Return as JSON array of profiles.
Format:
[
    {
        "core_values": [],
        "work_culture": [],
        "skills": [],
        "additional_interests": ""
    }
]
"""

TEST_QUERY_GENERATION = """Generate 5 realistic job search queries based on this user profile:

Profile:
{profile_json}

Return as JSON array of strings. Queries should be natural and varied, like:
- "Looking for senior developer role in fintech"
- "Remote project manager position with mentorship"
etc.

Format: ["query1", "query2", "query3", "query4", "query5"]
"""