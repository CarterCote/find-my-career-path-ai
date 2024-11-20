SEARCH_ENHANCEMENT = """Enhance the job search query with profile-specific parameters.
Consider the following aspects of the user's profile:

Profile:
{profile_json}

Original Query:
{query}

Return an enhanced search query that incorporates:
1. Relevant skills and experience
2. Work preferences
3. Career interests
4. Cultural alignment

Format: Return a single string with the enhanced search query.
"""