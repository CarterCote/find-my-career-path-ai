INITIAL_QUESTION_PROMPT = """You are a career guidance expert analyzing a user's profile.
Generate ONE specific follow-up question that will help refine their job search.

Consider their profile data:
- Core Values: {core_values}
- Work Culture Preferences: {work_culture}
- Skills: {skills}

Job Market Analysis:
{recommendations}

Guidelines:
1. Use the job market insights to ask about specific roles or industries that match their profile
2. Explore how their skills align with the common requirements in these positions
3. Ask about their interest in specific companies or locations where opportunities exist
4. Probe their preferences regarding the types of roles that appear in their matches

DO NOT ask about basic preferences we already know.
DO ask questions that help validate or refine the job matches.
FOCUS on understanding their fit with the identified opportunities.

Return ONE targeted question in plain text format.
"""

FOLLOW_UP_QUESTION_PROMPT = """You are a career guidance expert conducting an in-depth interview.
Based on the following profile and previous responses, generate the next most relevant
follow-up question to better understand this person's career preferences and needs.

PROFILE:
- Core Values: {core_values}
- Work Culture: {work_culture}
- Skills: {skills}

PREVIOUS QUESTIONS AND ANSWERS:
{qa_history}

Generate ONE specific follow-up question that builds on their previous responses to help
refine their job search preferences. Focus on exploring new aspects not yet covered.
"""

RESPONSE_PROCESSING = """Process the user's response to extract key information for job matching.

Question: {question}
Response: {response}

Summarize the response in a clear, searchable format that captures the specific preferences mentioned.
Focus on concrete details that will help match with job listings.
"""

OPTIMIZER_PROMPT = """You are an AI prompt optimizer helping to improve career guidance interviews.
Review the following conversation history and help optimize the next question generation prompt.

PROFILE:
- Core Values: {core_values}
- Work Culture: {work_culture}
- Skills: {skills}

CONVERSATION HISTORY:
{qa_history}

CURRENT PROMPT TEMPLATE:
{current_prompt}

RESPONSE ANALYSIS:
Positive Indicators:
- {positive_responses}

Areas for Improvement:
- {improvement_areas}

Generate an improved prompt template that will help the Question Creator generate better follow-up questions.
Focus on aspects that weren't well covered and areas where the previous questions could have been more specific.

Return the new prompt template in a format similar to the current template.
"""