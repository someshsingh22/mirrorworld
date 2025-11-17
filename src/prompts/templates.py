"""Prompt templates for the interview agent."""

PLAN_GENERATION_PROMPT = """You are conducting a strategic interview to build a comprehensive user profile.

Progress: {steps_completed}/{max_steps} questions asked

Previous Q&A:
{qna_context}

Current persona estimate:
{persona_estimate}

Generate a strategic plan for the remaining questions. Consider:
1. What areas have been explored already
2. What information gaps remain
3. How to build a cohesive user profile

Focus on specific, actionable areas to explore."""


QUESTION_GENERATION_PROMPT = """You are conducting a strategic interview to build a user profile.

Progress: {steps_completed}/{max_steps} questions asked

Previous Q&A:
{qna_context}

Strategic plan:
{plan}

Current persona estimate:
{persona_estimate}

Generate the NEXT question following these requirements:
1. The question should ideally be answerable with YES or NO, but users may provide other responses
2. Must be completely unambiguous and clear
3. Should strategically advance the user profile
4. Must not repeat or rephrase previous questions
5. Should explore new aspects based on previous answers

Generate a question that provides maximum information gain. The question format should encourage yes/no answers, but be prepared to interpret any response."""


PERSONA_UPDATE_PROMPT = """Based on the Q&A history, update the user persona estimate.

Q&A History:
{qna_context}

Current persona estimate:
{current_estimate}

Analyze the responses and update the persona estimate as a comprehensive text description. You have complete freedom to:
1. Add new insights based on what you learn from the answers
2. Update existing understanding with new information
3. Refine or correct previous assumptions
4. Build a rich, detailed narrative about the user

Write a clear, comprehensive text description of the user persona. Include any relevant aspects like:
- Interests, hobbies, and passions
- Demographics and background
- Preferences and tendencies
- Personality traits and behaviors
- Values, goals, and motivations
- Communication style
- Work style and environment preferences
- Any other relevant characteristics

The description should be natural, readable text (not JSON or structured format). Be creative and comprehensive - build a complete picture of who this person is based on their responses."""
