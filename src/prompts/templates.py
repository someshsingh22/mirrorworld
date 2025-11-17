"""Prompt templates for the interview agent."""

PLAN_GENERATION_PROMPT = """You are conducting a strategic interview to build a comprehensive user profile.

Progress: {steps_completed}/{max_steps} questions asked

Previous Q&A:
{qna_context}

Target task (objective):
{target_task}

Current persona estimate:
{persona_estimate}

Generate a strategic plan for the remaining questions. Consider:
1. What areas have been explored already
2. What information gaps remain
3. How to build a cohesive user profile

Focus on specific, actionable areas to explore."""

TARGET_TASK_PROMPT = """Based on the current persona estimate (and Q&A context if available), define ONE concrete objective for this interview.

Persona estimate:
{persona_estimate}

Q&A context (optional):
{qna_context}

Requirements for the objective:
1. It should be concise and outcome-oriented (imperative sentence).
2. It should be achievable via a short sequence of targeted yes/no questions.
3. It should be useful for downstream personalization (e.g., recommendations, messaging, outreach, role-fit).

Return ONLY the objective sentence, no preamble or explanations."""


QUESTION_GENERATION_PROMPT = """You are conducting a strategic interview to build a user profile.

Progress: {steps_completed}/{max_steps} questions asked

Previous Q&A:
{qna_context}

Strategic plan:
{plan}

Target task (objective):
{target_task}

Current persona estimate:
{persona_estimate}

Generate the NEXT question following these requirements:
1. The question should ideally be answerable with YES or NO, but users may provide other responses
2. Must be completely unambiguous and clear
3. Should strategically advance the user profile
4. Must not repeat or rephrase previous questions
5. Should explore new aspects based on previous answers
6. Prioritize information that advances the target task

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


def persona_to_roleplay_system_prompt(persona_estimate: str) -> str:
    """Convert a persona description into a system prompt for roleplay.

    Args:
        persona_estimate: Free-form text description of the user persona.

    Returns:
        A system prompt instructing the model to adopt and consistently roleplay the persona.
    """
    description = persona_estimate.strip() if persona_estimate else ""
    if not description:
        # Fallback when no persona is available; keep generic but well-scoped
        return "You are a helpful, neutral assistant. Be concise, factual, and polite."

    return (
        "You are roleplaying as the following persona.\n\n"
        "Persona description:\n"
        f"{description}\n\n"
        "Roleplay instructions:\n"
        "- Always stay in character based on the persona description.\n"
        "- Speak in first-person singular (I/me) as the persona.\n"
        "- Match tone, vocabulary, and knowledge to the persona's background.\n"
        "- If information is unknown to the persona, say you don't know rather than inventing facts.\n"
        "- Keep responses natural, specific, and consistent with the persona.\n"
    )
