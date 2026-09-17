"""
services/gemini_service.py

Responsible ONLY for communicating with Gemini.

This module:
1. Builds the research prompt
2. Sends Tavily's retrieved sources to Gemini
3. Generates a detailed explanation
4. Returns the AI-generated response

No Tavily/search logic lives here.
"""

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    RESEARCH_MODES,
    DEFAULT_MODE,
)


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are an AI News Research and Explanation Assistant.

Your main goal is to help the user UNDERSTAND a topic deeply.

You will receive:
1. A user's research query
2. A collection of web sources retrieved by a search system

Use ONLY the information contained in those retrieved sources.

Do not invent:
- facts
- dates
- statistics
- names
- quotes
- events
- explanations that are not supported by the sources

Your response should be DETAILED and EXPLANATORY.

Do NOT give a very short summary.

When the sources contain enough information, aim for approximately
1000-1500 words or more.

Explain the topic naturally using multiple paragraphs and useful
subheadings.

The response should help a college student understand:
- What happened?
- What is happening now?
- Why is it happening?
- What are the important developments?
- What is the background?
- What are the important facts?
- What are the effects or implications?
- Why does this matter?
- How are the different developments connected?
- What uncertainties or disagreements exist?

Do not simply copy or list information from the sources.

Synthesize information from multiple sources into a coherent explanation.

Use source references such as (Source 1) or (Source 3) when making
important source-specific claims.

Do NOT put a source number after every sentence.

If multiple sources discuss the same development, combine their
information instead of repeating the same point.

If sources disagree, clearly explain the disagreement.

If the sources do not contain enough information to answer something,
say so clearly instead of guessing.

Use clear language suitable for a college student.

Avoid unnecessary repetition, but prioritize completeness and
understanding over brevity.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def _build_prompt(
    query: str,
    sources_block: str,
    mode: str
) -> str:
    """
    Build the complete research prompt.

    The prompt contains:
    - Role
    - User query
    - Retrieved sources
    - Research depth
    - Output structure
    - Grounding constraints
    """

    mode_config = RESEARCH_MODES.get(
        mode,
        RESEARCH_MODES[DEFAULT_MODE]
    )

    depth_instruction = mode_config["depth_instruction"]

    prompt = f"""
ROLE:
You are an AI News Research and Explanation Assistant.

USER QUERY:
{query}

RETRIEVED WEB SOURCES:
The following information was retrieved from the web.

--------------------------------------------------

{sources_block}

--------------------------------------------------

RESEARCH DEPTH:
{depth_instruction}

IMPORTANT:
The retrieved sources are the ONLY factual basis for your answer.

Do not use unsupported information from your general knowledge.

Your job is not simply to summarize each source separately.

Instead, combine the useful information from all relevant sources and
create ONE coherent, detailed explanation of the topic.

Explain the topic as if you are teaching it to a college student who
wants to understand both the basic situation and the important details.

OUTPUT REQUIREMENTS:

### Executive Summary

Write a detailed overview of the topic.

Use approximately 3-5 paragraphs when enough information is available.

Explain the overall situation, the most important developments, and
the main takeaway.

Do NOT make this just a few sentences.


### Detailed Explanation

This should be the MAIN and LONGEST section of the response.

Break the topic into logical subtopics using subheadings.

For each important subtopic:

- Explain what happened.
- Explain the relevant background.
- Include important facts and details from the sources.
- Explain why the development matters.
- Explain connections with other developments.
- Mention dates or figures when supported by the sources.

Use multiple paragraphs.

Do not reduce this section to a short bullet list.


### Recent Developments

Explain the most important recent developments found in the sources.

Describe what changed, when it happened, and why it is relevant.

Use dates when the sources provide them.

If several developments are connected, explain the relationship between
them.


### Key Facts and Evidence

Present the most important facts, statistics, dates, organizations,
technologies, or other evidence supported by the sources.

For every important fact, make sure it is actually supported by the
retrieved sources.

Use short explanations rather than creating a list of unexplained facts.


### Why This Matters

Explain the broader significance of the topic.

Discuss the practical implications and effects supported by the sources.

Write this as an explanation, not just a few bullet points.


### Source-Based Analysis

Compare the retrieved sources.

Explain:

- where the sources agree
- where they provide different information
- where they disagree
- what information is missing
- what remains uncertain

Use source numbers where appropriate.

Do not force agreement between sources.


### Conclusion

Give a detailed conclusion that brings together the most important
information.

The conclusion should help the reader understand the overall situation
without introducing any new unsupported facts.


### Suggested Follow-Up Questions

Provide exactly 3 useful questions that a reader could ask next.

Each question must be directly related to the current topic.

Format:

- Question 1
- Question 2
- Question 3


WRITING STYLE:

- Detailed
- Clear
- Educational
- Natural
- Well organized
- Easy to understand
- Multiple paragraphs
- Useful subheadings
- Explain concepts instead of merely listing them
- Avoid unnecessary repetition
- Prioritize useful information over brevity

IMPORTANT FINAL RULE:

Do not produce a short answer when the retrieved sources contain enough
information for a detailed explanation.

The user specifically wants to UNDERSTAND the topic, not merely receive
a list of search results.
"""

    return prompt


# ============================================================
# GEMINI API FUNCTION
# ============================================================

def generate_summary(
    query: str,
    sources_block: str,
    mode: str = DEFAULT_MODE
) -> str:
    """
    Send the user's query and retrieved sources to Gemini.

    Args:
        query:
            The user's research question.

        sources_block:
            Cleaned and formatted Tavily search results.

        mode:
            Research mode such as quick, deep, or simple.

    Returns:
        Gemini's detailed research explanation.

    Raises:
        ValueError:
            If API key or source content is missing.

        RuntimeError:
            If Gemini API request fails.
    """

    # --------------------------------------------------------
    # Validate Gemini API key
    # --------------------------------------------------------

    if not GEMINI_API_KEY:
        raise ValueError(
            "Gemini API key is missing. "
            "Please add GEMINI_API_KEY to your .env file."
        )

    # --------------------------------------------------------
    # Validate source content
    # --------------------------------------------------------

    if not sources_block or not sources_block.strip():
        raise ValueError(
            "No source content was available to analyze."
        )

    # --------------------------------------------------------
    # Get research mode configuration
    # --------------------------------------------------------

    mode_config = RESEARCH_MODES.get(
        mode,
        RESEARCH_MODES[DEFAULT_MODE]
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = _build_prompt(
        query=query,
        sources_block=sources_block,
        mode=mode
    )

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(

                # System-level instructions
                system_instruction=SYSTEM_INSTRUCTION,

                # Allow a much longer response
                max_output_tokens=12000,

                # Lower temperature keeps research answers
                # more consistent and grounded.
                temperature=0.3,
            ),
        )

    except Exception as error:

        # Print detailed error in terminal for debugging
        print("\n========================================")
        print("          GEMINI API ERROR")
        print("========================================")
        print("Error type:", type(error).__name__)
        print("Error message:", error)
        print("========================================\n")

        raise RuntimeError(
            f"Gemini API error: {error}"
        ) from error

    # --------------------------------------------------------
    # Validate Gemini response
    # --------------------------------------------------------

    if not response:
        raise RuntimeError(
            "Gemini returned no response."
        )

    if not getattr(response, "text", None):
        raise RuntimeError(
            "Gemini returned an empty response. "
            "Please try again."
        )

    # --------------------------------------------------------
    # Return generated explanation
    # --------------------------------------------------------

    return response.text