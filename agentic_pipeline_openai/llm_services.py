# llm_services.py
import json
from typing import Literal

import requests
from openai import APIConnectionError, RateLimitError, APIStatusError, OpenAIError
import httpx # Langchain's OpenAI client often uses httpx


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from config import (
    OPENAI_API_KEY_VALUE,
    DECISION_LLM_MODEL,
    PLATFORM_LLM_MODEL
)
from data_models import (
    Layer2Input, Layer2Output,
    PlatformAgentInput, PlatformAgentOutput,
    TranslatorAgentInput, TranslatorAgentOutput
)

# --- LLM Initialization ---
decision_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY_VALUE,
    model=DECISION_LLM_MODEL,
    temperature=1,
)

platform_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY_VALUE,
    model=PLATFORM_LLM_MODEL,
    temperature=1.1,
)

# --- Layer 2: Core Text Generation LLM - Strategist ---
LAYER_2_SYSTEM_PROMPT = """
You are a Master Social Media Strategist and Content Planner for {company_name}.
Your Mission: Analyze the provided information and write a foundational `core_post_text` for the given subject. 
This text will be later adapted for specific social media platforms, each with a pre-defined content format (e.g., Text-only, Text + Image, or Text + Video).

Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Target Platforms: {platforms_to_target}
Post Tonality/Sentiment: {tone}
Post Type: {platform_post_type}

Your Task:
1.  Consider the `subject`, `company_mission`, `company_sentiment` and `target_platforms`. Adhere to Tonality yet don't push it too hard, allow free-flow as well.
Also consider Post Type and bind the text in sync with the type.
2.  Write a versatile `core_post_text`. This text is a foundational message that platform-specific Agents will adapt. 
It should be rich enough to support text-only, image-accompanied, or video-accompanied posts depending on the target platform's specific needs.


Output Format:
Return a single JSON object with the following key:
- "core_post_text": string
"""


async def run_layer_2_decision_maker(inputs: Layer2Input) -> Layer2Output:
    print("\n--- Running Layer 2: Decision Maker Strategist ---")
    system_message_content = LAYER_2_SYSTEM_PROMPT.format(
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        company_sentiment=inputs["company_sentiment"],
        platforms_to_target=", ".join(inputs["platforms_to_target"]),
        tone=inputs["tone"],
        language=inputs["language"],
        platform_post_type=inputs["platform_post_type"]

    )
    human_message_content = f"""
            Subject to address: {inputs['subject']}
            Specific requirements: {json.dumps(inputs['requirements']) if inputs['requirements'] else 'None'}
            Posts history: {json.dumps(inputs['posts_history']) if inputs['posts_history'] else 'No past history provided.'}
            Please provide your strategic decision in the specified JSON format.
        """
    prompt_messages_list = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content)
    ]
    prompt_template = ChatPromptTemplate.from_messages(prompt_messages_list)
    # chain = prompt_template | decision_llm | JsonOutputParser() # Not used directly like this for streaming raw output
    raw_llm_output_str = ""

    print(f"[LAYER 2] About to invoke LLM. API Key (first 5 chars): {OPENAI_API_KEY_VALUE}") # Add this for key check

    try:
        llm_response_message = await (prompt_template | decision_llm).ainvoke({})
        raw_llm_output_str = llm_response_message.content
        print(f"\n--- Layer 2: Raw LLM Response ---\n{raw_llm_output_str}\n--- End of Raw LLM Response ---")

        response: Layer2Output = JsonOutputParser().parse(raw_llm_output_str)
        print(
            f"Layer 2 Decision (Parsed): Core Text: {response.get('core_post_text', '')[:100]}...")
        return response

    # --- MORE SPECIFIC ERROR HANDLING ---
    except APIConnectionError as e:  # OpenAI SDK specific
        print(f"[LAYER 2 CRITICAL] OpenAI APIConnectionError: {e}")
        print(f"   Request URL: {e.request.url if hasattr(e, 'request') and e.request else 'N/A'}")
        print(f"   Error details: {vars(e)}")
        raise Exception(f"Layer 2 failed due to OpenAI APIConnectionError: {e}") from e

    except httpx.ConnectError as e:  # httpx specific connection error
        print(f"[LAYER 2 CRITICAL] httpx.ConnectError: {e}")
        print(f"   Request URL: {e.request.url if hasattr(e, 'request') and e.request else 'N/A'}")
        raise Exception(f"Layer 2 failed due to httpx.ConnectError: {e}") from e

    except requests.exceptions.ConnectionError as e:  # requests specific connection error
        print(f"[LAYER 2 CRITICAL] requests.exceptions.ConnectionError: {e}")
        print(f"   Request URL: {e.request.url if hasattr(e, 'request') and e.request else 'N/A'}")
        raise Exception(f"Layer 2 failed due to requests.exceptions.ConnectionError: {e}") from e

    except RateLimitError as e:
        print(f"[LAYER 2 ERROR] OpenAI RateLimitError: {e}")
        raise Exception(f"Layer 2 failed due to OpenAI RateLimitError: {e}") from e

    except APIStatusError as e:
        print(
            f"[LAYER 2 ERROR] OpenAI APIStatusError: {e.status_code} - {e.response.text if hasattr(e, 'response') else str(e)}")
        raise Exception(f"Layer 2 failed due to OpenAI APIStatusError: {e.status_code}") from e

    except OpenAIError as e:
        print(f"[LAYER 2 ERROR] OpenAI General Error: {type(e).__name__} - {e}")
        raise Exception(f"Layer 2 failed due to OpenAI General Error: {e}") from e
    # --- END SPECIFIC ERROR HANDLING ---

    except Exception as e:  # Fallback for other errors
        # THIS IS THE BLOCK THAT IS CURRENTLY CATCHING THE ERROR
        print(f"Error in Layer 2 (Caught by generic Exception): {type(e).__name__} - {e}")  # PRINT THE TYPE
        if raw_llm_output_str:
            print(f"Problematic raw LLM output on error:\n{raw_llm_output_str}")
        import traceback
        print("Full Traceback for generic Exception in Layer 2:")
        print(traceback.format_exc())  # PRINT THE FULL TRACEBACK
        raise  # Re-raise the original exception to be caught by the FastAPI handler

# --- Layer 3: Platform Adaptation LLM Prompts - allowing Tone to be passed on in the fine-layers - not direct parameter - for possible human-like more feeling Inference.
# --- TODO Final Layer - Handle Translation ----
LINKEDIN_SYSTEM_PROMPT = """
You are an expert Social Media Content Creator for {company_name}, specializing in LinkedIn.
Your goal is to adapt a core message into a professional and engaging LinkedIn post.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}
Required Post Format for LinkedIn: {platform_post_type}

LinkedIn Specific Guidelines:
-   Tone: Professional, insightful, authoritative, value-driven. Align with "{company_sentiment}".
-   Hashtags: Use 3-5 relevant, professional hashtags. Consider #{company_name_no_spaces}.
-   Media:
    -   If `platform_post_type` is "Image" or "Video": Craft a `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 1.91:1 or 1:1 (square for carousel).
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for LinkedIn.
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
"""

INSTAGRAM_SYSTEM_PROMPT = """
You are a creative Social Media Content Creator for {company_name}, specializing in Instagram.
Your goal is to adapt a core message into a visually appealing and engaging Instagram post.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}
Required Post Format for Instagram: {platform_post_type}

Instagram Specific Guidelines:
-   Tone: Engaging, friendly, authentic, visually descriptive. "{company_sentiment}".
-   Hashtags: Use 5-10 relevant hashtags. Mix popular and niche. Include #{company_name_no_spaces}.
-   Media:
    -   If `platform_post_type` is "Image" or "Video": Craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio square 1:1 or portrait 4:5. Emphasize "{company_sentiment}".
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for Instagram.
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
"""

TWITTER_SYSTEM_PROMPT = """
You are a charming and concise Social Media Content Creator for {company_name}, specializing in Twitter (X).
Your goal is to adapt a core message into brief, impactful Tweets.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}
Required Post Format for Twitter (X): {platform_post_type}


Twitter (X) Specific Guidelines:
-   Tone: Conversational, direct, encouraging. "{company_sentiment}" adapted for brevity.
-   Length: Max 280 characters.
-   Hashtags: Use 1-3 highly relevant hashtags (e.g., #{company_name_no_spaces}).
-   Media:
    -   If `platform_post_type` is "Image" or "Video": Craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 16:9 or 1:1.
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for Twitter (X).
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
"""

FACEBOOK_SYSTEM_PROMPT = """
You are a versatile Social Media Content Creator for {company_name}, specializing in Facebook.
Your goal is to adapt a core message into an engaging Facebook post that encourages community interaction.

Company Name: {company_name}
Company Mission: {company_mission}
Company Sentiment: {company_sentiment}
Original Subject: {subject}
Core Post Text Suggestion from Strategist: {core_post_text_suggestion}
Required Post Format for Facebook: {platform_post_type}

Facebook Specific Guidelines:
-   Tone: Friendly, approachable, informative, community-oriented. "{company_sentiment}" should be evident. 
-   Hashtags: Use 1-3 relevant hashtags (e.g., #{company_name_no_spaces}).
-   Media:
    -   If `platform_post_type` is "Image" or "Video": Craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 1.91:1 (landscape) or 1:1 (square). Emphasize "{company_sentiment}".
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for Facebook.
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
"""

PLATFORM_PROMPT_MAP = {
    "linkedin": LINKEDIN_SYSTEM_PROMPT,
    "instagram": INSTAGRAM_SYSTEM_PROMPT,
    "twitter": TWITTER_SYSTEM_PROMPT,
    "facebook": FACEBOOK_SYSTEM_PROMPT,
}


async def run_platform_adaptation_agent(inputs: PlatformAgentInput) -> PlatformAgentOutput:
    target_platform_lower = inputs['target_platform'].lower()
    print(f"\n--- Running Layer 3: Platform Adaptation for {target_platform_lower} ---")

    system_prompt_template_str = PLATFORM_PROMPT_MAP.get(target_platform_lower)
    if not system_prompt_template_str:
        raise ValueError(f"No system prompt defined for platform: {inputs['target_platform']}")

    # Determine the media type for the prompt based on the platform's required post type
    platform_post_type_value = inputs["platform_post_type"]
    if platform_post_type_value == "Video":
        media_type_for_prompt: Literal["image", "video"] = "video"
    else:  # Covers "Image" and "Text". For "Text", prompt instructions make media_prompt null.
        media_type_for_prompt: Literal["image", "video"] = "image"

    company_name_no_spaces = inputs["company_name"].replace(" ", "")

    # Ensure all required placeholders are passed to format
    format_kwargs = {
        "company_name": inputs["company_name"],
        "company_mission": inputs["company_mission"],
        "company_sentiment": inputs["company_sentiment"],
        "language": inputs["language"],  # Added
        "subject": inputs["subject"],
        "core_post_text_suggestion": inputs["core_post_text_suggestion"],
        "platform_post_type": inputs["platform_post_type"],  # Added - FIXES THE KEYERROR
        # "target_platform": inputs["target_platform"], # This is not used as a placeholder in current prompts
        "media_type_for_prompt": media_type_for_prompt,  # Logic updated
        "company_name_no_spaces": company_name_no_spaces
    }
    formatted_system_prompt = system_prompt_template_str.format(**format_kwargs)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(
            content=f"Please generate the tailored content for {inputs['target_platform']}. Remember to output in the specified JSON format.")
    ])
    # chain = prompt_template | platform_llm | JsonOutputParser() # Not used directly like this for streaming raw output
    raw_llm_output_str = ""
    try:
        # To see the raw output before parsing (for debugging)
        llm_response_message = await (prompt_template | platform_llm).ainvoke({})
        raw_llm_output_str = llm_response_message.content
        print(
            f"\n--- Platform Agent ({inputs['target_platform']}): Raw LLM Response ---\n{raw_llm_output_str}\n--- End Raw LLM Response ---")

        response: PlatformAgentOutput = JsonOutputParser().parse(raw_llm_output_str)  # type: ignore

        print(
            f"Platform Agent ({inputs['target_platform']}) Text: {response.get('platform_specific_text', '')[:100]}...")
        if response.get('platform_media_generation_prompt'):
            print(
                f"Platform Agent ({inputs['target_platform']}) Media Prompt: {response['platform_media_generation_prompt'][:100]}...")
        return response
    except Exception as e:
        print(f"Error in Platform Adaptation for {inputs['target_platform']}: {e}")
        if raw_llm_output_str:
            print(f"Problematic raw LLM output on error for {inputs['target_platform']}:\n{raw_llm_output_str}")
        raise


# --- Final Layer - Translator Agent ---
TRANSLATOR_AGENT_SYSTEM_PROMPT = """
You are a genius multi-lingual translator and cultural adaptation specialist for {company_name}.
Your expertise transcends literal word-for-word translation - you are a master of cultural nuance, emotional resonance, and authentic voice preservation across languages.

Your Mission: Transform the provided English social media content into {target_language} while maintaining:
- The original emotional impact and sentiment
- Platform-specific tone and style
- Cultural appropriateness for the target audience
- Brand voice consistency
- Social media engagement potential

Company Context:
- Company Name: {company_name}
- Company Mission: {company_mission}
- Company Sentiment: {company_sentiment}
- Target Platform: {target_platform}
- Original Subject: {subject}

Translation Philosophy:
You don't just translate words - you translate feelings, cultural context, and social dynamics. Consider:
- How does this message resonate in {target_language} culture?
- What local expressions, idioms, or cultural references would make this more authentic?
- How do social media conventions differ in {target_language} speaking regions?
- What hashtags and engagement patterns work best for this audience?

Guidelines:
1. Preserve the core message and call-to-action
2. Adapt hashtags to be culturally relevant and discoverable in the target language
3. Maintain appropriate formality/informality for the platform and culture
4. Ensure the translated text feels native, not translated
5. Keep platform-specific character limits and formatting in mind
6. Preserve any mentions, tags, or special formatting from the original

Your Task:
Translate the provided platform-specific text into {target_language}, creating content that feels authentically crafted for that language and culture, not merely translated.

Output Format:
Return a single JSON object with the following key:
- "translated_text": string (the culturally adapted and translated social media post)
"""


async def run_translator_agent(inputs: TranslatorAgentInput) -> TranslatorAgentOutput:
    print(f"\n--- Running Final Translator Agent for language: {inputs['target_language']} ---")

    system_message_content = TRANSLATOR_AGENT_SYSTEM_PROMPT.format(
        target_language=inputs["target_language"],
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        original_subject=inputs["original_subject"]
    )

    human_message_content = f"""
        Text to translate (from English to {inputs['target_language']}):
        "{inputs['text_to_translate']}"

        Please provide the translation in the specified JSON format.
    """

    prompt_messages_list = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content)
    ]
    prompt_template = ChatPromptTemplate.from_messages(prompt_messages_list)

    raw_llm_output_str = ""
    try:
        llm_response_message = await (prompt_template | platform_llm).ainvoke({})  # Using platform_llm as requested
        raw_llm_output_str = llm_response_message.content
        print(f"\n--- Translator Agent: Raw LLM Response ---\n{raw_llm_output_str}\n--- End of Raw LLM Response ---")

        response: TranslatorAgentOutput = JsonOutputParser().parse(raw_llm_output_str)  # type: ignore
        print(
            f"Translator Agent: Translated text ({inputs['target_language']}) generated: {response.get('translated_text', '')[:100]}...")
        return response
    except Exception as e:
        print(f"Error in Translator Agent for language {inputs['target_language']}: {e}")
        if raw_llm_output_str:
            print(f"Problematic raw LLM output on error (Translator):\n{raw_llm_output_str}")
        raise
