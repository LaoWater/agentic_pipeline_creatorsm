# llm_services.py
import json
import re  # Added for regex operations
from typing import Literal
import logging

# Google Generative AI imports
from google import genai
from google.api_core import exceptions as google_exceptions
# Ensure necessary types are available, if FinishReason needs explicit import, it would be here.
# For now, we'll assume genai.types.FinishReason is the correct path.
from google.genai.types import GenerateContentConfig, HttpOptions

from visual_model import get_google_genai_client  # Assuming this is not used for google_gemini_client init

from config import (
    DECISION_LLM_MODEL,
    PLATFORM_LLM_MODEL,
    PLATFORM_PROMPT_MAP,
    TRANSLATOR_TASK_PROMPT
)
from data_models import (
    Layer2Input, Layer2Output,
    PlatformAgentInput, PlatformAgentOutput,
    TranslatorAgentInput, TranslatorAgentOutput
)

# Setup logger if visual_model.py doesn't set a global one
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Initialize Google Genai client with proper configuration
google_gemini_client = get_google_genai_client()


def _clean_llm_json_output(raw_output: str) -> str:
    """
    Cleans the raw LLM string output to extract a JSON object.
    Handles:
    - UTF-8 BOM.
    - Leading/trailing whitespace.
    - Markdown code blocks (e.g., ```json ... ``` or ``` ... ```) wrapping the entire content.
    """
    if not isinstance(raw_output, str):
        logger.warning(f"Expected string output from LLM, got {type(raw_output)}. Returning empty string.")
        return ""

    text = raw_output

    # 1. Remove UTF-8 BOM if it's at the very start of the raw string
    if text.startswith('\ufeff'):
        text = text[1:]

    # 2. Strip leading/trailing whitespace. This is important to do before regex for markdown.
    text = text.strip()

    # 3. Attempt to remove markdown code block fences if they wrap the entire content.
    markdown_match = re.fullmatch(r"```(?:json\b)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if markdown_match:
        processed_text = markdown_match.group(1).strip()
    else:
        processed_text = text

    return processed_text


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
Do not use special markdown characters to cover text, just simply output the json object text. 
"""


async def run_layer_2_decision_maker(inputs: Layer2Input) -> Layer2Output:
    logger.info("--- Running Layer 2: Decision Maker Strategist ---")

    task_details_content = LAYER_2_SYSTEM_PROMPT.format(
        company_name=inputs["company_name"],
        company_mission=inputs["company_mission"],
        company_sentiment=inputs["company_sentiment"],
        platforms_to_target=", ".join(inputs["platforms_to_target"]),
        tone=inputs["tone"],
        platform_post_type=inputs["platform_post_type"]
    )
    human_message_content = f"""
Subject to address: {inputs['subject']}
Specific requirements: {json.dumps(inputs['requirements']) if inputs['requirements'] else 'None'}
Posts history: {json.dumps(inputs['posts_history']) if inputs['posts_history'] else 'No past history provided.'}
Please provide your strategic decision in the specified JSON format.
"""
    full_contents_prompt = f"{task_details_content}\n\n{human_message_content}"
    logger.info(f"[LAYER 2] About to invoke Gemini LLM ({DECISION_LLM_MODEL}) using new async method.")

    raw_llm_output_str = ""
    cleaned_json_str = ""

    try:
        config = GenerateContentConfig(
            temperature=0.7
        )
        response = await google_gemini_client.aio.models.generate_content(
            model=DECISION_LLM_MODEL,
            contents=full_contents_prompt,
            config=config
        )

        if not response.candidates:
            raise Exception("Layer 2 Gemini call returned no candidates.")

        candidate = response.candidates[0]
        text_from_llm = None
        if candidate.content and candidate.content.parts:
            first_part = candidate.content.parts[0]
            if hasattr(first_part, "text") and isinstance(first_part.text, str):
                text_from_llm = first_part.text

        if text_from_llm is not None:
            raw_llm_output_str = text_from_llm
        else:
            raise Exception("Layer 2 Gemini call returned empty or malformed content (no valid text in primary part).")

        logger.info(
            f"\n--- Layer 2: Raw LLM Response (before cleaning) ---\n{raw_llm_output_str}\n--- End of Raw LLM Response ---")

        cleaned_json_str = _clean_llm_json_output(raw_llm_output_str)
        logger.info(
            f"\n--- Layer 2: Cleaned LLM Response (for JSON parsing) ---\n{cleaned_json_str}\n--- End of Cleaned LLM Response ---")

        if not cleaned_json_str:
            logger.error(
                f"After cleaning, the LLM output string is empty. Original raw output was: '{raw_llm_output_str}'")
            raise json.JSONDecodeError("Cleaned LLM output is empty, cannot parse JSON.",
                                       cleaned_json_str if cleaned_json_str else " ",
                                       0)

        parsed_response: Layer2Output = json.loads(cleaned_json_str)
        logger.info(f"Layer 2 Decision (Parsed): Core Text: {parsed_response.get('core_post_text', '')[:100]}...")
        return parsed_response

    except google_exceptions.GoogleAPIError as e:
        logger.critical(f"[LAYER 2 CRITICAL] Google Gemini API Error: {e}", exc_info=True)
        raise Exception(f"Layer 2 failed due to Google Gemini API Error: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"[LAYER 2 ERROR] JSON parsing error: {e.msg} (line {e.lineno} col {e.colno})", exc_info=True)
        doc_preview = e.doc[:500] + "..." if len(e.doc) > 500 else e.doc
        logger.error(f"Problematic document for json.loads() (cleaned string): '{doc_preview}'")
        if raw_llm_output_str and raw_llm_output_str != e.doc:
            logger.error(f"Original raw LLM output (before cleaning):\n{raw_llm_output_str}")
        raise Exception(f"Layer 2 failed due to JSON parsing error: {e.msg}") from e
    except Exception as e:
        logger.error(f"Error in Layer 2 (Generic Exception): {type(e).__name__} - {e}", exc_info=True)
        if raw_llm_output_str:
            logger.error(f"Raw LLM output at time of generic error (before cleaning):\n{raw_llm_output_str}")
        raise


async def run_platform_adaptation_agent(inputs: PlatformAgentInput) -> PlatformAgentOutput:
    target_platform_lower = inputs['target_platform'].lower()
    logger.info(f"\n--- Running Layer 3: Platform Adaptation for {target_platform_lower} ---")

    system_prompt_template_str = PLATFORM_PROMPT_MAP.get(target_platform_lower)
    if not system_prompt_template_str:
        logger.error(f"No system prompt defined for platform: {inputs['target_platform']}")
        raise ValueError(f"No system prompt defined for platform: {inputs['target_platform']}")

    platform_post_type_value = inputs["platform_post_type"]
    media_type_for_prompt: Literal["image", "video"] = "video" if platform_post_type_value == "Video" else "image"
    company_name_no_spaces = inputs["company_name"].replace(" ", "")

    format_kwargs = {
        "company_name": inputs["company_name"],
        "company_mission": inputs["company_mission"],
        "company_sentiment": inputs["company_sentiment"],
        "language": inputs["language"],
        "subject": inputs["subject"],
        "core_post_text_suggestion": inputs["core_post_text_suggestion"],
        "platform_post_type": inputs["platform_post_type"],
        "media_type_for_prompt": media_type_for_prompt,
        "company_name_no_spaces": company_name_no_spaces
    }
    formatted_system_prompt = system_prompt_template_str.format(**format_kwargs)

    human_message_content = f"Please generate the tailored content for {inputs['target_platform']}. Remember to output in the specified JSON format."
    full_prompt = f"{formatted_system_prompt}\n\n{human_message_content}"

    logger.info(f"[PLATFORM AGENT {target_platform_lower}] About to invoke Gemini LLM ({PLATFORM_LLM_MODEL}).")
    raw_llm_output_str = ""
    cleaned_json_str = ""
    try:
        config = GenerateContentConfig(
            temperature=0.8
        )
        response = await google_gemini_client.aio.models.generate_content(
            model=PLATFORM_LLM_MODEL,
            contents=full_prompt,
            config=config
        )

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason_msg = getattr(response.prompt_feedback, 'block_reason_message', 'No message')
            logger.error(
                f"[PLATFORM AGENT {target_platform_lower}] Gemini call failed: Prompt blocked. Reason: {response.prompt_feedback.block_reason.name} - {block_reason_msg}")
            raise Exception(
                f"Platform Agent ({target_platform_lower}) Gemini call failed: Prompt blocked. Reason: {response.prompt_feedback.block_reason.name} - {block_reason_msg}")

        if not response.candidates:
            logger.error(f"[PLATFORM AGENT {target_platform_lower}] Gemini call returned no candidates.")
            raise Exception(f"Platform Agent ({target_platform_lower}) Gemini call returned no candidates.")

        candidate = response.candidates[0]

        # Log the type and value of finish_reason for debugging if needed
        # logger.debug(f"[PLATFORM AGENT {target_platform_lower}] Candidate finish_reason type: {type(candidate.finish_reason)}, value: {candidate.finish_reason}")

        # Corrected FinishReason path: genai.types.FinishReason instead of genai.types.Candidate.FinishReason
        # Ensure that `genai.types.FinishReason` is the correct enum object.
        # This requires `FinishReason` to be available under `google.generativeai.types`
        if candidate.finish_reason is not None and candidate.finish_reason not in (genai.types.FinishReason.STOP,
                                                                                   genai.types.FinishReason.MAX_TOKENS):
            finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(
                candidate.finish_reason)
            safety_ratings_info = str(candidate.safety_ratings) if candidate.safety_ratings else "N/A"
            logger.error(
                f"[PLATFORM AGENT {target_platform_lower}] Gemini call finished unexpectedly. Reason: {finish_reason_name}. Safety Ratings: {safety_ratings_info}")
            raise Exception(
                f"Platform Agent ({target_platform_lower}) Gemini call finished unexpectedly. Reason: {finish_reason_name}. Safety Ratings: {safety_ratings_info}")
        elif candidate.finish_reason is None:
            logger.warning(
                f"[PLATFORM AGENT {target_platform_lower}] Candidate finish_reason is None. Proceeding, but this might indicate an issue.")

        if candidate.content and candidate.content.parts:
            part_texts = [part.text for part in candidate.content.parts if
                          hasattr(part, 'text') and isinstance(part.text, str)]
            raw_llm_output_str = "".join(part_texts)

        if not raw_llm_output_str:
            logger.error(f"[PLATFORM AGENT {target_platform_lower}] Gemini call returned no text content from parts.")
            raise Exception(
                f"Platform Agent ({target_platform_lower}) Gemini call returned empty or malformed content (no text).")

        logger.info(
            f"\n--- Platform Agent ({inputs['target_platform']}): Raw LLM Response (before cleaning) ---\n{raw_llm_output_str}\n--- End Raw LLM Response ---")

        cleaned_json_str = _clean_llm_json_output(raw_llm_output_str)
        logger.info(
            f"\n--- Platform Agent ({inputs['target_platform']}): Cleaned LLM Response (for JSON parsing) ---\n{cleaned_json_str}\n--- End Cleaned LLM Response ---")

        if not cleaned_json_str:
            logger.error(
                f"[PLATFORM AGENT {target_platform_lower}] After cleaning, LLM output string is empty. Original raw: '{raw_llm_output_str}'")
            raise json.JSONDecodeError("Cleaned LLM output is empty, cannot parse JSON.",
                                       cleaned_json_str if cleaned_json_str else " ", 0)

        parsed_response: PlatformAgentOutput = json.loads(cleaned_json_str)

        logger.info(
            f"Platform Agent ({inputs['target_platform']}) Text: {parsed_response.get('platform_specific_text', '')[:100]}...")
        if parsed_response.get('platform_media_generation_prompt'):
            logger.info(
                f"Platform Agent ({inputs['target_platform']}) Media Prompt: {parsed_response['platform_media_generation_prompt'][:100]}...")
        return parsed_response

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"[PLATFORM AGENT ERROR] Google Gemini API Error for {inputs['target_platform']}: {e}",
                     exc_info=True)
        raise Exception(
            f"Platform Agent for {inputs['target_platform']} failed due to Google Gemini API Error: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(
            f"[PLATFORM AGENT ERROR] JSON parsing error for {inputs['target_platform']}: {e.msg} (line {e.lineno} col {e.colno})",
            exc_info=True)
        doc_preview = e.doc[:500] + "..." if len(e.doc) > 500 else e.doc
        logger.error(f"Problematic document for json.loads() (cleaned string): '{doc_preview}'")
        if raw_llm_output_str and raw_llm_output_str != e.doc:
            logger.error(
                f"Original raw LLM output (before cleaning) for {inputs['target_platform']}:\n{raw_llm_output_str}")
        raise Exception(
            f"Platform Agent for {inputs['target_platform']} failed due to JSON parsing error: {e.msg}") from e
    except AttributeError as e:  # Catch AttributeError specifically if it persists
        logger.error(f"AttributeError in Platform Adaptation for {inputs['target_platform']}: {e}", exc_info=True)
        # Log relevant objects if helpful
        if 'candidate' in locals():
            logger.error(f"Candidate object: {candidate}")
            if hasattr(candidate, 'finish_reason'):
                logger.error(
                    f"Candidate finish_reason: {candidate.finish_reason}, type: {type(candidate.finish_reason)}")
            else:
                logger.error("Candidate has no 'finish_reason' attribute.")
        raise
    except Exception as e:
        logger.error(f"Error in Platform Adaptation for {inputs['target_platform']}: {type(e).__name__} - {e}",
                     exc_info=True)
        if raw_llm_output_str:
            logger.error(
                f"Raw LLM output (before cleaning) on error for {inputs['target_platform']}:\n{raw_llm_output_str}")
        raise



async def run_translator_agent(inputs: TranslatorAgentInput) -> TranslatorAgentOutput:
    logger.info(f"--- Running Final Translator Agent for language: {inputs['target_language']} ---")

    logger.debug(f"[TRANSLATOR AGENT DEBUG] Inputs received:\n{json.dumps(inputs, indent=2)}")


    task_details_content = TRANSLATOR_TASK_PROMPT.format(text_to_translate=inputs["text_to_translate"],
                                                         target_language=inputs["target_language"],
                                                         company_name=inputs["company_name"],
                                                         company_mission=inputs["company_mission"],
                                                         original_subject=inputs["original_subject"],
                                                         # target_platform=inputs['target_platform']
                                                         )
    human_message_content = f"""
Text to translate (from English to {inputs['target_language']}):
"{inputs['text_to_translate']}"

Please provide the translation in the specified JSON format.
"""
    full_contents_prompt = f"{task_details_content}\n\n{human_message_content}"

    logger.info(
        f"[TRANSLATOR AGENT {inputs['target_language']}] About to invoke Gemini LLM ({PLATFORM_LLM_MODEL}) using new async method.")

    raw_llm_output_str = ""
    cleaned_json_str = ""
    try:
        config = GenerateContentConfig(
            temperature=0.7
        )
        response = await google_gemini_client.aio.models.generate_content(
            model=PLATFORM_LLM_MODEL,
            contents=full_contents_prompt,
            config=config
        )

        if not response.candidates:
            raise Exception(f"Translator Agent ({inputs['target_language']}) Gemini call returned no candidates.")

        candidate = response.candidates[0]
        text_from_llm = None
        if candidate.content and candidate.content.parts:
            first_part = candidate.content.parts[0]
            if hasattr(first_part, "text") and isinstance(first_part.text, str):
                text_from_llm = first_part.text

        if text_from_llm is not None:
            raw_llm_output_str = text_from_llm
        else:
            raise Exception(
                f"Translator Agent ({inputs['target_language']}) Gemini call returned empty or malformed content (no valid text in primary part).")

        logger.info(
            f"\n--- Translator Agent: Raw LLM Response (before cleaning) ---\n{raw_llm_output_str}\n--- End of Raw LLM Response ---")

        cleaned_json_str = _clean_llm_json_output(raw_llm_output_str)
        logger.info(
            f"\n--- Translator Agent: Cleaned LLM Response (for JSON parsing) ---\n{cleaned_json_str}\n--- End of Cleaned LLM Response ---")

        if not cleaned_json_str:
            logger.error(
                f"[TRANSLATOR AGENT {inputs['target_language']}] After cleaning, LLM output string is empty. Original raw: '{raw_llm_output_str}'")
            raise json.JSONDecodeError("Cleaned LLM output is empty, cannot parse JSON.",
                                       cleaned_json_str if cleaned_json_str else " ", 0)

        parsed_response: TranslatorAgentOutput = json.loads(cleaned_json_str)
        logger.info(
            f"Translator Agent: Translated text ({inputs['target_language']}) generated: {parsed_response.get('translated_text', '')[:100]}...")
        return parsed_response

    except json.JSONDecodeError as e:
        logger.error(
            f"[TRANSLATOR AGENT ERROR] JSON parsing error for {inputs['target_language']}: {e.msg} (line {e.lineno} col {e.colno})",
            exc_info=True)
        doc_preview = e.doc[:500] + "..." if len(e.doc) > 500 else e.doc
        logger.error(f"Problematic document for json.loads() (cleaned string): '{doc_preview}'")
        if raw_llm_output_str and raw_llm_output_str != e.doc:
            logger.error(
                f"Original raw LLM output (before cleaning) for {inputs['target_language']}:\n{raw_llm_output_str}")
        raise Exception(
            f"Translator Agent for {inputs['target_language']} failed due to JSON parsing error: {e.msg}") from e
    except Exception as e:
        logger.error(f"Error in Translator Agent for {inputs['target_language']}: {type(e).__name__} - {e}",
                     exc_info=True)
        if raw_llm_output_str:
            logger.error(
                f"Raw LLM output (before cleaning) on error for {inputs['target_language']}:\n{raw_llm_output_str}")
        raise
