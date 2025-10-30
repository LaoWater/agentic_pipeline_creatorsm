# config.py
from dotenv import dotenv_values
import os

# --- API Keys & Environment ---
config = dotenv_values(".env")


def get_openai_api_key(config: dict) -> str:
    # 1. Try config dict
    key = config.get("OPENAI_API_KEY_VALUE")
    if key:
        print("[INFO] Using OPENAI_API_KEY_VALUE from config.")
        return key

    # 2. Try environment variable
    key = os.environ.get("OPENAI_API_KEY_VALUE")
    if key:
        print("[INFO] Using OPENAI_API_KEY_VALUE from environment.")
        return key

    # 3. Fallback failed
    raise ValueError("Missing OPENAI_API_KEY_VALUE. Set it in config or as an environment variable.")



OPENAI_API_KEY_VALUE = get_openai_api_key(config)


# --- LLM Model Configuration ---
DECISION_LLM_MODEL = "gemini-2.5-flash"
PLATFORM_LLM_MODEL = "gemini-2.5-flash"

# --- Image Generation Model Configuration ---
IMAGE_GENERATION_MODEL = "imagen-4.0-fast-generate-001"  # Centralized image generation model for all visual asset creation

# --- Output Configuration ---
if os.environ.get("RUNNING_IN_DOCKER") == "true":
    BASE_OUTPUT_FOLDER = "/app/output_data"
    print(f"RUNNING IN DOKCER: True, BASE_OUTPUT_FOLDER: {BASE_OUTPUT_FOLDER}")
else:
    BASE_OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data_local")
    print(f"RUNNING IN DOKCER: False, BASE_OUTPUT_FOLDER: {BASE_OUTPUT_FOLDER}")


IMAGE_FILE_EXTENSION = "png"  # Can be moved to config.py if needed




TRANSLATOR_TASK_PROMPT  = """
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
- Original Subject: {original_subject}

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



# --- Layer 3: Platform Adaptation LLM Prompts ---
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
    -   If `platform_post_type` is "Image" or "Video": Craft a `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 16:9 (landscape) or 1:1 (square).
    -   CRITICAL: The media prompt must describe ONLY visual elements (scenes, objects, colors, composition, style). Do NOT include text, captions, labels, words, hashtags, or technical specifications (like hex codes) in the media prompt.
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for LinkedIn.
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.
4. Always output in English.

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
    -   If `platform_post_type` is "Image" or "Video": Craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio square 1:1 or portrait 3:4. Emphasize "{company_sentiment}".
    -   CRITICAL: The media prompt must describe ONLY visual elements (scenes, objects, colors, composition, style). Do NOT include text, captions, labels, words, hashtags, or technical specifications (like hex codes) in the media prompt.
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for Instagram.
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.
4. Always output in English.

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
    -   CRITICAL: The media prompt must describe ONLY visual elements (scenes, objects, colors, composition, style). Do NOT include text, captions, labels, words, hashtags, or technical specifications (like hex codes) in the media prompt.
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for Twitter (X).
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.
4. Always output in English.

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
    -   If `platform_post_type` is "Image" or "Video": Craft `platform_media_generation_prompt` for an {media_type_for_prompt}. Aspect ratio 16:9 (landscape) or 1:1 (square). Emphasize "{company_sentiment}".
    -   CRITICAL: The media prompt must describe ONLY visual elements (scenes, objects, colors, composition, style). Do NOT include text, captions, labels, words, hashtags, or technical specifications (like hex codes) in the media prompt.
    -   If `platform_post_type` is "Text": The `platform_media_generation_prompt` must be null.
    -   If `platform_post_type` is `Let Model Decide`, you will decide type of post based on subject and your working platform. (Text, Photo) - then adhere `platform_media_generation_prompt` according to this decision.

Your Tasks:
1.  Craft `platform_specific_text` for Facebook.
2.  If `platform_post_type` is "Image" or "Video" or decided by model for media, you MUST generate a `platform_media_generation_prompt`.
3.  If `platform_post_type` is "Text", ensure `platform_media_generation_prompt` is null in the output JSON.
4. Always output in English.

Output Format:
Return a single JSON object with keys "platform_specific_text" (string) and "platform_media_generation_prompt" (string or null).
"""

PLATFORM_PROMPT_MAP = {
    "linkedin": LINKEDIN_SYSTEM_PROMPT,
    "instagram": INSTAGRAM_SYSTEM_PROMPT,
    "twitter": TWITTER_SYSTEM_PROMPT,
    "facebook": FACEBOOK_SYSTEM_PROMPT,
}


# # --- Company & Request Configuration ---
# COMPANY_NAME = "Creators Multiverse"
# COMPANY_MISSION = "Empowering creators to build their digital presence with AI-powered tools that transform ideas into viral content across platforms"
# COMPANY_SENTIMENT = ("Inspirational & Empowering. Cosmic/Magical Theme yet not too much."
#                      "The brand positions itself as a creative partner that amplifies human creativity rather than replacing it.")
# # This will be passed to the main orchestrator, but a default can be here
# DEFAULT_POST_SUBJECT = "Hello World! Intro post about our company, starting out, vision, etc"
#
# LANGUAGE = "English"
#
# TONE = 'Casual/Friendly'
#
# PLATFORMS_POST_TYPES_MAP = [
#     {"linkedin": "Image"}, # "Text", "Image", or "Video"
#     # {"facebook": "Text"},
#     {"twitter": "Text"},
#     {"instagram": "Text"}
# ]