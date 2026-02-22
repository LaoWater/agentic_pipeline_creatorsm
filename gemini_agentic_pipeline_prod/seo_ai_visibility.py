# seo_ai_visibility.py
"""
SEO Pillar 2: AI Visibility Check.
Queries AI systems (ChatGPT) to determine how well they know about a brand
and whether they would recommend it.
"""

import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SEO_AI_VISIBILITY_MODEL = os.environ.get("SEO_AI_VISIBILITY_MODEL", "gpt-4.1-mini")


def _get_openai_client() -> AsyncOpenAI:
    """Create a fresh AsyncOpenAI client."""
    return AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY_VALUE", ""),
        max_retries=2,
        timeout=30.0,
    )


async def check_ai_visibility(
    company_name: str,
    company_description: str,
    website_url: str,
    keywords: list[str],
) -> dict:
    """
    Check how visible a brand is to AI systems.

    Pipeline:
    1. Query OpenAI about the company
    2. Score the response for awareness and accuracy
    3. Return structured results

    Returns:
        dict with overall_score, systems breakdown, key_findings, recommendations
    """
    logger.info(f"[SEO AI Visibility] Checking AI awareness for '{company_name}'")

    keywords_str = ", ".join(keywords[:5]) if keywords else "general"

    # Step 1: Ask ChatGPT about the company
    chatgpt_result = await _query_chatgpt(company_name, keywords_str)

    # Step 2: Score the response
    scoring = await _score_ai_response(
        company_name=company_name,
        company_description=company_description,
        website_url=website_url,
        ai_response=chatgpt_result.get("response", ""),
    )

    # Build result
    overall_score = scoring.get("score", 0)
    awareness = _score_to_awareness(overall_score)

    result = {
        "overall_score": overall_score,
        "systems": {
            "chatgpt": {
                "model": SEO_AI_VISIBILITY_MODEL,
                "score": overall_score,
                "awareness": awareness,
                "response_summary": chatgpt_result.get("response", "")[:500],
                "accuracy_notes": scoring.get("accuracy_notes", ""),
                "knows_company": scoring.get("knows_company", False),
                "sentiment": scoring.get("sentiment", "neutral"),
            },
        },
        "key_findings": scoring.get("key_findings", []),
        "recommendations": scoring.get("recommendations", []),
    }

    logger.info(
        f"[SEO AI Visibility] Complete: score={overall_score}, awareness={awareness}"
    )
    return result


async def _query_chatgpt(company_name: str, keywords: str) -> dict:
    """Ask ChatGPT what it knows about a company."""
    api_key = os.environ.get("OPENAI_API_KEY_VALUE", "").strip()
    if not api_key:
        logger.warning("[SEO AI Visibility] OPENAI_API_KEY_VALUE not set, skipping ChatGPT query")
        return {"response": "", "error": "API key not configured"}

    prompt = (
        f"What do you know about {company_name}? What do they do? "
        f"How would you rate them in the {keywords} space? "
        f"If you don't know about them specifically, say so honestly."
    )

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=SEO_AI_VISIBILITY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer honestly about what you know about companies and brands. If you don't have specific information, say so clearly.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        content = response.choices[0].message.content if response.choices else ""
        return {"response": content}
    except Exception as e:
        logger.error(f"[SEO AI Visibility] ChatGPT query failed: {e}")
        return {"response": "", "error": str(e)}


async def _score_ai_response(
    company_name: str,
    company_description: str,
    website_url: str,
    ai_response: str,
) -> dict:
    """Score the AI's response for awareness and accuracy."""
    if not ai_response:
        return {
            "score": 0,
            "knows_company": False,
            "accuracy_notes": "AI system could not be queried.",
            "sentiment": "neutral",
            "key_findings": ["AI visibility check could not be completed."],
            "recommendations": [
                "Ensure your brand has a strong online presence with consistent information across multiple sources.",
                "Create authoritative content that AI systems can learn from.",
            ],
        }

    api_key = os.environ.get("OPENAI_API_KEY_VALUE", "").strip()
    if not api_key:
        return _heuristic_score(company_name, ai_response)

    prompt = f"""You are evaluating how well an AI system knows about a specific company.

Company: {company_name}
Website: {website_url}
Actual description: {company_description}

The AI system responded:
"{ai_response}"

Score this response:
1. Does the AI actually know about this specific company? (not just guessing based on the name)
2. Is the information accurate based on the actual description?
3. Would the AI recommend this company to someone asking about their space?

Return a JSON object with:
- "score": 0-100 (0=no awareness, 100=perfect knowledge)
- "knows_company": boolean (does it genuinely know the company, not just guessing?)
- "accuracy_notes": brief assessment of accuracy
- "sentiment": "positive" | "neutral" | "negative"
- "key_findings": array of 2-3 key observations
- "recommendations": array of 2-3 actionable suggestions to improve AI visibility"""

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=SEO_AI_VISIBILITY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI visibility analyst. Score how well AI systems know about brands. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        import json

        content = response.choices[0].message.content if response.choices else "{}"
        parsed = json.loads(content)

        # Normalize score
        score = parsed.get("score", 0)
        try:
            score = max(0, min(int(score), 100))
        except (ValueError, TypeError):
            score = 0

        return {
            "score": score,
            "knows_company": parsed.get("knows_company", False),
            "accuracy_notes": parsed.get("accuracy_notes", ""),
            "sentiment": parsed.get("sentiment", "neutral"),
            "key_findings": parsed.get("key_findings", []),
            "recommendations": parsed.get("recommendations", []),
        }
    except Exception as e:
        logger.error(f"[SEO AI Visibility] Scoring failed: {e}")
        return _heuristic_score(company_name, ai_response)


def _heuristic_score(company_name: str, ai_response: str) -> dict:
    """Fallback heuristic scoring when LLM scoring fails."""
    response_lower = ai_response.lower()
    name_lower = company_name.lower()

    # Check if the AI mentions the company name
    mentions_name = name_lower in response_lower

    # Check for "don't know" / "not familiar" indicators
    no_knowledge_phrases = [
        "don't have specific information",
        "not familiar with",
        "don't have data",
        "i'm not aware",
        "i don't have information",
        "no specific information",
        "unable to find",
        "not able to provide",
    ]
    denies_knowledge = any(phrase in response_lower for phrase in no_knowledge_phrases)

    if denies_knowledge:
        score = 5
        knows = False
    elif mentions_name:
        score = 45
        knows = True
    else:
        score = 15
        knows = False

    return {
        "score": score,
        "knows_company": knows,
        "accuracy_notes": "Scored via heuristic fallback (LLM scoring unavailable).",
        "sentiment": "neutral",
        "key_findings": [
            f"AI {'acknowledges' if knows else 'does not recognize'} the brand '{company_name}'.",
            "Heuristic scoring was used — results are approximate.",
        ],
        "recommendations": [
            "Build authoritative backlinks and mentions across high-quality sources.",
            "Create comprehensive, factual content about your brand that AI can reference.",
            "Get listed in industry directories and knowledge bases.",
        ],
    }


def _score_to_awareness(score: int) -> str:
    """Convert numeric score to awareness level."""
    if score >= 75:
        return "excellent"
    elif score >= 50:
        return "good"
    elif score >= 25:
        return "partial"
    return "none"
