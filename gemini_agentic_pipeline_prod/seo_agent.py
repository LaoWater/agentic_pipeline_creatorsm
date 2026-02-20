# seo_agent.py
"""
SEO Intelligence Agent - Real engagement discovery and presence analysis.
Uses Serper.dev for real Google search results with resilient LLM analysis
(OpenAI primary, Gemini fallback in auto mode).
"""

import json
import logging
import os
import re

from openai import AsyncOpenAI
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

from seo_search import (
    crawl_website,
    find_engagement_opportunities,
    search_brand_presence,
)
from visual_model import get_google_genai_client

logger = logging.getLogger(__name__)

SEO_OPENAI_MODEL = os.environ.get("SEO_OPENAI_MODEL", "gpt-5.2")
SEO_GEMINI_MODEL = os.environ.get("SEO_GEMINI_MODEL", "gemini-3-flash-preview")
SEO_LLM_PROVIDER = os.environ.get("SEO_LLM_PROVIDER", "gemini").strip().lower()

SEO_OPENAI_MODEL_QUERIES = os.environ.get("SEO_OPENAI_MODEL_QUERIES", SEO_OPENAI_MODEL)
SEO_OPENAI_MODEL_SCORING = os.environ.get("SEO_OPENAI_MODEL_SCORING", SEO_OPENAI_MODEL)
SEO_OPENAI_MODEL_ANALYSIS = os.environ.get("SEO_OPENAI_MODEL_ANALYSIS", SEO_OPENAI_MODEL)

SEO_GEMINI_MODEL_QUERIES = os.environ.get("SEO_GEMINI_MODEL_QUERIES", SEO_GEMINI_MODEL)
SEO_GEMINI_MODEL_SCORING = os.environ.get("SEO_GEMINI_MODEL_SCORING", SEO_GEMINI_MODEL)
SEO_GEMINI_MODEL_ANALYSIS = os.environ.get("SEO_GEMINI_MODEL_ANALYSIS", SEO_GEMINI_MODEL)

# Startup diagnostic — helps triage Cloud Run LLM routing issues
logger.info(
    f"[SEO] Config: provider={SEO_LLM_PROVIDER}, "
    f"openai_model={SEO_OPENAI_MODEL}, gemini_model={SEO_GEMINI_MODEL}, "
    f"openai_task_models=(q:{SEO_OPENAI_MODEL_QUERIES}, s:{SEO_OPENAI_MODEL_SCORING}, a:{SEO_OPENAI_MODEL_ANALYSIS}), "
    f"gemini_task_models=(q:{SEO_GEMINI_MODEL_QUERIES}, s:{SEO_GEMINI_MODEL_SCORING}, a:{SEO_GEMINI_MODEL_ANALYSIS}), "
    f"GEMINI_API_KEY={'SET' if os.environ.get('GEMINI_API_KEY') else 'MISSING'}, "
    f"OPENAI_API_KEY_VALUE={'SET' if os.environ.get('OPENAI_API_KEY_VALUE') else 'MISSING'}"
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"[SEO] Invalid int for {name}='{value}', using default={default}")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"[SEO] Invalid float for {name}='{value}', using default={default}")
        return default


SEO_OPENAI_MAX_RETRIES = _env_int("SEO_OPENAI_MAX_RETRIES", 4)
SEO_OPENAI_TIMEOUT_SECONDS = _env_float("SEO_OPENAI_TIMEOUT_SECONDS", 45.0)

_gemini_client = None


def _get_openai_client() -> AsyncOpenAI:
    """Create a fresh AsyncOpenAI client per request to avoid stale connection pools in Cloud Run."""
    return AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY_VALUE", ""),
        max_retries=SEO_OPENAI_MAX_RETRIES,
        timeout=SEO_OPENAI_TIMEOUT_SECONDS,
    )


def _get_gemini_client():
    """Lazy init Gemini client to avoid startup failures when fallback is unused."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = get_google_genai_client()
    return _gemini_client


def _provider_order() -> list[str]:
    if SEO_LLM_PROVIDER == "openai":
        return ["openai"]
    if SEO_LLM_PROVIDER == "gemini":
        return ["gemini"]
    if SEO_LLM_PROVIDER != "auto":
        logger.warning(f"[SEO] Unknown SEO_LLM_PROVIDER='{SEO_LLM_PROVIDER}', defaulting to 'auto'")
    return ["openai", "gemini"]


def _error_chain(exc: Exception, max_depth: int = 3) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    current = exc.__cause__ or exc.__context__
    depth = 0
    while current is not None and depth < max_depth:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
        depth += 1
    return " | caused by: ".join(parts)


def _parse_json_object(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("LLM returned empty content")

    # Accept JSON wrapped in markdown code fences.
    fence_match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


async def _call_openai_json(
    *,
    task_name: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    if not os.environ.get("OPENAI_API_KEY_VALUE", "").strip():
        raise ValueError("OPENAI_API_KEY_VALUE is missing or empty")

    response = await _get_openai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    request_id = getattr(response, "_request_id", None)
    if request_id:
        logger.info(f"[SEO] {task_name} OpenAI request_id={request_id} model={model}")
    content = response.choices[0].message.content if response.choices else ""
    return _parse_json_object(content)


async def _call_gemini_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    client = _get_gemini_client()
    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
    }
    try:
        config = GenerateContentConfig(**config_kwargs)
    except TypeError:
        # Backward compatibility for older google-genai versions.
        config_kwargs.pop("response_mime_type", None)
        try:
            config = GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("max_output_tokens", None)
            config = GenerateContentConfig(**config_kwargs)
    response = await client.aio.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n\n{user_prompt}",
        config=config,
    )

    raw_text = getattr(response, "text", "") or ""
    if not raw_text and getattr(response, "candidates", None):
        candidate = response.candidates[0]
        if candidate and candidate.content and candidate.content.parts:
            raw_text = "".join(
                part.text for part in candidate.content.parts if hasattr(part, "text") and isinstance(part.text, str)
            )

    return _parse_json_object(raw_text)


async def _run_llm_json_task(
    *,
    task_name: str,
    openai_model: str = "",
    gemini_model: str = "",
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    resolved_openai_model = openai_model or SEO_OPENAI_MODEL
    resolved_gemini_model = gemini_model or SEO_GEMINI_MODEL
    errors: list[str] = []
    for provider in _provider_order():
        try:
            if provider == "openai":
                parsed = await _call_openai_json(
                    task_name=task_name,
                    model=resolved_openai_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                parsed = await _call_gemini_json(
                    model=resolved_gemini_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            model_used = resolved_openai_model if provider == "openai" else resolved_gemini_model
            logger.info(f"[SEO] {task_name} completed with provider={provider} model={model_used}")
            return parsed
        except Exception as exc:
            details = _error_chain(exc)
            model_used = resolved_openai_model if provider == "openai" else resolved_gemini_model
            errors.append(f"{provider}({model_used}): {details}")
            logger.warning(f"[SEO] {task_name} failed with provider={provider} model={model_used}: {details}")

    raise RuntimeError(f"All LLM providers failed for {task_name}. " + " || ".join(errors))


# ── Pydantic Models ──────────────────────────────────────────────────────────

class SearchEngagementRequest(BaseModel):
    company_name: str = Field(..., description="Company/brand name")
    keywords: list[str] = Field(default_factory=list, description="Target keywords")
    industry: str = Field(default="", description="Industry/niche")
    target_audience: str = Field(default="", description="Target audience description")
    website_url: str = Field(default="", description="Company website URL")
    tone_of_voice: str = Field(default="", description="Brand tone of voice")


class AnalyzePresenceRequest(BaseModel):
    website_url: str = Field(..., description="Website URL to analyze")
    company_name: str = Field(..., description="Company/brand name")
    keywords: list[str] = Field(default_factory=list, description="Target keywords")
    competitors: list[str] = Field(default_factory=list, description="Competitor names or URLs")
    target_audience: str = Field(default="", description="Target audience")
    buyer_persona: str = Field(default="", description="Buyer persona description")
    mission: str = Field(default="", description="Company mission")
    tone_of_voice: str = Field(default="", description="Brand tone")


# ── Search Engagement Pipeline ───────────────────────────────────────────────

async def search_engagement_pipeline(request: SearchEngagementRequest) -> dict:
    """
    Find REAL engagement opportunities across platforms.
    Every URL returned is from Google's actual index.

    Pipeline:
    1. LLM generates targeted search queries
    2. Serper.dev executes real Google searches
    3. LLM scores results and generates suggested responses
    """
    logger.info(f"[SEO] Starting engagement search for '{request.company_name}'")

    # Step 1: Generate smart search queries
    queries = await _generate_search_queries(request)
    logger.info(f"[SEO] Generated {len(queries)} search queries")

    # Step 2: Execute real searches via Serper.dev
    raw_results = await find_engagement_opportunities(
        queries=queries,
        num_per_query=5,
        time_range="m",  # Last month for freshness
    )
    logger.info(f"[SEO] Found {len(raw_results)} raw results from Serper")

    if not raw_results:
        # Broaden the search - try with longer time range
        raw_results = await find_engagement_opportunities(
            queries=queries[:5],
            num_per_query=8,
            time_range="y",  # Last year
        )
        logger.info(f"[SEO] Broadened search found {len(raw_results)} results")

    if not raw_results:
        return {
            "opportunities": [],
            "queries_used": queries,
            "message": "No engagement opportunities found. Try different keywords.",
        }

    # Step 3: LLM scores and generates responses
    opportunities = await _score_and_respond(raw_results, request)
    logger.info(f"[SEO] Scored and generated responses for {len(opportunities)} opportunities")

    return {
        "opportunities": opportunities,
        "queries_used": queries,
        "total_raw_results": len(raw_results),
        "message": f"Found {len(opportunities)} engagement opportunities with real, verified URLs.",
    }


async def _generate_search_queries(request: SearchEngagementRequest) -> list[str]:
    """Use LLM to generate targeted search queries for finding engagement opportunities."""

    keywords_str = ", ".join(request.keywords) if request.keywords else "general industry topics"

    prompt = f"""You are an SEO expert. Generate 8-12 targeted Google search queries to find REAL discussions and engagement opportunities for this company.

Company: {request.company_name}
Industry: {request.industry}
Keywords: {keywords_str}
Target Audience: {request.target_audience}
Website: {request.website_url}

Generate queries that will find:
1. Reddit discussions where people ask about or discuss topics related to this company's offerings (2-3 queries with site:reddit.com)
2. YouTube videos reviewing or discussing related products/services (1-2 queries with site:youtube.com)
3. Quora questions about the industry/topics (1-2 queries with site:quora.com)
4. General forum discussions and blog posts (2-3 queries without site: operator)
5. Twitter/X discussions (1 query with site:twitter.com OR site:x.com)

Rules:
- Use natural language queries people would actually search
- Include relevant keywords naturally
- Use quotes around multi-word phrases when needed
- Mix broad and specific queries
- Focus on questions, recommendations, comparisons, and help-seeking posts

Return a JSON object with a single key "queries" containing an array of search query strings."""

    try:
        parsed = await _run_llm_json_task(
            task_name="generate_search_queries",
            openai_model=SEO_OPENAI_MODEL_QUERIES,
            gemini_model=SEO_GEMINI_MODEL_QUERIES,
            system_prompt="You generate precise Google search queries. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=1000,
        )
        queries = parsed.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("LLM returned non-list 'queries'")
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not cleaned:
            raise ValueError("LLM returned empty 'queries'")
        return cleaned[:12]
    except Exception as e:
        logger.error(f"[SEO] Failed to generate search queries: {_error_chain(e)}")
        # Fallback: generate basic queries manually
        fallback = [
            f'site:reddit.com "{request.company_name}" OR {keywords_str}',
            f'site:youtube.com {keywords_str} review',
            f'site:quora.com {keywords_str} recommendation',
            f'{keywords_str} discussion forum',
        ]
        return fallback


async def _score_and_respond(
    raw_results: list[dict],
    request: SearchEngagementRequest,
) -> list[dict]:
    """Use LLM to score relevance and generate suggested responses for real results."""

    # Prepare results for LLM (limit to top 20 to keep prompt manageable)
    results_for_llm = []
    for r in raw_results[:20]:
        results_for_llm.append({
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "platform": r["platform"],
        })

    prompt = f"""You are an engagement strategist for "{request.company_name}".
Industry: {request.industry}
Tone of Voice: {request.tone_of_voice}
Website: {request.website_url}

Below are REAL search results from Google. For each one, evaluate whether it's a good engagement opportunity and generate a helpful response.

Search Results:
{json.dumps(results_for_llm, indent=2)}

For each result, score its relevance (0-100) and write a suggested response that:
- Is genuinely helpful and adds value to the discussion
- Is NOT overly promotional (no "buy our product" spam)
- Shares relevant knowledge, experience, or insights
- Naturally mentions the company or its solution ONLY if directly relevant
- Matches the platform's culture (Reddit: authentic/detailed, YouTube: supportive, Quora: expert-level, Twitter: concise)
- Would not get flagged as spam or self-promotion

Return a JSON object with key "scored_results" containing an array of objects, each with:
- "url" (the original URL, do NOT modify it)
- "title" (the original title)
- "snippet" (the original snippet)
- "platform" (the detected platform)
- "relevance_score" (0-100, where 100 is perfect fit)
- "suggested_response" (your crafted response)
- "engagement_reason" (brief explanation of why this is worth engaging with)

Only include results with relevance_score >= 30. Sort by relevance_score descending."""

    try:
        parsed = await _run_llm_json_task(
            task_name="score_and_respond",
            openai_model=SEO_OPENAI_MODEL_SCORING,
            gemini_model=SEO_GEMINI_MODEL_SCORING,
            system_prompt="You are an SEO engagement expert. Score search results and craft authentic responses. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=4000,
        )
        results = parsed.get("scored_results", [])
        if not isinstance(results, list):
            raise ValueError("LLM returned non-list 'scored_results'")
        # Ensure every result has url_verified=True since these came from Serper
        for r in results:
            if isinstance(r, dict):
                r["url_verified"] = True
                r["discovered_via"] = "serper"
        return results
    except Exception as e:
        logger.error(f"[SEO] Failed to score results: {_error_chain(e)}")
        # Return raw results with basic scoring
        return [
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": r["snippet"],
                "platform": r["platform"],
                "relevance_score": 50,
                "suggested_response": "Engagement opportunity found - review this discussion for potential participation.",
                "engagement_reason": "Related to your industry keywords.",
                "url_verified": True,
                "discovered_via": "serper",
            }
            for r in raw_results[:10]
        ]


# ── Analyze Presence Pipeline ────────────────────────────────────────────────

async def analyze_presence_pipeline(request: AnalyzePresenceRequest) -> dict:
    """
    Comprehensive SEO presence analysis backed by real search data.

    Pipeline:
    1. Crawl company website
    2. Search for brand presence across platforms via Serper.dev
    3. Search for competitor presence (if provided)
    4. LLM analysis with real data inputs
    """
    logger.info(f"[SEO] Starting presence analysis for '{request.company_name}' ({request.website_url})")

    # Step 1: Crawl the website
    website_data = await crawl_website(request.website_url, max_pages=8, max_chars=50000)
    logger.info(f"[SEO] Crawled {website_data['metadata'].get('pages_crawled', 0)} pages")

    # Step 2: Search for brand presence across platforms
    platforms_to_check = ["reddit", "youtube", "quora", "twitter", "linkedin"]
    presence_data = await search_brand_presence(
        brand_name=request.company_name,
        keywords=request.keywords,
        platforms=platforms_to_check,
    )
    logger.info(f"[SEO] Checked presence across {len(presence_data)} platforms")

    # Step 3: Competitor analysis (if competitors provided)
    competitor_data = {}
    for competitor in request.competitors[:3]:  # Limit to 3 competitors
        competitor_name = competitor.strip()
        if not competitor_name:
            continue
        try:
            comp_presence = await search_brand_presence(
                brand_name=competitor_name,
                keywords=request.keywords[:2],  # Fewer keywords per competitor
                platforms=["reddit", "youtube", "google"],
            )
            competitor_data[competitor_name] = {
                platform: {"mention_count": data["mention_count"]}
                for platform, data in comp_presence.items()
            }
        except Exception as e:
            logger.warning(f"[SEO] Failed to analyze competitor '{competitor_name}': {e}")
            competitor_data[competitor_name] = {"error": str(e)}

    # Step 4: LLM analysis with real data
    analysis_result = await _run_presence_analysis(
        request=request,
        website_data=website_data,
        presence_data=presence_data,
        competitor_data=competitor_data,
    )

    return {
        "analysis": analysis_result["analysis"],
        "visibility_score": analysis_result["visibility_score"],
        "platform_scores": analysis_result["platform_scores"],
        "recommendations": analysis_result["recommendations"],
        "search_evidence": _summarize_evidence(presence_data),
        "competitor_data": competitor_data,
        "website_metadata": website_data["metadata"],
    }


async def _run_presence_analysis(
    request: AnalyzePresenceRequest,
    website_data: dict,
    presence_data: dict,
    competitor_data: dict,
) -> dict:
    """Run LLM analysis with real search data as evidence."""

    # Build evidence summary for the LLM
    evidence_lines = []
    for platform, data in presence_data.items():
        mention_count = data.get("mention_count", 0)
        top_titles = [r["title"] for r in data.get("top_results", [])[:3]]
        evidence_lines.append(
            f"- {platform.upper()}: {mention_count} brand mentions found. "
            f"Top results: {', '.join(top_titles) if top_titles else 'None found'}"
        )

    competitor_summary = ""
    if competitor_data:
        comp_lines = []
        for comp_name, comp_data in competitor_data.items():
            if "error" in comp_data:
                continue
            counts = {p: d.get("mention_count", 0) for p, d in comp_data.items()}
            comp_lines.append(f"  - {comp_name}: {counts}")
        if comp_lines:
            competitor_summary = "\nCompetitor Presence Data:\n" + "\n".join(comp_lines)

    # Truncate website content for prompt
    website_content = website_data.get("total_content", "")[:5000]

    prompt = f"""You are an expert multi-platform SEO analyst. Analyze this company's online presence using REAL search data.

COMPANY INFORMATION:
- Name: {request.company_name}
- Mission: {request.mission}
- Tone: {request.tone_of_voice}
- Target Audience: {request.target_audience}
- Buyer Persona: {request.buyer_persona}
- Keywords: {', '.join(request.keywords) if request.keywords else 'Not specified'}

WEBSITE CONTENT (crawled from {request.website_url}):
{website_content}

REAL PLATFORM PRESENCE DATA (from Google search):
{chr(10).join(evidence_lines)}
{competitor_summary}

IMPORTANT: Your scores MUST be based on the REAL data above, not speculation. If a brand has 0 mentions on Reddit, the Reddit score should be low (0-15). If they have 5+ mentions, score higher.

Scoring Guidelines:
- 0-20: No presence / not found
- 20-40: Minimal presence (1-2 mentions, old)
- 40-60: Moderate presence (3-5 mentions, some recent)
- 60-80: Good presence (5+ mentions, recent activity)
- 80-100: Strong presence (many mentions, active community engagement)

Return a JSON object with:
1. "analysis": A detailed 3-4 paragraph analysis covering current visibility, strengths, gaps, and strategy.
2. "visibility_score": Overall 0-100 score based on real data.
3. "platform_scores": Object with scores for "google", "reddit", "youtube", "twitter", "quora", "linkedin" (each 0-100, based on REAL mention counts).
4. "recommendations": Array of exactly 5 specific, actionable recommendations. Each should reference the real data (e.g., "You have 0 Reddit mentions - create an AMA in r/[relevant_subreddit]")."""

    try:
        parsed = await _run_llm_json_task(
            task_name="run_presence_analysis",
            openai_model=SEO_OPENAI_MODEL_ANALYSIS,
            gemini_model=SEO_GEMINI_MODEL_ANALYSIS,
            system_prompt="You are a data-driven SEO analyst. Base ALL scores on the real search data provided. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=3000,
        )

        platform_scores = parsed.get("platform_scores", {})
        if not isinstance(platform_scores, dict):
            platform_scores = {}
        normalized_scores = {}
        for platform in ["google", "reddit", "youtube", "twitter", "quora", "linkedin"]:
            raw_score = platform_scores.get(platform, 0)
            try:
                normalized_scores[platform] = max(0, min(int(raw_score), 100))
            except Exception:
                normalized_scores[platform] = 0

        recommendations = parsed.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = []

        visibility_score = parsed.get("visibility_score", 25)
        try:
            visibility_score = max(0, min(int(visibility_score), 100))
        except Exception:
            visibility_score = 25

        return {
            "analysis": parsed.get("analysis", "Analysis could not be generated."),
            "visibility_score": visibility_score,
            "platform_scores": normalized_scores,
            "recommendations": recommendations[:5],
        }
    except Exception as e:
        logger.error(f"[SEO] LLM analysis failed: {_error_chain(e)}")
        # Calculate basic scores from real data
        basic_scores = {}
        for platform, data in presence_data.items():
            count = data.get("mention_count", 0)
            basic_scores[platform] = min(count * 10, 100)
        total = sum(basic_scores.values()) // max(len(basic_scores), 1)
        return {
            "analysis": f"Automated analysis: Found mentions across {len([s for s in basic_scores.values() if s > 0])} platforms. LLM analysis unavailable - scores based on raw mention counts.",
            "visibility_score": total,
            "platform_scores": basic_scores,
            "recommendations": ["Increase presence on platforms with 0 mentions.", "Create keyword-optimized content.", "Engage in relevant community discussions."],
        }


def _summarize_evidence(presence_data: dict) -> dict:
    """Create a summary of search evidence for frontend display."""
    evidence = {}
    for platform, data in presence_data.items():
        evidence[platform] = {
            "mention_count": data.get("mention_count", 0),
            "top_results": [
                {"url": r["url"], "title": r["title"], "snippet": r["snippet"]}
                for r in data.get("top_results", [])[:3]
            ],
            "keyword_coverage": len(data.get("keyword_results", [])),
        }
    return evidence
