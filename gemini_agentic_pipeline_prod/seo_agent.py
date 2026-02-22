# seo_agent.py
"""
SEO Intelligence Agent - Real engagement discovery and presence analysis.
Uses Serper.dev for real Google search results with resilient LLM analysis
(OpenAI primary, Gemini fallback in auto mode).
"""

import asyncio
import json
import logging
import os
import re

import httpx

from openai import AsyncOpenAI
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

from seo_search import (
    crawl_website,
    find_engagement_opportunities,
    search_brand_presence,
    search_brand_presence_enhanced,
    verify_search_results,
)
from seo_reddit import find_reddit_engagement
from seo_youtube import find_youtube_engagement
from seo_website_audit import audit_website
from seo_ai_visibility import check_ai_visibility
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_reasoning_effort(name: str, default: str) -> str:
    valid = {"none", "low", "medium", "high", "xhigh"}
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in valid:
        return normalized
    logger.warning(f"[SEO] Invalid reasoning effort for {name}='{value}', using default={default}")
    return default


SEO_OPENAI_REASONING_EFFORT_ANALYSIS = _env_reasoning_effort(
    "SEO_OPENAI_REASONING_EFFORT_ANALYSIS",
    "medium",
)
SEO_USE_OPENAI_FOR_FULL_ANALYSIS = _env_bool("SEO_USE_OPENAI_FOR_FULL_ANALYSIS", True)

# Startup diagnostic — helps triage Cloud Run LLM routing issues
logger.info(
    f"[SEO] Config: provider={SEO_LLM_PROVIDER}, "
    f"openai_model={SEO_OPENAI_MODEL}, gemini_model={SEO_GEMINI_MODEL}, "
    f"openai_task_models=(q:{SEO_OPENAI_MODEL_QUERIES}, s:{SEO_OPENAI_MODEL_SCORING}, a:{SEO_OPENAI_MODEL_ANALYSIS}), "
    f"gemini_task_models=(q:{SEO_GEMINI_MODEL_QUERIES}, s:{SEO_GEMINI_MODEL_SCORING}, a:{SEO_GEMINI_MODEL_ANALYSIS}), "
    f"openai_reasoning_effort_analysis={SEO_OPENAI_REASONING_EFFORT_ANALYSIS}, "
    f"use_openai_for_full_analysis={SEO_USE_OPENAI_FOR_FULL_ANALYSIS}, "
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


def _resolve_provider_order(custom_order: list[str] | None) -> list[str]:
    if not custom_order:
        return _provider_order()
    normalized: list[str] = []
    for provider in custom_order:
        p = (provider or "").strip().lower()
        if p in {"openai", "gemini"} and p not in normalized:
            normalized.append(p)
    if not normalized:
        return _provider_order()
    return normalized


def _error_chain(exc: Exception, max_depth: int = 3) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    current = exc.__cause__ or exc.__context__
    depth = 0
    while current is not None and depth < max_depth:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
        depth += 1
    return " | caused by: ".join(parts)


def _openai_model_supports_reasoning(model: str) -> bool:
    model_lower = (model or "").strip().lower()
    return model_lower.startswith("gpt-5")


def _is_reasoning_parameter_error(exc: Exception) -> bool:
    message = str(exc).lower()
    reasoning_tokens = ("reasoning", "reasoning_effort")
    parameter_tokens = ("unknown", "unsupported", "unexpected", "invalid", "extra inputs")
    return any(t in message for t in reasoning_tokens) and any(t in message for t in parameter_tokens)


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
    reasoning_effort: str = "none",
) -> dict:
    if not os.environ.get("OPENAI_API_KEY_VALUE", "").strip():
        raise ValueError("OPENAI_API_KEY_VALUE is missing or empty")
    base_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    client = _get_openai_client()
    normalized_effort = (reasoning_effort or "none").strip().lower()
    if normalized_effort not in {"none", "low", "medium", "high", "xhigh"}:
        logger.warning(
            f"[SEO] Invalid reasoning_effort='{reasoning_effort}' for {task_name}, falling back to 'none'"
        )
        normalized_effort = "none"

    attempt_kwargs: list[dict] = [{}]
    if normalized_effort != "none":
        if _openai_model_supports_reasoning(model):
            attempt_kwargs = [
                {"reasoning": {"effort": normalized_effort}},
                {"reasoning_effort": normalized_effort},
                {},
            ]
        else:
            logger.info(
                f"[SEO] Skipping reasoning effort for {task_name}; model={model} may not support it"
            )

    response = None
    last_reasoning_exc: Exception | None = None
    for extra_kwargs in attempt_kwargs:
        try:
            response = await client.chat.completions.create(**base_kwargs, **extra_kwargs)
            break
        except TypeError as exc:
            if extra_kwargs:
                last_reasoning_exc = exc
                continue
            raise
        except Exception as exc:
            if extra_kwargs and _is_reasoning_parameter_error(exc):
                last_reasoning_exc = exc
                continue
            raise

    if response is None:
        if last_reasoning_exc is not None:
            raise last_reasoning_exc
        raise RuntimeError("OpenAI call failed without a concrete exception")

    request_id = getattr(response, "_request_id", None)
    if request_id:
        logger.info(
            f"[SEO] {task_name} OpenAI request_id={request_id} model={model} reasoning_effort={normalized_effort}"
        )
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
    openai_reasoning_effort: str = "none",
    provider_order: list[str] | None = None,
) -> dict:
    resolved_openai_model = openai_model or SEO_OPENAI_MODEL
    resolved_gemini_model = gemini_model or SEO_GEMINI_MODEL
    resolved_provider_order = _resolve_provider_order(provider_order)
    errors: list[str] = []
    for provider in resolved_provider_order:
        try:
            if provider == "openai":
                parsed = await _call_openai_json(
                    task_name=task_name,
                    model=resolved_openai_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=openai_reasoning_effort,
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

    Multi-agent pipeline:
    1. LLM generates search keywords (not site:-prefixed queries)
    2. Three agents run in parallel:
       - Reddit Agent  (free native API)
       - YouTube Agent (free Data API v3)
       - Serper Agent  (Google Search — scoped to Quora + forums only)
    3. Results merged, deduplicated, scored by LLM
    4. URL verification via async HEAD requests
    """
    logger.info(f"[SEO] Starting engagement search for '{request.company_name}'")

    # Step 1: Generate search keywords (shared across all agents)
    keywords = _extract_keywords(request)
    logger.info(f"[SEO] Keywords for multi-agent search: {keywords}")

    # Also generate Serper-specific queries for Quora + forums
    serper_queries = await _generate_serper_queries(request)
    logger.info(f"[SEO] Generated {len(serper_queries)} Serper queries (Quora/forums)")

    # Step 2: Run all three agents in parallel
    reddit_task = find_reddit_engagement(
        keywords=keywords,
        num_per_keyword=5,
        time_filter="week",
    )
    youtube_task = find_youtube_engagement(
        keywords=keywords,
        num_per_keyword=3,
        time_filter="week",
        include_comments=True,
    )
    serper_task = _serper_with_fallback(serper_queries)

    reddit_results, youtube_results, serper_results = await asyncio.gather(
        reddit_task, youtube_task, serper_task,
        return_exceptions=False,
    )

    # Handle exceptions from individual agents gracefully
    if isinstance(reddit_results, Exception):
        logger.error(f"[SEO] Reddit agent failed: {reddit_results}")
        reddit_results = []
    if isinstance(youtube_results, Exception):
        logger.error(f"[SEO] YouTube agent failed: {youtube_results}")
        youtube_results = []
    if isinstance(serper_results, Exception):
        logger.error(f"[SEO] Serper agent failed: {serper_results}")
        serper_results = []

    # Tag discovered_via on raw results
    for r in reddit_results:
        r["discovered_via"] = "reddit_api"
    for r in youtube_results:
        r["discovered_via"] = "youtube_api"
    for r in serper_results:
        r["discovered_via"] = "serper"

    # Step 3: Merge and deduplicate
    raw_results = _merge_and_deduplicate(reddit_results, youtube_results, serper_results)
    logger.info(
        f"[SEO] Multi-agent results: "
        f"Reddit={len(reddit_results)}, YouTube={len(youtube_results)}, "
        f"Serper={len(serper_results)} → {len(raw_results)} unique"
    )

    if not raw_results:
        return {
            "opportunities": [],
            "queries_used": keywords + serper_queries,
            "message": "No engagement opportunities found. Try different keywords.",
        }

    # Step 4: LLM scores and generates responses
    opportunities = await _score_and_respond(raw_results, request)
    logger.info(f"[SEO] Scored and generated responses for {len(opportunities)} opportunities")

    # Step 5: Verify URLs are actually alive (async HEAD requests)
    opportunities = await _verify_urls(opportunities)

    return {
        "opportunities": opportunities,
        "queries_used": keywords + serper_queries,
        "total_raw_results": len(raw_results),
        "sources": {
            "reddit": len(reddit_results),
            "youtube": len(youtube_results),
            "serper": len(serper_results),
        },
        "message": f"Found {len(opportunities)} engagement opportunities.",
    }


def _extract_keywords(request: SearchEngagementRequest) -> list[str]:
    """Extract clean search keywords from the request (no site: operators)."""
    keywords = list(request.keywords) if request.keywords else []

    # Add company name as a keyword
    if request.company_name and request.company_name not in keywords:
        keywords.append(request.company_name)

    # If we have very few keywords, generate some from the audience/industry
    if len(keywords) < 3:
        if request.industry:
            keywords.append(request.industry)
        if request.target_audience:
            # Extract key phrases (first 3 words of target audience)
            words = request.target_audience.split()[:5]
            if len(words) >= 2:
                keywords.append(" ".join(words))

    # Deduplicate while preserving order
    seen = set()
    clean = []
    for kw in keywords:
        kw_lower = kw.strip().lower()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            clean.append(kw.strip())

    return clean[:8]  # Cap at 8 keywords


def _merge_and_deduplicate(*result_lists: list[dict]) -> list[dict]:
    """Merge multiple result lists, deduplicate by URL."""
    seen_urls: set[str] = set()
    merged: list[dict] = []
    for results in result_lists:
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(r)
    return merged


async def _serper_with_fallback(queries: list[str]) -> list[dict]:
    """Run Serper searches with week → month → year fallback."""
    results = await find_engagement_opportunities(
        queries=queries,
        num_per_query=5,
        time_range="w",
    )
    if not results:
        results = await find_engagement_opportunities(
            queries=queries,
            num_per_query=5,
            time_range="m",
        )
    if not results:
        results = await find_engagement_opportunities(
            queries=queries,
            num_per_query=8,
            time_range="y",
        )
    return results


async def _generate_serper_queries(request: SearchEngagementRequest) -> list[str]:
    """
    Generate Google search queries scoped to Quora + forums only.
    Reddit and YouTube are handled by their native APIs.
    """
    keywords_str = ", ".join(request.keywords) if request.keywords else "general industry topics"

    prompt = f"""You are an SEO expert. Generate 4-6 targeted Google search queries to find discussions on Quora, forums, and blogs.

Company: {request.company_name}
Industry: {request.industry}
Keywords: {keywords_str}
Target Audience: {request.target_audience}
Website: {request.website_url}

IMPORTANT: Do NOT generate queries for Reddit or YouTube (those are handled separately via native APIs).

Generate queries that will find:
1. Quora questions about the industry/topics (2 queries with site:quora.com)
2. General forum discussions, blog comments, and community posts (2-4 queries — use keywords naturally, optionally with "forum", "discussion", "community" terms)

Rules:
- Use natural language queries people would actually search
- Include relevant keywords naturally
- Use quotes around multi-word phrases when needed
- Focus on questions, recommendations, comparisons, and help-seeking posts

Return a JSON object with a single key "queries" containing an array of search query strings."""

    try:
        parsed = await _run_llm_json_task(
            task_name="generate_serper_queries",
            openai_model=SEO_OPENAI_MODEL_QUERIES,
            gemini_model=SEO_GEMINI_MODEL_QUERIES,
            system_prompt="You generate precise Google search queries. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=600,
        )
        queries = parsed.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("LLM returned non-list 'queries'")
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not cleaned:
            raise ValueError("LLM returned empty 'queries'")
        return cleaned[:6]
    except Exception as e:
        logger.error(f"[SEO] Failed to generate Serper queries: {_error_chain(e)}")
        fallback = [
            f'site:quora.com {keywords_str} recommendation',
            f'{keywords_str} discussion forum',
            f'{keywords_str} community advice',
        ]
        return fallback


async def _verify_urls(results: list[dict]) -> list[dict]:
    """HEAD-request each URL; set url_verified based on HTTP status < 400."""
    if not results:
        return results

    async with httpx.AsyncClient(
        timeout=5.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; BrandVerseBot/1.0)"},
    ) as client:
        tasks = [client.head(r["url"]) for r in results if r.get("url")]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    idx = 0
    for r in results:
        if not r.get("url"):
            r["url_verified"] = False
            continue
        resp = responses[idx]
        idx += 1
        if isinstance(resp, Exception):
            r["url_verified"] = False
            logger.debug(f"[SEO] URL check failed for {r['url']}: {resp}")
        else:
            r["url_verified"] = resp.status_code < 400

    # Sort verified URLs to the top, preserve relative order otherwise
    results.sort(key=lambda r: (not r.get("url_verified", False)))
    verified_count = sum(1 for r in results if r.get("url_verified"))
    logger.info(f"[SEO] URL verification: {verified_count}/{len(results)} alive")
    return results


async def _score_and_respond(
    raw_results: list[dict],
    request: SearchEngagementRequest,
) -> list[dict]:
    """Use LLM to score relevance and generate suggested responses for real results."""

    # Prepare results for LLM (limit to top 20 to keep prompt manageable)
    results_for_llm = []
    for r in raw_results[:20]:
        entry = {
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "platform": r["platform"],
        }
        # Include rich metadata from native APIs (helps LLM score better)
        if r.get("upvotes"):
            entry["upvotes"] = r["upvotes"]
        if r.get("num_comments"):
            entry["num_comments"] = r["num_comments"]
        if r.get("subreddit"):
            entry["subreddit"] = r["subreddit"]
        if r.get("channel_title"):
            entry["channel_title"] = r["channel_title"]
        if r.get("date"):
            entry["date"] = r["date"]
        results_for_llm.append(entry)

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
        # Carry over discovered_via from raw results (set by orchestrator)
        raw_by_url = {r["url"]: r for r in raw_results}
        for r in results:
            if isinstance(r, dict) and not r.get("discovered_via"):
                raw = raw_by_url.get(r.get("url", ""), {})
                r["discovered_via"] = raw.get("discovered_via", "serper")
        return results
    except Exception as e:
        logger.error(f"[SEO] Failed to score results: {_error_chain(e)}")
        # Return raw results with basic scoring; url_verified set by _verify_urls later
        return [
            {
                "url": r["url"],
                "title": r["title"],
                "snippet": r["snippet"],
                "platform": r["platform"],
                "relevance_score": 50,
                "suggested_response": "Engagement opportunity found - review this discussion for potential participation.",
                "engagement_reason": "Related to your industry keywords.",
                "discovered_via": r.get("discovered_via", "serper"),
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
            openai_reasoning_effort=SEO_OPENAI_REASONING_EFFORT_ANALYSIS,
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
            "verified_mention_count": data.get("verified_mention_count", data.get("mention_count", 0)),
            "top_results": [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r["snippet"],
                    "verified_relevant": r.get("verified_relevant", True),
                    "trustworthiness_score": r.get("trustworthiness_score"),
                    "visibility_impact": r.get("visibility_impact"),
                }
                for r in data.get("top_results", [])[:5]
            ],
            "keyword_coverage": len(data.get("keyword_results", [])),
        }
    return evidence


# ── Manage Keywords Pipeline ─────────────────────────────────────────────────

class ManageKeywordsRequest(BaseModel):
    company_name: str = Field(..., description="Company name")
    company_description: str = Field(default="", description="What the company does")
    industry: str = Field(default="", description="Industry")
    target_audience: str = Field(default="", description="Target audience")
    existing_keywords: list[str] = Field(default_factory=list, description="Already tracked keywords")
    action: str = Field(default="suggest", description="'suggest' or 'analyze'")


async def manage_keywords_pipeline(request: ManageKeywordsRequest) -> dict:
    """
    Keyword management via LLM.
    - action='suggest': Generate new keyword suggestions
    - action='analyze': Rate existing keywords for SEO value
    """
    if request.action == "analyze" and request.existing_keywords:
        return await _analyze_keywords(request)
    return await _suggest_keywords(request)


async def _suggest_keywords(request: ManageKeywordsRequest) -> dict:
    """LLM suggests keywords based on company info."""
    existing_str = ", ".join(request.existing_keywords) if request.existing_keywords else "None yet"

    prompt = f"""You are an SEO keyword strategist. Suggest 10-15 keywords for this company.

Company: {request.company_name}
Description: {request.company_description}
Industry: {request.industry}
Target Audience: {request.target_audience}
Already Tracking: {existing_str}

For each keyword, categorize as:
- "primary": Core business terms (high volume, high competition)
- "secondary": Related terms (medium volume)
- "long-tail": Specific phrases (lower volume, higher intent)
- "brand": Brand-related terms
- "competitor": Competitor-related terms

Return JSON with key "keywords" containing array of objects:
[{{"keyword": "...", "category": "primary|secondary|long-tail|brand|competitor", "reasoning": "brief reason", "estimated_difficulty": "easy|medium|hard"}}]

Do NOT suggest keywords already being tracked."""

    try:
        parsed = await _run_llm_json_task(
            task_name="suggest_keywords",
            openai_model=SEO_OPENAI_MODEL_QUERIES,
            gemini_model=SEO_GEMINI_MODEL_QUERIES,
            system_prompt="You are an SEO keyword expert. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=1500,
        )
        suggestions = parsed.get("keywords", [])
        return {"keywords": suggestions, "action": "suggest"}
    except Exception as e:
        logger.error(f"[SEO] Keyword suggestion failed: {e}")
        return {"keywords": [], "action": "suggest", "error": str(e)}


async def _analyze_keywords(request: ManageKeywordsRequest) -> dict:
    """LLM rates existing keywords for SEO value."""
    keywords_str = ", ".join(request.existing_keywords)

    prompt = f"""Rate these keywords for SEO value for this company:

Company: {request.company_name}
Description: {request.company_description}
Industry: {request.industry}
Keywords: {keywords_str}

Return JSON with key "analysis" containing array matching keyword order:
[{{"keyword": "...", "seo_value_score": 0-100, "search_volume_estimate": "low|medium|high|very_high", "difficulty_estimate": "easy|medium|hard", "recommendation": "keep|optimize|replace|expand"}}]"""

    try:
        parsed = await _run_llm_json_task(
            task_name="analyze_keywords",
            openai_model=SEO_OPENAI_MODEL_QUERIES,
            gemini_model=SEO_GEMINI_MODEL_QUERIES,
            system_prompt="You are an SEO keyword analyst. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=1200,
        )
        analysis = parsed.get("analysis", [])
        return {"analysis": analysis, "action": "analyze"}
    except Exception as e:
        logger.error(f"[SEO] Keyword analysis failed: {e}")
        return {"analysis": [], "action": "analyze", "error": str(e)}


# ── 4-Pillar Analysis Pipeline (V2) ─────────────────────────────────────────

async def analyze_presence_pipeline_v2(request: AnalyzePresenceRequest) -> dict:
    """
    4-Pillar SEO presence analysis.

    Pipeline:
    Step 0: Crawl website (needed by multiple pillars)
    Step 1: Run all 4 pillars in parallel
    Step 2: Competitor analysis
    Step 3: Art of Marketing synthesis
    """
    logger.info(f"[SEO v2] Starting 4-pillar analysis for '{request.company_name}' ({request.website_url})")

    # Step 0: Crawl website
    website_data = await crawl_website(request.website_url, max_pages=8, max_chars=50000)
    logger.info(f"[SEO v2] Crawled {website_data['metadata'].get('pages_crawled', 0)} pages")

    company_desc = request.mission or website_data.get("metadata", {}).get("description", "")

    # Helper: LLM caller adapter for verify_search_results
    async def _llm_caller(task_name, system_prompt, user_prompt, temperature=0.3, max_tokens=800):
        return await _run_llm_json_task(
            task_name=task_name,
            openai_model=SEO_OPENAI_MODEL_QUERIES,
            gemini_model=SEO_GEMINI_MODEL_QUERIES,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Step 1: Run all 4 pillars in parallel
    async def _pillar_google_search():
        try:
            platforms_to_check = ["reddit", "youtube", "quora", "twitter", "linkedin"]
            return await search_brand_presence_enhanced(
                brand_name=request.company_name,
                company_description=company_desc,
                keywords=request.keywords,
                platforms=platforms_to_check,
                llm_caller=_llm_caller,
            )
        except Exception as e:
            logger.error(f"[SEO v2] Pillar 1 (Google Search) failed: {e}")
            return {}

    async def _pillar_ai_visibility():
        try:
            return await check_ai_visibility(
                company_name=request.company_name,
                company_description=company_desc,
                website_url=request.website_url,
                keywords=request.keywords,
            )
        except Exception as e:
            logger.error(f"[SEO v2] Pillar 2 (AI Visibility) failed: {e}")
            return {"overall_score": 0, "systems": {}, "key_findings": [], "recommendations": []}

    async def _pillar_community():
        """Existing presence data, elevated as pillar."""
        try:
            platforms_to_check = ["reddit", "youtube", "quora", "twitter", "linkedin"]
            return await search_brand_presence(
                brand_name=request.company_name,
                keywords=request.keywords,
                platforms=platforms_to_check,
            )
        except Exception as e:
            logger.error(f"[SEO v2] Pillar 3 (Community) failed: {e}")
            return {}

    async def _pillar_website_technical():
        try:
            return await audit_website(request.website_url)
        except Exception as e:
            logger.error(f"[SEO v2] Pillar 4 (Website Technical) failed: {e}")
            return {"crawlability_score": 0, "issues": [], "inferred_tech_stack": []}

    google_data, ai_data, community_data, audit_data = await asyncio.gather(
        _pillar_google_search(),
        _pillar_ai_visibility(),
        _pillar_community(),
        _pillar_website_technical(),
    )

    logger.info("[SEO v2] All 4 pillars completed")

    # Step 2: Competitor analysis
    competitor_data = {}
    for competitor in request.competitors[:3]:
        competitor_name = competitor.strip()
        if not competitor_name:
            continue
        try:
            comp_presence = await search_brand_presence(
                brand_name=competitor_name,
                keywords=request.keywords[:2],
                platforms=["reddit", "youtube", "google"],
            )
            competitor_data[competitor_name] = {
                platform: {"mention_count": data["mention_count"]}
                for platform, data in comp_presence.items()
            }
        except Exception as e:
            logger.warning(f"[SEO v2] Competitor '{competitor_name}' failed: {e}")
            competitor_data[competitor_name] = {"error": str(e)}

    # Calculate pillar scores
    # Pillar 1: Google Search — based on verified mentions
    google_score = _calculate_google_pillar_score(google_data)
    # Pillar 2: AI Visibility — directly from check_ai_visibility
    ai_score = ai_data.get("overall_score", 0)
    # Pillar 3: Community — based on mention counts across platforms
    community_score = _calculate_community_pillar_score(community_data)
    # Pillar 4: Website Technical — crawlability score
    technical_score = audit_data.get("crawlability_score", 0)

    pillar_scores = {
        "google_search": google_score,
        "ai_visibility": ai_score,
        "community": community_score,
        "website_technical": technical_score,
    }

    # Overall visibility score: weighted average
    overall = int(
        google_score * 0.30
        + ai_score * 0.20
        + community_score * 0.25
        + technical_score * 0.25
    )

    # Step 3: Art of Marketing synthesis
    synthesis = await _synthesize_4_pillar_analysis(
        request=request,
        website_data=website_data,
        pillar_scores=pillar_scores,
        google_data=google_data,
        ai_data=ai_data,
        community_data=community_data,
        audit_data=audit_data,
        competitor_data=competitor_data,
    )

    # Build platform scores from google_data for backward compat
    platform_scores = {}
    for platform in ["google", "reddit", "youtube", "twitter", "quora", "linkedin"]:
        p_data = google_data.get(platform, {})
        count = p_data.get("verified_mention_count", p_data.get("mention_count", 0))
        platform_scores[platform] = min(count * 10, 100)

    # Pillar details for frontend
    pillar_details = {
        "google_search": {
            "evidence": _summarize_evidence(google_data),
            "total_mentions": sum(d.get("mention_count", 0) for d in google_data.values()),
            "verified_mentions": sum(d.get("verified_mention_count", 0) for d in google_data.values()),
        },
        "ai_visibility": ai_data,
        "community": _summarize_evidence(community_data),
        "website_technical": {
            "crawlability_score": audit_data.get("crawlability_score", 0),
            "issues": audit_data.get("issues", []),
            "tech_stack": audit_data.get("inferred_tech_stack", []),
            "js_dependent_content": audit_data.get("js_dependent_content", False),
            "has_meta_title": audit_data.get("has_meta_title", False),
            "has_meta_description": audit_data.get("has_meta_description", False),
            "has_og_tags": audit_data.get("has_og_tags", False),
            "has_sitemap": audit_data.get("has_sitemap", False),
            "has_robots_txt": audit_data.get("has_robots_txt", False),
            "has_canonical_url": audit_data.get("has_canonical_url", False),
            "response_time_ms": audit_data.get("response_time_ms", 0),
            "technical_summary": audit_data.get("technical_summary", ""),
        },
    }

    return {
        "analysis": synthesis.get("analysis", ""),
        "visibility_score": overall,
        "platform_scores": platform_scores,
        "recommendations": synthesis.get("recommendations", []),
        "pillar_scores": pillar_scores,
        "pillar_details": pillar_details,
        "marketing_plan": synthesis.get("marketing_plan", {}),
        "search_evidence": _summarize_evidence(google_data),
        "competitor_data": competitor_data,
        "website_metadata": website_data["metadata"],
    }


def _calculate_google_pillar_score(google_data: dict) -> int:
    """Calculate Google Search pillar score from verified mentions."""
    if not google_data:
        return 0
    total_verified = sum(d.get("verified_mention_count", d.get("mention_count", 0)) for d in google_data.values())
    total_mentions = sum(d.get("mention_count", 0) for d in google_data.values())
    # Score based on total presence (cap at 100)
    base_score = min(total_mentions * 3, 80)
    # Bonus for verified results
    if total_mentions > 0:
        verification_ratio = total_verified / total_mentions
        base_score += int(verification_ratio * 20)
    return min(base_score, 100)


def _calculate_community_pillar_score(community_data: dict) -> int:
    """Calculate Community pillar score from mention counts."""
    if not community_data:
        return 0
    # Community platforms (exclude google)
    community_platforms = {k: v for k, v in community_data.items() if k != "google"}
    if not community_platforms:
        return 0
    total = sum(d.get("mention_count", 0) for d in community_platforms.values())
    platforms_present = sum(1 for d in community_platforms.values() if d.get("mention_count", 0) > 0)
    # Score: mentions contribute + platform breadth bonus
    base = min(total * 5, 70)
    breadth_bonus = platforms_present * 6  # Up to 30 for 5 platforms
    return min(base + breadth_bonus, 100)


async def _synthesize_4_pillar_analysis(
    request: AnalyzePresenceRequest,
    website_data: dict,
    pillar_scores: dict,
    google_data: dict,
    ai_data: dict,
    community_data: dict,
    audit_data: dict,
    competitor_data: dict,
) -> dict:
    """
    Synthesize all 4 pillars into a unified analysis using the Art of Marketing mental model.
    """
    # Build evidence summaries
    google_summary = []
    for platform, data in google_data.items():
        count = data.get("mention_count", 0)
        verified = data.get("verified_mention_count", count)
        google_summary.append(f"  - {platform}: {count} mentions ({verified} verified)")

    ai_summary = f"  - ChatGPT awareness: {ai_data.get('overall_score', 0)}/100 ({ai_data.get('systems', {}).get('chatgpt', {}).get('awareness', 'unknown')})"

    community_summary = []
    for platform, data in community_data.items():
        if platform == "google":
            continue
        count = data.get("mention_count", 0)
        community_summary.append(f"  - {platform}: {count} mentions")

    issues = audit_data.get("issues", [])
    critical_issues = [i for i in issues if i.get("severity") == "critical"]
    warning_issues = [i for i in issues if i.get("severity") == "warning"]
    audit_summary = (
        f"  - Crawlability: {audit_data.get('crawlability_score', 0)}/100\n"
        f"  - Critical issues: {len(critical_issues)}, Warnings: {len(warning_issues)}\n"
        f"  - JS-dependent: {audit_data.get('js_dependent_content', False)}\n"
        f"  - Tech: {', '.join(t['name'] for t in audit_data.get('inferred_tech_stack', []))}"
    )

    competitor_summary = ""
    if competitor_data:
        comp_lines = []
        for name, data in competitor_data.items():
            if "error" in data:
                continue
            counts = {p: d.get("mention_count", 0) for p, d in data.items()}
            comp_lines.append(f"  - {name}: {counts}")
        if comp_lines:
            competitor_summary = "\nCompetitor Data:\n" + "\n".join(comp_lines)

    website_content = website_data.get("total_content", "")[:3000]

    prompt = f"""You are a senior marketing strategist and SEO expert. Synthesize a comprehensive analysis using the "Art of Marketing" framework.

═══ COMPANY ═══
Name: {request.company_name}
Mission: {request.mission}
Target Audience: {request.target_audience}
Keywords: {', '.join(request.keywords) if request.keywords else 'Not specified'}

═══ 4-PILLAR SCORES ═══
1. Google Search Presence: {pillar_scores['google_search']}/100
2. AI Visibility: {pillar_scores['ai_visibility']}/100
3. Community Presence: {pillar_scores['community']}/100
4. Website Technical: {pillar_scores['website_technical']}/100

═══ PILLAR 1: GOOGLE SEARCH DATA ═══
{chr(10).join(google_summary)}

═══ PILLAR 2: AI VISIBILITY ═══
{ai_summary}
Key findings: {json.dumps(ai_data.get('key_findings', []))}

═══ PILLAR 3: COMMUNITY PRESENCE ═══
{chr(10).join(community_summary)}

═══ PILLAR 4: WEBSITE TECHNICAL ═══
{audit_summary}
{competitor_summary}

═══ WEBSITE CONTENT (excerpt) ═══
{website_content}

═══ ART OF MARKETING FRAMEWORK ═══
Apply these principles in your analysis:
1. Trust Signals: How well does the brand build trust across digital touchpoints?
2. Social Proof Loop: Are real people talking about this brand? Where?
3. Modern Decision Funnel: 73% of buying decisions happen outside Google (TikTok, YouTube, Reddit, AI assistants). How visible is this brand in the actual decision journey?
4. Crawlability Foundation: Can search engines AND AI systems actually access and understand the content?
5. Keyword Strategy: Are they targeting the right terms at the right intent level?

═══ YOUR TASK ═══
Return a JSON object with:

1. "analysis": A detailed 4-5 paragraph analysis covering all 4 pillars. Be specific — reference real data. Write for a founder/marketer, not a technical audience.

2. "recommendations": Array of exactly 5 specific, actionable recommendations. Each should reference real data.

3. "marketing_plan": Object with 4 priority tiers:
   - "quick_wins": Array of 2-3 actions that can be done this week (low effort, high impact)
   - "foundation": Array of 2-3 foundational fixes (medium effort, critical for long-term)
   - "growth": Array of 2-3 growth engines to build (higher effort, compounding returns)
   - "long_term": Array of 1-2 long-term positioning strategies

Each item in the marketing plan should be an object with: "action", "effort" ("low"|"medium"|"high"), "impact" ("low"|"medium"|"high"), "timeframe" ("1 week"|"1 month"|"3 months"|"6 months"), "details" (1-2 sentences)."""

    full_analysis_provider_order = ["openai", "gemini"] if SEO_USE_OPENAI_FOR_FULL_ANALYSIS else None

    try:
        parsed = await _run_llm_json_task(
            task_name="synthesize_4_pillar",
            openai_model=SEO_OPENAI_MODEL_ANALYSIS,
            gemini_model=SEO_GEMINI_MODEL_ANALYSIS,
            system_prompt="You are a marketing strategist and SEO expert. Provide data-driven analysis using the Art of Marketing framework. Return valid JSON only.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=4000,
            openai_reasoning_effort=SEO_OPENAI_REASONING_EFFORT_ANALYSIS,
            provider_order=full_analysis_provider_order,
        )

        return {
            "analysis": parsed.get("analysis", "Analysis could not be generated."),
            "recommendations": parsed.get("recommendations", [])[:5],
            "marketing_plan": parsed.get("marketing_plan", {}),
        }
    except Exception as e:
        logger.error(f"[SEO v2] Synthesis failed: {e}")
        return {
            "analysis": f"4-pillar analysis completed. Scores: Google Search {pillar_scores['google_search']}/100, AI Visibility {pillar_scores['ai_visibility']}/100, Community {pillar_scores['community']}/100, Website Technical {pillar_scores['website_technical']}/100. LLM synthesis unavailable.",
            "recommendations": [
                "Improve your weakest pillar first for the biggest gains.",
                "Ensure your website is crawlable by search engines and AI systems.",
                "Build community presence on platforms where your audience lives.",
            ],
            "marketing_plan": {},
        }
