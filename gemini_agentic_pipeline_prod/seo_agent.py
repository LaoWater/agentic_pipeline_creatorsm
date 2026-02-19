# seo_agent.py
"""
SEO Intelligence Agent - Real engagement discovery and presence analysis.
Uses Serper.dev for real Google search results and OpenAI for analysis.
"""

import json
import logging
import os
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from seo_search import (
    crawl_website,
    find_engagement_opportunities,
    search_brand_presence,
    search_platform,
)

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY_VALUE", ""))

SEO_LLM_MODEL = "gpt-5.2"


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
        response = await openai_client.chat.completions.create(
            model=SEO_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You generate precise Google search queries. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed.get("queries", [])
    except Exception as e:
        logger.error(f"[SEO] Failed to generate search queries: {e}")
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
        response = await openai_client.chat.completions.create(
            model=SEO_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an SEO engagement expert. Score search results and craft authentic responses. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        results = parsed.get("scored_results", [])
        # Ensure every result has url_verified=True since these came from Serper
        for r in results:
            r["url_verified"] = True
            r["discovered_via"] = "serper"
        return results
    except Exception as e:
        logger.error(f"[SEO] Failed to score results: {e}")
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
        response = await openai_client.chat.completions.create(
            model=SEO_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data-driven SEO analyst. Base ALL scores on the real search data provided. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return {
            "analysis": parsed.get("analysis", "Analysis could not be generated."),
            "visibility_score": parsed.get("visibility_score", 25),
            "platform_scores": parsed.get("platform_scores", {}),
            "recommendations": parsed.get("recommendations", []),
        }
    except Exception as e:
        logger.error(f"[SEO] LLM analysis failed: {e}")
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
