# seo_search.py
"""
Serper.dev API wrapper for real Google search results.
Used by the SEO Agent to find real engagement opportunities and measure platform visibility.
Enhanced with LLM-verified search results for the 4-pillar analysis system.
"""

import json
import os
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = "https://google.serper.dev"


async def search_google(
    query: str,
    num_results: int = 10,
    search_type: str = "search",
    time_range: Optional[str] = None,
) -> dict:
    """
    Execute a Google search via Serper.dev API.

    Args:
        query: Search query string (supports site: operators)
        num_results: Number of results to return (max 100)
        search_type: "search" (web), "news", "images", "videos"
        time_range: Filter by time - "h" (hour), "d" (day), "w" (week), "m" (month), "y" (year)

    Returns:
        Dict with 'organic' results containing url, title, snippet, position, date
    """
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY environment variable is not set")

    payload = {
        "q": query,
        "num": num_results,
    }
    if time_range:
        payload["tbs"] = f"qdr:{time_range}"

    endpoint = f"{SERPER_BASE_URL}/{search_type}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            endpoint,
            json=payload,
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return response.json()


async def search_platform(
    query: str,
    platform: str,
    num_results: int = 10,
    time_range: Optional[str] = "y",
) -> list[dict]:
    """
    Search for content on a specific platform using Google site: operator.

    Args:
        query: Search terms
        platform: Platform name (reddit, youtube, quora, twitter, linkedin, facebook)
        num_results: Number of results
        time_range: Time filter

    Returns:
        List of results with url, title, snippet, platform
    """
    site_map = {
        "reddit": "site:reddit.com",
        "youtube": "site:youtube.com",
        "quora": "site:quora.com",
        "twitter": "site:twitter.com OR site:x.com",
        "linkedin": "site:linkedin.com",
        "facebook": "site:facebook.com",
    }

    site_filter = site_map.get(platform, "")
    full_query = f"{site_filter} {query}" if site_filter else query

    try:
        data = await search_google(full_query, num_results=num_results, time_range=time_range)
        results = []
        for item in data.get("organic", []):
            results.append({
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0),
                "date": item.get("date", ""),
                "platform": platform,
            })
        return results
    except Exception as e:
        logger.error(f"Error searching {platform} for '{query}': {e}")
        return []


async def search_brand_presence(
    brand_name: str,
    keywords: list[str],
    platforms: list[str] = None,
) -> dict:
    """
    Search for a brand's presence across multiple platforms.
    Returns counts and top results per platform.

    Args:
        brand_name: Company/brand name
        keywords: List of target keywords
        platforms: Platforms to check (defaults to all major ones)

    Returns:
        Dict keyed by platform with mention_count, top_results, keyword_results
    """
    if platforms is None:
        platforms = ["reddit", "youtube", "quora", "twitter", "linkedin"]

    presence = {}

    for platform in platforms:
        # Search for brand name on this platform
        brand_results = await search_platform(
            query=f'"{brand_name}"',
            platform=platform,
            num_results=10,
            time_range=None,  # All time for brand presence
        )

        # Search for top keywords on this platform
        keyword_results = []
        for keyword in keywords[:3]:  # Limit to top 3 keywords to save API calls
            kw_results = await search_platform(
                query=keyword,
                platform=platform,
                num_results=5,
                time_range="y",  # Last year
            )
            keyword_results.extend(kw_results)

        presence[platform] = {
            "mention_count": len(brand_results),
            "top_results": brand_results[:5],
            "keyword_results": keyword_results[:10],
        }

    # Also check general Google presence
    try:
        general_data = await search_google(f'"{brand_name}"', num_results=10)
        general_results = [
            {
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0),
                "date": item.get("date", ""),
                "platform": "google",
            }
            for item in general_data.get("organic", [])
        ]
        presence["google"] = {
            "mention_count": len(general_results),
            "top_results": general_results,
            "keyword_results": [],
        }
    except Exception as e:
        logger.error(f"Error checking general Google presence: {e}")
        presence["google"] = {"mention_count": 0, "top_results": [], "keyword_results": []}

    return presence


async def find_engagement_opportunities(
    queries: list[str],
    num_per_query: int = 5,
    time_range: str = "m",
) -> list[dict]:
    """
    Execute multiple search queries and collect all results.
    Used after LLM generates targeted search queries.

    Args:
        queries: List of search queries (may include site: operators)
        num_per_query: Results per query
        time_range: Time filter (default: last month)

    Returns:
        Deduplicated list of results with url, title, snippet, platform
    """
    all_results = []
    seen_urls = set()

    for query in queries:
        try:
            data = await search_google(query, num_results=num_per_query, time_range=time_range)
            for item in data.get("organic", []):
                url = item.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    # Detect platform from URL
                    platform = _detect_platform(url)
                    all_results.append({
                        "url": url,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0),
                        "date": item.get("date", ""),
                        "platform": platform,
                        "search_query": query,
                    })
        except Exception as e:
            logger.error(f"Error executing search '{query}': {e}")
            continue

    return all_results


def _detect_platform(url: str) -> str:
    """Detect platform from URL."""
    url_lower = url.lower()
    if "reddit.com" in url_lower:
        return "reddit"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    elif "quora.com" in url_lower:
        return "quora"
    elif "linkedin.com" in url_lower:
        return "linkedin"
    elif "facebook.com" in url_lower:
        return "facebook"
    elif any(f in url_lower for f in ["forum", "community", "discuss", "boards"]):
        return "forum"
    else:
        return "other"


async def crawl_website(url: str, max_pages: int = 8, max_chars: int = 50000) -> dict:
    """
    Crawl a website and extract clean text content.

    Args:
        url: Website URL to crawl
        max_pages: Maximum pages to fetch (including homepage)
        max_chars: Maximum total characters to extract

    Returns:
        Dict with pages crawled, total content, metadata
    """
    from bs4 import BeautifulSoup
    import re as stdlib_re

    if not url.startswith("http"):
        url = f"https://{url}"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BrandverseBot/2.0; SEO Analysis)"
    }

    pages = []
    total_chars = 0

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        # Fetch homepage
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            homepage_html = resp.text
        except Exception as e:
            logger.error(f"Failed to fetch homepage {url}: {e}")
            return {"pages": [], "total_content": "", "metadata": {"error": str(e)}}

        # Parse homepage
        soup = BeautifulSoup(homepage_html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")

        # Extract clean text
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        homepage_text = soup.get_text(separator=" ", strip=True)
        homepage_text = stdlib_re.sub(r"\s+", " ", homepage_text)[:8000]
        total_chars += len(homepage_text)

        pages.append({
            "url": url,
            "title": title,
            "content": homepage_text,
            "is_homepage": True,
        })

        # Find internal links
        internal_links = set()
        from urllib.parse import urljoin, urlparse
        base_domain = urlparse(url).netloc

        # Priority paths
        priority_paths = ["/about", "/services", "/products", "/pricing", "/features", "/blog", "/contact", "/team", "/solutions", "/case-studies"]

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain and full_url != url:
                internal_links.add(full_url)

        # Sort: priority paths first
        sorted_links = sorted(
            internal_links,
            key=lambda u: next((i for i, p in enumerate(priority_paths) if p in u.lower()), 99),
        )

        # Fetch subpages
        for link in sorted_links[: max_pages - 1]:
            if total_chars >= max_chars:
                break
            try:
                resp = await client.get(link, headers=headers)
                if resp.status_code != 200:
                    continue
                sub_soup = BeautifulSoup(resp.text, "html.parser")
                sub_title = sub_soup.title.string.strip() if sub_soup.title and sub_soup.title.string else ""
                for tag in sub_soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    tag.decompose()
                sub_text = sub_soup.get_text(separator=" ", strip=True)
                sub_text = stdlib_re.sub(r"\s+", " ", sub_text)[:3000]
                total_chars += len(sub_text)

                pages.append({
                    "url": link,
                    "title": sub_title,
                    "content": sub_text,
                    "is_homepage": False,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch subpage {link}: {e}")
                continue

    # Combine all content
    combined = ""
    for page in pages:
        label = "HOMEPAGE" if page["is_homepage"] else urlparse(page["url"]).path.upper()
        combined += f"\n=== {label} ===\nTitle: {page['title']}\n{page['content']}\n\n"

    return {
        "pages": pages,
        "total_content": combined[:max_chars],
        "metadata": {
            "title": title,
            "description": meta_desc,
            "pages_crawled": len(pages),
            "total_characters": total_chars,
            "url": url,
        },
    }


# ── LLM-Verified Search (Enhanced for 4-Pillar System) ──────────────────────

async def verify_search_results(
    results: list[dict],
    company_name: str,
    company_description: str,
    llm_caller=None,
) -> list[dict]:
    """
    Batch-verify search results using an LLM for relevance and trustworthiness.

    Args:
        results: Raw search results with url, title, snippet
        company_name: Brand name for context
        company_description: What the company does
        llm_caller: Async function(task_name, system_prompt, user_prompt, ...) -> dict
                    Passed from seo_agent to avoid circular imports.

    Returns:
        Same results list with added verification fields per result
    """
    if not results or llm_caller is None:
        # Return results with default verification if no LLM available
        for r in results:
            r["verified_relevant"] = True
            r["trustworthiness_score"] = 50
            r["visibility_impact"] = "medium"
            r["llm_reasoning"] = "Unverified (LLM unavailable)"
        return results

    # Process in batches of 5
    batch_size = 5
    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]
        batch_data = [
            {"url": r["url"], "title": r["title"], "snippet": r.get("snippet", "")}
            for r in batch
        ]

        prompt = f"""You are verifying Google search results for relevance to a specific brand.

Brand: {company_name}
Description: {company_description}

Search results to verify:
{json.dumps(batch_data, indent=2)}

For each result, determine:
1. Is it genuinely relevant to this brand (not just keyword coincidence)?
2. How trustworthy is the source (0-100)?
3. What's the visibility impact (high/medium/low)?

Return JSON with key "verified" containing an array matching the input order:
[{{"verified_relevant": bool, "trustworthiness_score": int, "visibility_impact": "high"|"medium"|"low", "llm_reasoning": "brief reason"}}]"""

        try:
            parsed = await llm_caller(
                task_name="verify_search_results",
                system_prompt="You verify search results for brand relevance. Return valid JSON only.",
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=800,
            )
            verified_list = parsed.get("verified", [])
            for j, v in enumerate(verified_list):
                if j < len(batch) and isinstance(v, dict):
                    batch[j]["verified_relevant"] = v.get("verified_relevant", True)
                    batch[j]["trustworthiness_score"] = max(
                        0, min(int(v.get("trustworthiness_score", 50)), 100)
                    )
                    batch[j]["visibility_impact"] = v.get("visibility_impact", "medium")
                    batch[j]["llm_reasoning"] = v.get("llm_reasoning", "")
        except Exception as e:
            logger.warning(f"[SEO] Search verification batch failed: {e}")
            for r in batch:
                r.setdefault("verified_relevant", True)
                r.setdefault("trustworthiness_score", 50)
                r.setdefault("visibility_impact", "medium")
                r.setdefault("llm_reasoning", "Verification failed")

    return results


async def search_brand_presence_enhanced(
    brand_name: str,
    company_description: str,
    keywords: list[str],
    platforms: list[str] = None,
    llm_caller=None,
) -> dict:
    """
    Enhanced brand presence search with LLM verification.
    Wraps search_brand_presence() and adds verification + company name search.

    Args:
        brand_name: Company/brand name
        company_description: What the company does
        keywords: Target keywords
        platforms: Platforms to check
        llm_caller: LLM function for verification

    Returns:
        Dict per platform with mention_count, verified_mention_count, top_results
    """
    # Run standard brand presence search
    presence = await search_brand_presence(brand_name, keywords, platforms)

    # Also search Google specifically for company name (broader than site:-scoped)
    try:
        company_search = await search_google(f'"{brand_name}"', num_results=20)
        company_results = [
            {
                "url": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position", 0),
                "date": item.get("date", ""),
                "platform": _detect_platform(item.get("link", "")),
            }
            for item in company_search.get("organic", [])
        ]
    except Exception as e:
        logger.warning(f"[SEO] Company name search failed: {e}")
        company_results = []

    # LLM-verify all top results across platforms
    all_results_to_verify = []
    result_platform_map = []

    for platform, data in presence.items():
        for r in data.get("top_results", []):
            all_results_to_verify.append(r)
            result_platform_map.append(platform)

    if all_results_to_verify and llm_caller:
        all_results_to_verify = await verify_search_results(
            all_results_to_verify, brand_name, company_description, llm_caller
        )

    # Put verified results back and compute verified_mention_count
    idx = 0
    for platform, data in presence.items():
        verified_count = 0
        for r in data.get("top_results", []):
            if idx < len(all_results_to_verify):
                verified = all_results_to_verify[idx]
                r["verified_relevant"] = verified.get("verified_relevant", True)
                r["trustworthiness_score"] = verified.get("trustworthiness_score", 50)
                r["visibility_impact"] = verified.get("visibility_impact", "medium")
                if r.get("verified_relevant"):
                    verified_count += 1
                idx += 1
        data["verified_mention_count"] = verified_count

    # Add company_name_results to Google platform data
    if "google" in presence:
        presence["google"]["company_name_results"] = company_results[:10]

    return presence
