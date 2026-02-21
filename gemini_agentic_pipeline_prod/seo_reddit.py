# seo_reddit.py
"""
Reddit search agent — uses the public .json API (no OAuth required).

Rate limits without OAuth: ~10 req/min general, ~3-5 for search.
With OAuth (future): 100 req/min.

Returns results in the same format as seo_search.find_engagement_opportunities
so the orchestrator can merge them seamlessly.
"""

import logging
from typing import Optional
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = "BrandVerseSEO/1.0 (engagement research)"
_BASE = "https://www.reddit.com"
_TIMEOUT = 10.0


async def search_reddit(
    query: str,
    *,
    sort: str = "relevance",
    time_filter: str = "week",
    limit: int = 10,
    subreddit: Optional[str] = None,
) -> list[dict]:
    """
    Search Reddit via the public .json API.

    Args:
        query:       Search terms.
        sort:        relevance | hot | top | new | comments
        time_filter: hour | day | week | month | year | all
        limit:       Max results (1-100).
        subreddit:   Optional — restrict search to a single subreddit.

    Returns:
        List of dicts matching the seo_search result schema:
        {url, title, snippet, platform, date, search_query, upvotes, num_comments, subreddit}
    """
    path = f"/r/{subreddit}/search.json" if subreddit else "/search.json"
    params = {
        "q": query,
        "sort": sort,
        "t": time_filter,
        "limit": str(min(limit, 100)),
        "type": "link",         # Posts only (not subreddits/users)
        "restrict_sr": "on" if subreddit else "off",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}{path}",
                params=params,
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"[Reddit] HTTP {e.response.status_code} for query '{query}': {e}")
        return []
    except Exception as e:
        logger.error(f"[Reddit] Search failed for '{query}': {e}")
        return []

    results = []
    children = data.get("data", {}).get("children", [])
    for child in children:
        post = child.get("data", {})
        if not post:
            continue

        # Skip removed / deleted posts
        if post.get("removed_by_category") or post.get("selftext") == "[removed]":
            continue

        permalink = post.get("permalink", "")
        url = f"https://www.reddit.com{permalink}" if permalink else ""
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        snippet = selftext[:300] if selftext else title

        # Unix timestamp → ISO date string
        created_utc = post.get("created_utc", 0)
        date_str = ""
        if created_utc:
            from datetime import datetime, timezone
            date_str = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime("%Y-%m-%d")

        results.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "platform": "reddit",
            "date": date_str,
            "search_query": query,
            # Reddit-specific metadata (extra, won't break the pipeline)
            "upvotes": post.get("ups", 0),
            "num_comments": post.get("num_comments", 0),
            "subreddit": post.get("subreddit", ""),
            "is_self": post.get("is_self", False),
        })

    logger.info(f"[Reddit] query='{query}' → {len(results)} results")
    return results


async def find_reddit_engagement(
    keywords: list[str],
    *,
    num_per_keyword: int = 5,
    time_filter: str = "week",
    subreddits: Optional[list[str]] = None,
) -> list[dict]:
    """
    High-level: search Reddit for engagement opportunities from a list of keywords.
    Deduplicates by URL.

    Args:
        keywords:        List of search terms / phrases.
        num_per_keyword: Results per keyword.
        time_filter:     Recency filter.
        subreddits:      Optional list of subreddits to search within.

    Returns:
        Deduplicated list in the standard result format.
    """
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for kw in keywords:
        targets = subreddits if subreddits else [None]
        for sub in targets:
            hits = await search_reddit(
                query=kw,
                sort="relevance",
                time_filter=time_filter,
                limit=num_per_keyword,
                subreddit=sub,
            )
            for hit in hits:
                if hit["url"] and hit["url"] not in seen_urls:
                    seen_urls.add(hit["url"])
                    all_results.append(hit)

    # Sort by engagement signal: upvotes + comments
    all_results.sort(
        key=lambda r: r.get("upvotes", 0) + r.get("num_comments", 0),
        reverse=True,
    )

    logger.info(f"[Reddit] find_reddit_engagement → {len(all_results)} unique results from {len(keywords)} keywords")
    return all_results
