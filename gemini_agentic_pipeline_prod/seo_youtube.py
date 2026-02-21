# seo_youtube.py
"""
YouTube search agent — uses the YouTube Data API v3.

Free quota: 10,000 units/day.
  - search.list    = 100 units  (~100 searches/day)
  - commentThreads = 1 unit     (~10,000 reads/day)
  - videos.list    = 1 unit

Strategy: Use search.list sparingly, commentThreads.list heavily.

Env var: YOUTUBE_API_KEY (from Google Cloud Console)

Returns results in the same format as seo_search.find_engagement_opportunities
so the orchestrator can merge them seamlessly.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
_BASE = "https://www.googleapis.com/youtube/v3"
_TIMEOUT = 15.0


def _published_after(time_filter: str) -> str:
    """Convert a simple time filter to RFC 3339 publishedAfter param."""
    now = datetime.now(timezone.utc)
    deltas = {
        "hour": timedelta(hours=1),
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
        "year": timedelta(days=365),
    }
    delta = deltas.get(time_filter, timedelta(weeks=1))
    return (now - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


async def search_youtube(
    query: str,
    *,
    max_results: int = 5,
    time_filter: str = "week",
    order: str = "relevance",
) -> list[dict]:
    """
    Search YouTube videos via Data API v3.

    Cost: 100 units per call.

    Args:
        query:       Search terms.
        max_results: 1-50 (default 5 to conserve quota).
        time_filter: hour | day | week | month | year
        order:       relevance | date | viewCount | rating

    Returns:
        List of dicts in the standard result format:
        {url, title, snippet, platform, date, search_query, video_id, channel_title}
    """
    if not YOUTUBE_API_KEY:
        logger.warning("[YouTube] YOUTUBE_API_KEY not set — skipping search")
        return []

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": str(min(max_results, 50)),
        "order": order,
        "publishedAfter": _published_after(time_filter),
        "relevanceLanguage": "en",
        "key": YOUTUBE_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_BASE}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"[YouTube] HTTP {e.response.status_code} for '{query}': {e}")
        return []
    except Exception as e:
        logger.error(f"[YouTube] Search failed for '{query}': {e}")
        return []

    results = []
    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId", "")
        snippet = item.get("snippet", {})
        if not video_id:
            continue

        pub_date = snippet.get("publishedAt", "")[:10]  # "2026-02-15"

        results.append({
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": snippet.get("title", ""),
            "snippet": snippet.get("description", "")[:300],
            "platform": "youtube",
            "date": pub_date,
            "search_query": query,
            # YouTube-specific metadata
            "video_id": video_id,
            "channel_title": snippet.get("channelTitle", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        })

    logger.info(f"[YouTube] search '{query}' → {len(results)} videos (cost: 100 units)")
    return results


async def get_video_comments(
    video_id: str,
    *,
    max_results: int = 20,
) -> list[dict]:
    """
    Fetch top-level comment threads for a video.

    Cost: 1 unit per call (very cheap).

    Returns:
        List of comment dicts: {text, author, likes, published_at, reply_count}
    """
    if not YOUTUBE_API_KEY:
        return []

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": str(min(max_results, 100)),
        "order": "relevance",
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_BASE}/commentThreads", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        # Comments disabled or video not found
        if e.response.status_code == 403:
            logger.debug(f"[YouTube] Comments disabled for {video_id}")
        else:
            logger.warning(f"[YouTube] commentThreads HTTP {e.response.status_code}: {e}")
        return []
    except Exception as e:
        logger.error(f"[YouTube] commentThreads failed for {video_id}: {e}")
        return []

    comments = []
    for item in data.get("items", []):
        top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        if not top:
            continue
        comments.append({
            "text": top.get("textDisplay", ""),
            "author": top.get("authorDisplayName", ""),
            "likes": top.get("likeCount", 0),
            "published_at": top.get("publishedAt", "")[:10],
            "reply_count": item.get("snippet", {}).get("totalReplyCount", 0),
        })

    return comments


async def find_youtube_engagement(
    keywords: list[str],
    *,
    num_per_keyword: int = 3,
    time_filter: str = "week",
    include_comments: bool = True,
    max_comments_per_video: int = 10,
) -> list[dict]:
    """
    High-level: search YouTube for engagement opportunities.

    Strategy:
    - search.list per keyword (100 units each — use sparingly)
    - commentThreads.list per result (1 unit each — use generously)
    - Enrich results with comment context for LLM scoring

    Args:
        keywords:               Search terms.
        num_per_keyword:        Videos per keyword (keep low to save quota).
        time_filter:            Recency.
        include_comments:       Whether to fetch top comments (adds engagement context).
        max_comments_per_video: Comments to fetch per video.

    Returns:
        Deduplicated list in the standard result format, enriched with comment data.
    """
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for kw in keywords:
        hits = await search_youtube(
            query=kw,
            max_results=num_per_keyword,
            time_filter=time_filter,
        )
        for hit in hits:
            if hit["url"] in seen_urls:
                continue
            seen_urls.add(hit["url"])

            # Enrich with comment threads (cheap: 1 unit each)
            if include_comments and hit.get("video_id"):
                comments = await get_video_comments(
                    hit["video_id"],
                    max_results=max_comments_per_video,
                )
                if comments:
                    # Add top comments as context for LLM scoring
                    top_comments = [c["text"][:150] for c in comments[:3]]
                    hit["snippet"] += "\n\nTop comments: " + " | ".join(top_comments)
                    hit["comment_count"] = len(comments)
                    hit["top_comment_likes"] = max((c["likes"] for c in comments), default=0)

            all_results.append(hit)

    logger.info(
        f"[YouTube] find_youtube_engagement → {len(all_results)} unique videos "
        f"from {len(keywords)} keywords "
        f"(est. cost: {len(keywords) * 100 + len(all_results)} units)"
    )
    return all_results
