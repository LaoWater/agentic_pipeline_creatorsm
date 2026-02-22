# seo_website_audit.py
"""
SEO Pillar 4: Website Technical Analysis.
Audits a website's crawlability, meta tags, tech stack, and search engine readiness.
"""

import asyncio
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

GOOGLEBOT_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


async def audit_website(url: str) -> dict:
    """
    Full technical audit of a website for SEO readiness.
    
    Pipeline:
    1. Fetch as Googlebot — get raw HTML, measure response time
    2. Fetch as browser — for comparison
    3. Parse HTML with BeautifulSoup
    4. Detect tech stack from HTML patterns
    5. Check robots.txt
    6. Check sitemap.xml
    7. Calculate crawlability score (0-100)
    8. Identify issues with severity
    
    Returns:
        dict with all audit data fields matching the seo_website_audits DB table
    """
    if not url.startswith("http"):
        url = f"https://{url}"
    
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    
    logger.info(f"[SEO Audit] Starting technical audit for {url}")
    
    # Step 1 & 2: Fetch as Googlebot and as browser in parallel
    googlebot_data = {"html": "", "status": 0, "response_time_ms": 0, "content_type": ""}
    browser_data = {"html": "", "status": 0}
    
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        # Fetch as Googlebot
        try:
            start = time.monotonic()
            resp = await client.get(url, headers={"User-Agent": GOOGLEBOT_UA})
            elapsed_ms = int((time.monotonic() - start) * 1000)
            googlebot_data = {
                "html": resp.text,
                "status": resp.status_code,
                "response_time_ms": elapsed_ms,
                "content_type": resp.headers.get("content-type", ""),
            }
        except Exception as e:
            logger.error(f"[SEO Audit] Googlebot fetch failed: {e}")
            googlebot_data["status"] = 0
        
        # Fetch as browser
        try:
            resp = await client.get(url, headers={"User-Agent": BROWSER_UA})
            browser_data = {"html": resp.text, "status": resp.status_code}
        except Exception as e:
            logger.warning(f"[SEO Audit] Browser fetch failed: {e}")
    
    if not googlebot_data["html"] and not browser_data["html"]:
        return _empty_audit(url, "Failed to fetch website")
    
    html = googlebot_data["html"] or browser_data["html"]
    
    # Step 3: Parse HTML
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract visible text
    visible_text = _extract_visible_text(soup)
    raw_html_length = len(html)
    visible_text_length = len(visible_text)
    
    # Compare Googlebot vs browser text
    browser_soup = BeautifulSoup(browser_data["html"], "html.parser") if browser_data["html"] else soup
    browser_text = _extract_visible_text(browser_soup)
    
    # If Googlebot sees >50% less text than browser, content is JS-dependent
    js_dependent = False
    if browser_text and visible_text:
        ratio = len(visible_text) / max(len(browser_text), 1)
        js_dependent = ratio < 0.5
    elif not visible_text and browser_text:
        js_dependent = True
    
    # Meta tags
    meta_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    has_meta_title = bool(meta_title)
    
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""
    has_meta_description = bool(meta_description)
    
    # OG tags
    og_tags = soup.find_all("meta", attrs={"property": re.compile(r"^og:")})
    has_og_tags = len(og_tags) >= 2  # At least og:title and og:description
    
    # Canonical URL
    canonical_tag = soup.find("link", attrs={"rel": "canonical"})
    canonical_url = canonical_tag.get("href", "").strip() if canonical_tag else ""
    has_canonical_url = bool(canonical_url)
    
    # Step 4: Detect tech stack
    tech_stack = _detect_tech_stack(html, soup)
    is_spa = any(t["name"] in ("React SPA", "Vue SPA", "Angular SPA") for t in tech_stack)
    has_ssr = any(t.get("ssr", False) for t in tech_stack)
    
    # Step 5: Check robots.txt
    has_robots_txt = await _check_robots(base_url)
    
    # Step 6: Check sitemap
    has_sitemap = await _check_sitemap(base_url)
    
    # Step 7: Calculate crawlability score
    score = 0
    # +25 for meta title + description
    if has_meta_title:
        score += 15
    if has_meta_description:
        score += 10
    # +20 for content visible to bots
    if not js_dependent and visible_text_length > 200:
        score += 20
    elif visible_text_length > 100:
        score += 10
    # +15 for OG tags
    if has_og_tags:
        score += 15
    # +15 for canonical URL
    if has_canonical_url:
        score += 15
    # +10 for robots.txt
    if has_robots_txt:
        score += 10
    # +10 for sitemap
    if has_sitemap:
        score += 10
    # +5 for fast response (<2s)
    if googlebot_data["response_time_ms"] > 0 and googlebot_data["response_time_ms"] < 2000:
        score += 5
    
    # Step 8: Identify issues
    issues = _identify_issues(
        has_meta_title=has_meta_title,
        has_meta_description=has_meta_description,
        has_og_tags=has_og_tags,
        has_canonical_url=has_canonical_url,
        has_robots_txt=has_robots_txt,
        has_sitemap=has_sitemap,
        js_dependent=js_dependent,
        response_time_ms=googlebot_data["response_time_ms"],
        is_spa=is_spa,
        has_ssr=has_ssr,
        visible_text_length=visible_text_length,
        tech_stack=tech_stack,
    )
    
    # Generate recommendations from issues
    recommendations = [
        {"priority": i["severity"], "category": i["category"], "action": i["recommendation"]}
        for i in issues if i.get("recommendation")
    ]
    
    # Technical summary
    tech_names = [t["name"] for t in tech_stack]
    summary_parts = []
    if tech_names:
        summary_parts.append(f"Tech stack: {', '.join(tech_names)}.")
    summary_parts.append(f"Crawlability score: {score}/100.")
    critical_count = sum(1 for i in issues if i["severity"] == "critical")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    if critical_count:
        summary_parts.append(f"{critical_count} critical issue{'s' if critical_count > 1 else ''}.")
    if warning_count:
        summary_parts.append(f"{warning_count} warning{'s' if warning_count > 1 else ''}.")
    if not critical_count and not warning_count:
        summary_parts.append("No major issues found.")
    
    result = {
        "url": url,
        "http_status": googlebot_data["status"],
        "response_time_ms": googlebot_data["response_time_ms"],
        "content_type": googlebot_data["content_type"],
        "raw_html_length": raw_html_length,
        "visible_text_length": visible_text_length,
        "js_dependent_content": js_dependent,
        "has_meta_title": has_meta_title,
        "meta_title": meta_title,
        "has_meta_description": has_meta_description,
        "meta_description": meta_description,
        "has_og_tags": has_og_tags,
        "has_canonical_url": has_canonical_url,
        "canonical_url": canonical_url,
        "has_robots_txt": has_robots_txt,
        "has_sitemap": has_sitemap,
        "inferred_tech_stack": tech_stack,
        "is_spa": is_spa,
        "has_ssr": has_ssr,
        "crawlability_score": score,
        "issues": issues,
        "technical_summary": " ".join(summary_parts),
        "recommendations": recommendations,
    }
    
    logger.info(f"[SEO Audit] Completed: score={score}, issues={len(issues)}, tech={tech_names}")
    return result


def _extract_visible_text(soup: BeautifulSoup) -> str:
    """Extract visible text content, stripping scripts/styles/nav/footer."""
    clone = BeautifulSoup(str(soup), "html.parser")
    for tag in clone(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    text = clone.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)


def _detect_tech_stack(html: str, soup: BeautifulSoup) -> list[dict]:
    """Detect technology stack from HTML patterns."""
    techs = []
    html_lower = html.lower()
    
    # Next.js
    if "__next_data__" in html_lower or "__next" in html_lower:
        techs.append({"name": "Next.js", "confidence": "high", "ssr": True})
    
    # React (SPA vs SSR)
    root_div = soup.find("div", {"id": "root"})
    if root_div:
        # If root div has minimal children, likely a React SPA
        children_text = root_div.get_text(strip=True)
        if len(children_text) < 100:
            techs.append({"name": "React SPA", "confidence": "high", "ssr": False})
        else:
            techs.append({"name": "React", "confidence": "medium", "ssr": True})
    elif "react" in html_lower and "__next" not in html_lower:
        techs.append({"name": "React", "confidence": "low", "ssr": False})
    
    # Vue.js
    if "data-v-" in html:
        techs.append({"name": "Vue.js", "confidence": "high", "ssr": False})
    
    # Nuxt.js
    if "_nuxt" in html_lower or "__nuxt" in html_lower:
        techs.append({"name": "Nuxt.js", "confidence": "high", "ssr": True})
    
    # Angular
    if "ng-version" in html_lower or "_ngcontent" in html_lower:
        if soup.find("app-root") and len(soup.find("app-root").get_text(strip=True)) < 100:
            techs.append({"name": "Angular SPA", "confidence": "high", "ssr": False})
        else:
            techs.append({"name": "Angular", "confidence": "medium", "ssr": False})
    
    # WordPress
    if "wp-content" in html_lower or "wordpress" in html_lower:
        techs.append({"name": "WordPress", "confidence": "high", "ssr": True})
    
    # Webflow
    if "webflow" in html_lower:
        techs.append({"name": "Webflow", "confidence": "high", "ssr": True})
    
    # Shopify
    if "shopify" in html_lower or "cdn.shopify.com" in html_lower:
        techs.append({"name": "Shopify", "confidence": "high", "ssr": True})
    
    # Wix
    if "wix.com" in html_lower or "_wix" in html_lower:
        techs.append({"name": "Wix", "confidence": "high", "ssr": True})
    
    # Gatsby
    if "gatsby" in html_lower or "___gatsby" in html_lower:
        techs.append({"name": "Gatsby", "confidence": "high", "ssr": True})
    
    # Svelte / SvelteKit
    if "svelte" in html_lower:
        techs.append({"name": "SvelteKit", "confidence": "medium", "ssr": True})
    
    return techs


async def _check_robots(base_url: str) -> bool:
    """Check if robots.txt exists and is accessible."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"{base_url}/robots.txt", headers={"User-Agent": GOOGLEBOT_UA})
            return resp.status_code == 200 and len(resp.text) > 10
    except Exception:
        return False


async def _check_sitemap(base_url: str) -> bool:
    """Check if sitemap.xml exists."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"{base_url}/sitemap.xml", headers={"User-Agent": GOOGLEBOT_UA})
            return resp.status_code == 200 and ("<?xml" in resp.text[:200] or "<urlset" in resp.text[:500])
    except Exception:
        return False


def _identify_issues(
    has_meta_title: bool,
    has_meta_description: bool,
    has_og_tags: bool,
    has_canonical_url: bool,
    has_robots_txt: bool,
    has_sitemap: bool,
    js_dependent: bool,
    response_time_ms: int,
    is_spa: bool,
    has_ssr: bool,
    visible_text_length: int,
    tech_stack: list[dict],
) -> list[dict]:
    """Identify SEO issues with severity levels."""
    issues = []
    
    # Critical issues
    if js_dependent:
        issues.append({
            "severity": "critical",
            "category": "crawlability",
            "title": "Your App Works. Indexing Doesn't.",
            "message": "Googlebot sees significantly less content than a regular browser. Your content is likely rendered via JavaScript, which search engines may not fully execute.",
            "recommendation": "Implement server-side rendering (SSR) or static site generation (SSG). Consider Next.js, Nuxt.js, or a pre-rendering service like Prerender.io.",
        })
    
    if not has_meta_title:
        issues.append({
            "severity": "critical",
            "category": "meta",
            "title": "Missing Page Title",
            "message": "No <title> tag found. This is the most important on-page SEO element.",
            "recommendation": "Add a descriptive <title> tag (50-60 characters) that includes your primary keyword.",
        })
    
    if visible_text_length < 100:
        issues.append({
            "severity": "critical",
            "category": "content",
            "title": "Insufficient Visible Content",
            "message": f"Only {visible_text_length} characters of visible text detected. Search engines need substantial text content to understand and rank your page.",
            "recommendation": "Add meaningful text content. Aim for at least 300+ words on your homepage.",
        })
    
    # Warning issues
    if not has_meta_description:
        issues.append({
            "severity": "warning",
            "category": "meta",
            "title": "Missing Meta Description",
            "message": "No meta description found. Google will auto-generate one, but it may not match your intent.",
            "recommendation": "Add a compelling meta description (150-160 characters) that summarizes the page and includes target keywords.",
        })
    
    if not has_og_tags:
        issues.append({
            "severity": "warning",
            "category": "social",
            "title": "Missing Open Graph Tags",
            "message": "No OG tags found. Links shared on social media will show a generic preview.",
            "recommendation": "Add og:title, og:description, og:image, and og:url meta tags for better social sharing.",
        })
    
    if not has_canonical_url:
        issues.append({
            "severity": "warning",
            "category": "crawlability",
            "title": "Missing Canonical URL",
            "message": "No canonical URL specified. This can lead to duplicate content issues.",
            "recommendation": "Add a <link rel='canonical'> tag pointing to the preferred URL for this page.",
        })
    
    if not has_sitemap:
        issues.append({
            "severity": "warning",
            "category": "crawlability",
            "title": "Missing Sitemap",
            "message": "No sitemap.xml found. Sitemaps help search engines discover and index all your pages.",
            "recommendation": "Generate a sitemap.xml and submit it to Google Search Console.",
        })
    
    if response_time_ms > 3000:
        issues.append({
            "severity": "warning",
            "category": "performance",
            "title": "Slow Response Time",
            "message": f"Response time: {response_time_ms}ms. Google recommends under 2 seconds for optimal crawling.",
            "recommendation": "Optimize server response time. Consider CDN, caching, or upgrading hosting.",
        })
    
    # Info issues
    if not has_robots_txt:
        issues.append({
            "severity": "info",
            "category": "crawlability",
            "title": "Missing robots.txt",
            "message": "No robots.txt found. While not critical, it helps communicate crawl preferences to search engines.",
            "recommendation": "Create a robots.txt file at your domain root.",
        })
    
    if is_spa and not has_ssr:
        tech_names = [t["name"] for t in tech_stack]
        issues.append({
            "severity": "info",
            "category": "technology",
            "title": f"Single Page Application Detected ({', '.join(tech_names)})",
            "message": "SPAs can have indexing challenges if content is only rendered client-side.",
            "recommendation": "Monitor Google Search Console for indexing issues. Consider implementing SSR or pre-rendering.",
        })
    
    return issues


def _empty_audit(url: str, error_msg: str) -> dict:
    """Return an empty audit result when the site cannot be fetched."""
    return {
        "url": url,
        "http_status": 0,
        "response_time_ms": 0,
        "content_type": "",
        "raw_html_length": 0,
        "visible_text_length": 0,
        "js_dependent_content": False,
        "has_meta_title": False,
        "meta_title": "",
        "has_meta_description": False,
        "meta_description": "",
        "has_og_tags": False,
        "has_canonical_url": False,
        "canonical_url": "",
        "has_robots_txt": False,
        "has_sitemap": False,
        "inferred_tech_stack": [],
        "is_spa": False,
        "has_ssr": False,
        "crawlability_score": 0,
        "issues": [{
            "severity": "critical",
            "category": "accessibility",
            "title": "Website Unreachable",
            "message": error_msg,
            "recommendation": "Verify the URL is correct and the site is accessible.",
        }],
        "technical_summary": f"Could not audit: {error_msg}",
        "recommendations": [],
    }
