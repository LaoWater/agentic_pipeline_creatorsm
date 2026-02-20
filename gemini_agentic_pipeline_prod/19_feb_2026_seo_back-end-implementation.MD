# SEO Backend Implementation - 19 Feb 2026

## What Changed

The SEO Agent system was split across two backends. The two heaviest operations - **presence analysis** and **engagement discovery** - were moved from Supabase Edge Functions to the Python Cloud Run backend. This was done because the Edge Functions had no ability to perform real web searches. They were calling an LLM and asking it to *imagine* what Reddit posts and YouTube videos might exist, producing fabricated URLs that returned 404s.

Now the Python backend uses **Serper.dev** (Google Search API) to find real URLs, real discussions, and real engagement opportunities.

---

## Architecture: Before vs After

### BEFORE (everything through Edge Functions)

```
Frontend (SeoAgent.tsx)
  │
  ├── runAnalysis()        → supabase.functions.invoke('seo-analysis')
  ├── searchEngagement()   → supabase.functions.invoke('seo-engine', {action: 'find_engagement'})
  └── generateBlogPost()   → supabase.functions.invoke('seo-engine', {action: 'generate_blog'})
        │
        ▼
  Supabase Edge Functions (Deno)
        │
        ├── seo-analysis/index.ts    → Crawls website (regex HTML, max 3 pages)
        │                             → Calls GPT for analysis (no real search data)
        │                             → Saves to seo_analysis table
        │
        └── seo-engine/index.ts      → generate_blog: Calls GPT for blog post
                                      → find_engagement: Calls GPT to FABRICATE opportunities
                                         (fake URLs, fake Reddit posts, fake discussions)
```

### AFTER (hybrid: Python backend + Edge Functions)

```
Frontend (SeoAgent.tsx)
  │
  ├── runAnalysis()        → [NEW] Python Cloud Run backend /seo/analyze-presence
  │                           Credits: deducted via supabase.rpc('deduct_credits') BEFORE calling
  │                           Results: saved to seo_analysis table by frontend after receiving
  │
  ├── searchEngagement()   → [NEW] Python Cloud Run backend /seo/search-engagement
  │                           Credits: deducted via supabase.rpc('deduct_credits') BEFORE calling
  │                           Results: saved to seo_engagement_opportunities table by frontend
  │
  └── generateBlogPost()   → [UNCHANGED] supabase.functions.invoke('seo-engine')
                              Still uses Edge Function (no web search needed for blog generation)
                              Credits: deducted inside Edge Function (server-side)
```

### Visual Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                              │
│                        SeoAgent.tsx                                   │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│  │  Analysis Tab    │  │   Engine Tab     │  │   Engine Tab      │   │
│  │  runAnalysis()   │  │  searchEngage()  │  │  generateBlog()   │   │
│  └───────┬──────────┘  └───────┬──────────┘  └───────┬───────────┘   │
│          │                     │                     │               │
│     ┌────┴─────┐          ┌────┴─────┐          ┌────┴─────┐        │
│     │ 1.deduct │          │ 1.deduct │          │          │        │
│     │ credits  │          │ credits  │          │          │        │
│     │ (RPC)    │          │ (RPC)    │          │          │        │
│     └────┬─────┘          └────┬─────┘          │          │        │
│          │                     │                │          │        │
│     ┌────┴──────┐         ┌────┴──────┐         │          │        │
│     │ 2.call    │         │ 2.call    │         │ invoke   │        │
│     │ Python    │         │ Python    │         │ Edge Fn  │        │
│     │ backend   │         │ backend   │         │          │        │
│     └────┬──────┘         └────┬──────┘         └────┬─────┘        │
│          │                     │                     │               │
│     ┌────┴──────┐         ┌────┴──────┐              │               │
│     │ 3.save to │         │ 3.save to │              │               │
│     │ Supabase  │         │ Supabase  │              │               │
│     └───────────┘         └───────────┘              │               │
└──────────┬─────────────────────┬─────────────────────┬───────────────┘
           │                     │                     │
           ▼                     ▼                     ▼
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐
│  Python Cloud Run  │ │  Python Cloud Run  │ │  Supabase Edge Fn      │
│  /seo/analyze-     │ │  /seo/search-      │ │  seo-engine/index.ts   │
│  presence          │ │  engagement        │ │  action: generate_blog │
│                    │ │                    │ │                        │
│ 1. Crawl website   │ │ 1. GPT generates   │ │ 1. Deducts credits     │
│    (httpx + BS4,   │ │    smart search    │ │ 2. Calls GPT-5.2       │
│    up to 8 pages,  │ │    queries         │ │    for blog post       │
│    50K chars)      │ │                    │ │ 3. Saves to            │
│                    │ │ 2. Serper.dev      │ │    seo_blog_posts      │
│ 2. Serper.dev      │ │    executes REAL   │ │                        │
│    searches brand  │ │    Google searches │ └────────────────────────┘
│    across Reddit,  │ │    (8-12 queries)  │
│    YouTube, Quora, │ │                    │
│    Twitter,        │ │ 3. GPT scores      │
│    LinkedIn,       │ │    results & crafts│
│    Google          │ │    responses       │
│                    │ │                    │
│ 3. Serper.dev      │ │ Returns:           │
│    searches        │ │  - Real URLs       │
│    competitors     │ │  - Real titles     │
│                    │ │  - Relevance score │
│ 4. GPT-5.2         │ │  - Suggested reply │
│    analysis with   │ │  - url_verified:   │
│    REAL data       │ │    true            │
│                    │ └────────────────────┘
│ Returns:           │
│  - Data-backed     │
│    platform scores │
│  - Search evidence │
│  - Competitor      │
│    comparison      │
│  - Recommendations │
└────────────────────┘
           │                     │
           ▼                     ▼
   ┌───────────────────────────────────┐
   │         Serper.dev API            │
   │   (Real Google Search Results)    │
   │                                   │
   │  site:reddit.com "keywords"       │
   │  site:youtube.com "keywords"      │
   │  site:quora.com "keywords"        │
   │  "brand name" general search      │
   │                                   │
   │  Returns: real URLs, real titles, │
   │  real snippets, real dates        │
   └───────────────────────────────────┘
```

---

## Files Created/Modified

### Python Backend (agentic_pipeline_creatorsm/gemini_agentic_pipeline_prod/)

| File | Status | Purpose |
|------|--------|---------|
| `seo_search.py` | **NEW** | Serper.dev API wrapper. Functions: `search_google()`, `search_platform()`, `search_brand_presence()`, `find_engagement_opportunities()`, `crawl_website()` |
| `seo_agent.py` | **NEW** | SEO intelligence pipelines. Endpoints: `/seo/search-engagement`, `/seo/analyze-presence`. Uses GPT-5.2 for query generation, scoring, and analysis |
| `main.py` | **MODIFIED** | Added imports for SEO agent, registered two new POST endpoints |

### Frontend (brandverse-ai-studio/)

| File | Status | Purpose |
|------|--------|---------|
| `src/services/seoService.ts` | **NEW** | Frontend service layer. `searchEngagement()` and `analyzePresence()` call Python backend directly (same pattern as ContentGenerator) |
| `src/pages/SeoAgent.tsx` | **MODIFIED** | `runAnalysis()` and `searchEngagement()` now call Python backend instead of Edge Functions. Added progress stages, "Verified URL" badges, mention counts on platform scores |
| `src/services/creditsService.ts` | **MODIFIED** | Updated `SEO_CREDITS`: Analysis 3→5, Engagement 2→3 (real search costs more) |
| `supabase/migrations/20260219000000_seo_schema_enhancements.sql` | **NEW** | Adds `search_data`, `competitor_data` columns to `seo_analysis`. Adds `url_verified`, `discovered_via` to `seo_engagement_opportunities`. Expands platform constraint to include youtube, linkedin |

### Edge Functions (still exist, reduced role)

| File | Status | Role |
|------|--------|------|
| `supabase/functions/seo-analysis/index.ts` | **BYPASSED** | No longer called by frontend. Kept as fallback. |
| `supabase/functions/seo-engine/index.ts` | **STILL USED** | Only for `action: 'generate_blog'`. The `action: 'find_engagement'` path is bypassed (frontend calls Python backend instead). |

---

## Credit Flow Change

### Before: Credits deducted server-side inside Edge Functions
```
Frontend → Edge Function → deduct_credits RPC → process → save → respond
```

### After (Analysis & Engagement): Credits deducted client-side before calling Python backend
```
Frontend → supabase.rpc('deduct_credits') → Python backend → Frontend saves to Supabase
```

### After (Blog Post): Unchanged
```
Frontend → Edge Function seo-engine → deduct_credits RPC → GPT → save → respond
```

---

## What Serper.dev Provides

- **2,500 free queries** then $1 per 1,000 queries
- Real Google search results via `site:` operators target specific platforms
- Each analysis uses ~15-20 Serper queries (brand presence across 5 platforms + keyword searches + competitor searches)
- Each engagement search uses ~8-12 Serper queries
- Every URL returned has been indexed by Google - it exists, it's real

---

## Database Schema Additions

```sql
-- seo_analysis table
search_data     JSONB   -- Raw Serper results per platform (mention counts, top URLs)
competitor_data JSONB   -- Competitor mention counts per platform

-- seo_engagement_opportunities table
url_verified    BOOLEAN -- true for Serper-sourced URLs
discovered_via  TEXT    -- 'serper', 'reddit_api', 'youtube_api', 'llm_generated'

-- Platform constraint expanded
platform IN ('reddit','twitter','facebook','youtube','quora','forum','linkedin','other')
```

---

## Deployment Checklist

1. **Add env var to Cloud Run**: `SERPER_API_KEY=`
2. **Redeploy Python backend** to Cloud Run (new files: seo_search.py, seo_agent.py, modified main.py)
3. **Run Supabase migration**: `20260219000000_seo_schema_enhancements.sql`
4. **Deploy frontend** (new seoService.ts, modified SeoAgent.tsx, modified creditsService.ts)
5. Edge functions do NOT need redeployment (blog generation path unchanged)
