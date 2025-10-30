# CreatorsM Agentic Pipeline - API Documentation for Frontend

**Version:** 1.0.0
**Base URL:** `http://localhost:8000` (Development) | `https://your-production-url.com` (Production)
**Last Updated:** 2025-10-30

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [CORS Configuration](#cors-configuration)
4. [Endpoints](#endpoints)
   - [POST /generate-posts](#post-generate-posts)
   - [POST /generate-posts-enhanced](#post-generate-posts-enhanced)
5. [Data Models](#data-models)
6. [Image Generation Model Selection](#image-generation-model-selection)
7. [Response Format](#response-format)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

---

## Overview

The CreatorsM Agentic Pipeline API provides AI-powered social media content generation with support for:

- Multi-platform content adaptation (LinkedIn, Instagram, Twitter/X, Facebook)
- Multi-language translation
- AI-generated visual assets with customizable image controls
- Hierarchical image styling (global and platform-specific)
- Cloud storage integration
- **Custom image generation model selection**

---

## Authentication

Currently, the API does not require authentication. This may change in production deployments.

---

## CORS Configuration

The API supports CORS for the following origins:
- `http://localhost:8080`
- `http://localhost:5173`
- `https://creators-multiverse.com`

All HTTP methods and headers are allowed.

---

## Endpoints

### POST /generate-posts-enhanced

**Description:** Enhanced pipeline with hierarchical image controls and advanced styling options.

**Endpoint:** `/generate-posts-enhanced`
**Method:** `POST`
**Content-Type:** `application/json`

#### Request Body Schema

```typescript
{
  company: {
    id: string;                                      // Required
    name: string;                                    // Required
    mission: string;                                 // Required
    tone_of_voice: string;                          // Required
    primary_color_1: string;                        // Required (hex format, e.g., "#FF5733")
    primary_color_2: string;                        // Required (hex format)
    logo_path?: string;                             // Optional
  };
  content: {
    topic: string;                                   // Required
    description: string;                            // Required
    hashtags: string[];                             // Required
    call_to_action: string;                         // Required
  };
  image_control: {
    level_1: {                                      // Global settings
      enabled: boolean;                             // Required
      style: string;                                // Required (e.g., "minimalist", "vibrant")
      guidance: string;                             // Required (creative direction)
      caption: string;                              // Required
      ratio: string;                                // Required: "1:1" | "3:4" | "4:3" | "9:16" | "16:9"
      starting_image_url?: string;                  // Optional
    };
    level_2: {                                      // Platform-specific overrides
      facebook?: PlatformImageControl;
      instagram?: PlatformImageControl;
      linkedin?: PlatformImageControl;
      twitter?: PlatformImageControl;
    };
  };
  platforms: Array<{
    platform: string;                               // Required: "facebook" | "instagram" | "linkedin" | "twitter"
    post_type: string;                              // Required: "Text" | "Image" | "Video" | "Let Model Decide"
    selected: boolean;                              // Required
  }>;
  language?: string;                                // Optional, default: "English"
  upload_to_cloud?: boolean;                        // Optional, default: true
  image_generation_model?: string;                  // Optional, e.g., "imagen-4.0-fast-generate-001"
}
```

#### PlatformImageControl Schema

```typescript
{
  enabled: boolean;                                 // Whether platform-specific control is enabled
  style: string;                                    // Platform-specific style
  guidance: string;                                 // Platform-specific creative direction
  caption: string;                                  // Platform-specific caption
  ratio: string;                                    // Platform-specific aspect ratio
  starting_image_url?: string;                      // Optional starting image URL
}
```

#### Image Control Hierarchy

- **Level 1 (Global):** Default settings applied to all platforms
- **Level 2 (Platform-specific):** Overrides Level 1 for specific platforms
- **Priority:** Level 2 always overrides Level 1 when both exist for a platform

#### Aspect Ratio Values

**IMPORTANT:** Only use Google Imagen's supported aspect ratios:

- `"1:1"` - Square (Instagram posts, general use)
- `"3:4"` - Portrait (Instagram feed)
- `"4:3"` - Landscape (General landscape)
- `"9:16"` - Vertical (Stories, Reels, TikTok)
- `"16:9"` - Wide landscape (YouTube thumbnails, LinkedIn)

**Supported values:** `"1:1"`, `"3:4"`, `"4:3"`, `"9:16"`, `"16:9"`

**Note:** Do NOT use `"auto"`, `"1.91:1"`, `"4:5"`, or any other ratios. Google Imagen only supports the five ratios listed above.

#### Example Request

```json
{
  "company": {
    "id": "comp_12345",
    "name": "EcoTech Solutions",
    "mission": "Creating sustainable technology for a better tomorrow",
    "tone_of_voice": "Inspirational and eco-conscious",
    "primary_color_1": "#2ECC71",
    "primary_color_2": "#3498DB",
    "logo_path": "/assets/logos/ecotech-logo.png"
  },
  "content": {
    "topic": "Launch of our new solar-powered smart home system",
    "description": "Revolutionary smart home technology that's 100% powered by renewable energy",
    "hashtags": ["#SustainableTech", "#SmartHome", "#GreenEnergy"],
    "call_to_action": "Pre-order now and save 20%!"
  },
  "image_control": {
    "level_1": {
      "enabled": true,
      "style": "modern and clean",
      "guidance": "Focus on eco-friendly aesthetics with natural lighting",
      "caption": "Sustainable living made simple",
      "ratio": "1:1",
      "starting_image_url": null
    },
    "level_2": {
      "instagram": {
        "enabled": true,
        "style": "vibrant and eye-catching",
        "guidance": "Use bold colors and dynamic composition",
        "caption": "Eco-innovation at your fingertips",
        "ratio": "3:4",
        "starting_image_url": null
      },
      "linkedin": {
        "enabled": true,
        "style": "professional and polished",
        "guidance": "Emphasize technology and innovation",
        "caption": "Engineering a sustainable future",
        "ratio": "16:9",
        "starting_image_url": null
      }
    }
  },
  "platforms": [
    {
      "platform": "instagram",
      "post_type": "Image",
      "selected": true
    },
    {
      "platform": "linkedin",
      "post_type": "Image",
      "selected": true
    },
    {
      "platform": "twitter",
      "post_type": "Text",
      "selected": true
    },
    {
      "platform": "facebook",
      "post_type": "Image",
      "selected": false
    }
  ],
  "language": "English",
  "upload_to_cloud": true,
  "image_generation_model": "imagen-4.0-fast-generate-001"
}
```

---

## Image Generation Model Selection

### Overview

Both endpoints now support custom image generation model selection via the optional `image_generation_model` field. This allows frontend users to choose which AI model generates their visual assets.

### Default Behavior

If `image_generation_model` is **not provided** or is `null`, the system uses the default model configured in the backend:
- **Default Model:** `"imagen-4.0-fast-generate-001"`

### Available Models

| Model ID | Speed | Quality | Cost | Use Case |
|----------|-------|---------|------|----------|
| `imagen-4.0-fast-generate-001` | Fast | High | Moderate | **Default** - Best balance of speed, quality, and cost |
| `imagen-3.0-generate-002` | Moderate | Good | Low | Budget-friendly option, slightly lower quality |
| `imagen-4.0-generate-preview-06-06` | Moderate | Very High | +15% | Premium quality, worth the extra cost |
| `imagen-4.0-ultra-generate-preview-06-06` | Slower | Exceptional | +25% | Ultra-premium quality for special campaigns |

### Implementation Examples

#### Use Default Model
```json
{
  "company_name": "Example Corp",
  // ... other fields
  // image_generation_model not specified - uses default
}
```

#### Specify Custom Model
```json
{
  "company_name": "Example Corp",
  // ... other fields
  "image_generation_model": "imagen-4.0-ultra-generate-preview-06-06"
}
```

### Model Selection UI Recommendations

**Frontend Implementation Suggestions:**

1. **Dropdown/Select Component:**
```typescript
const imageModels = [
  {
    id: "imagen-4.0-fast-generate-001",
    name: "Standard Quality (Fast)",
    description: "Best balance - Recommended",
    recommended: true
  },
  {
    id: "imagen-3.0-generate-002",
    name: "Basic Quality",
    description: "Budget-friendly option"
  },
  {
    id: "imagen-4.0-generate-preview-06-06",
    name: "Premium Quality",
    description: "Enhanced visual quality (+15% cost)"
  },
  {
    id: "imagen-4.0-ultra-generate-preview-06-06",
    name: "Ultra Premium",
    description: "Exceptional quality (+25% cost)"
  }
];
```

2. **Default Value:**
```typescript
const [selectedModel, setSelectedModel] = useState(null); // Uses backend default
```

3. **Advanced Settings Toggle:**
```jsx
<AdvancedSettings>
  <Select
    label="Image Generation Model"
    options={imageModels}
    value={selectedModel}
    onChange={setSelectedModel}
    placeholder="Use default (imagen-4.0-fast)"
  />
</AdvancedSettings>
```

---

## Response Format

Both endpoints return the same response structure.

### Success Response (200 OK)

```typescript
{
  pipeline_id: string;                              // Unique identifier for this generation
  subject: string;                                  // Original subject/topic
  generated_at: string;                             // ISO 8601 timestamp
  posts: Array<{
    platform: string;                               // Platform name
    text: string;                                   // Generated text content
    media_path?: string;                            // Local file path (if media generated)
    media_type?: string;                            // "image" | "video"
    language: string;                               // Content language
  }>;
  cloud_uploads: Array<{                            // If upload_to_cloud = true
    platform: string;
    text_url?: string;                              // Public URL to text file
    media_url?: string;                             // Public URL to media file
    media_type?: string;                            // "image" | "video"
  }>;
  summary_json_url?: string;                        // Public URL to complete summary JSON
}
```

### Example Success Response

```json
{
  "pipeline_id": "quantum_communicator_20251030_143022",
  "subject": "Announcing our new Quantum Entanglement Communicator!",
  "generated_at": "2025-10-30T14:30:22.123456Z",
  "posts": [
    {
      "platform": "linkedin",
      "text": "🚀 Exciting news! We're proud to announce our revolutionary Quantum Entanglement Communicator...",
      "media_path": "output_data_local/linkedin/quantum_communicator_20251030_143022.png",
      "media_type": "image",
      "language": "English"
    },
    {
      "platform": "twitter",
      "text": "🚀 Big news! Introducing our Quantum Entanglement Communicator - instant communication across any distance! #QuantumTech #Innovation",
      "language": "English"
    }
  ],
  "cloud_uploads": [
    {
      "platform": "linkedin",
      "text_url": "https://storage.googleapis.com/your-bucket/linkedin_text_12345.txt",
      "media_url": "https://storage.googleapis.com/your-bucket/linkedin_image_12345.png",
      "media_type": "image"
    },
    {
      "platform": "twitter",
      "text_url": "https://storage.googleapis.com/your-bucket/twitter_text_12345.txt"
    }
  ],
  "summary_json_url": "https://storage.googleapis.com/your-bucket/summary_12345.json"
}
```

---

## Error Handling

### Error Response Format

```typescript
{
  detail: string;                                   // Error description
}
```

### HTTP Status Codes

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid request data or validation error |
| 500 | Internal Server Error | Pipeline processing failed |

### Common Error Scenarios

#### 1. No Platforms Selected (400)

```json
{
  "error": "No platforms selected for content generation.",
  "pipeline_id": "error_20251030_143022",
  "subject": "Your topic",
  "generated_at": "2025-10-30T14:30:22Z",
  "posts": [],
  "cloud_uploads": []
}
```

#### 2. Invalid Request Data (422)

```json
{
  "detail": [
    {
      "loc": ["body", "company", "name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 3. Pipeline Processing Failed (500)

```json
{
  "detail": "Pipeline processing failed: [error details]"
}
```

---

## Examples

### TypeScript Integration

```typescript
interface GeneratePostsRequest {
  company_name: string;
  company_mission: string;
  company_sentiment: string;
  subject: string;
  language?: string;
  tone?: string;
  platforms_post_types_map: Array<{ [key: string]: string }>;
  upload_to_cloud?: boolean;
  image_generation_model?: string;
}

interface GeneratePostsResponse {
  pipeline_id: string;
  subject: string;
  generated_at: string;
  posts: Array<{
    platform: string;
    text: string;
    media_path?: string;
    media_type?: string;
    language: string;
  }>;
  cloud_uploads?: Array<{
    platform: string;
    text_url?: string;
    media_url?: string;
    media_type?: string;
  }>;
  summary_json_url?: string;
}

async function generatePosts(
  requestData: GeneratePostsRequest
): Promise<GeneratePostsResponse> {
  const response = await fetch('http://localhost:8000/generate-posts', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate posts');
  }

  return await response.json();
}

// Usage
const result = await generatePosts({
  company_name: "Tech Innovators Inc.",
  company_mission: "To boldly innovate where no tech has innovated before.",
  company_sentiment: "Futuristic and bold, slightly playful.",
  subject: "Announcing our new Quantum Entanglement Communicator!",
  language: "English",
  tone: "Excited and Awe-Inspiring",
  platforms_post_types_map: [
    { linkedin: "Image" },
    { twitter: "Text" }
  ],
  upload_to_cloud: true,
  image_generation_model: "imagen-4.0-fast-generate-001"
});

console.log(`Generated ${result.posts.length} posts`);
console.log(`Pipeline ID: ${result.pipeline_id}`);
```

### React Hook Example

```typescript
import { useState } from 'react';

interface UseGeneratePostsOptions {
  onSuccess?: (data: GeneratePostsResponse) => void;
  onError?: (error: Error) => void;
}

function useGeneratePosts(options?: UseGeneratePostsOptions) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [data, setData] = useState<GeneratePostsResponse | null>(null);

  const generatePosts = async (requestData: GeneratePostsRequest) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/generate-posts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate posts');
      }

      const result = await response.json();
      setData(result);
      options?.onSuccess?.(result);
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      options?.onError?.(error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return { generatePosts, loading, error, data };
}

// Usage in component
function ContentGenerator() {
  const { generatePosts, loading, error, data } = useGeneratePosts({
    onSuccess: (data) => console.log('Success!', data),
    onError: (error) => console.error('Error:', error)
  });

  const handleSubmit = async () => {
    await generatePosts({
      company_name: "Example Corp",
      company_mission: "Making the world better",
      company_sentiment: "Inspirational",
      subject: "New product launch",
      platforms_post_types_map: [{ instagram: "Image" }],
      image_generation_model: "imagen-4.0-fast-generate-001"
    });
  };

  return (
    <div>
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Posts'}
      </button>
      {error && <p>Error: {error.message}</p>}
      {data && <p>Generated {data.posts.length} posts!</p>}
    </div>
  );
}
```

---

## Important Notes

### 1. Processing Time
- Post generation typically takes **30-90 seconds** depending on:
  - Number of platforms
  - Media generation requirements
  - Translation needs
  - Selected image generation model (faster models = quicker results)
- Consider showing a loading indicator with estimated time

### 2. Image Generation
- Images are generated using **Google Imagen** AI models
- Default model: `imagen-4.0-fast-generate-001`
- The system automatically:
  - Sanitizes prompts to prevent text overlays
  - Applies brand colors naturally
  - Respects aspect ratios per platform
  - Handles style and guidance directives

### 3. Cloud Storage
- When `upload_to_cloud: true`, files are uploaded to **Google Cloud Storage**
- URLs are publicly accessible
- Files persist in cloud storage
- Local files are also saved in the `output_data_local` directory

### 4. Language Support
- Primary language: English
- Translation layer activated when `language` is not "English" or "en"
- Translations are culturally adapted, not literal
- Hashtags and formatting are localized

### 5. Platform-Specific Behavior
- Each platform has unique character limits, tone, and hashtag conventions
- The AI adapts content automatically for each platform
- LinkedIn: Professional, 3-5 hashtags
- Instagram: Engaging, 5-10 hashtags
- Twitter: Concise (280 chars), 1-3 hashtags
- Facebook: Friendly, 1-3 hashtags

### 6. Image Control Best Practices
- Use **Level 1** for consistent brand styling across all platforms
- Use **Level 2** to optimize for platform-specific requirements
- Keep style descriptions clear and concise
- Avoid technical terms in guidance (the system handles that)
- Colors are automatically extracted from hex codes and applied naturally

### 7. Model Selection Strategy
- **Default model** (`imagen-4.0-fast-generate-001`) is recommended for most use cases
- Use **premium models** for:
  - High-profile campaigns
  - Brand launch materials
  - Marketing materials requiring exceptional quality
- Use **basic model** (`imagen-3.0-generate-002`) for:
  - Testing and development
  - High-volume generation
  - Budget-constrained projects

---

## Support & Contact

For technical support or questions about the API:
- **GitHub Issues:** [Report issues here](https://github.com/your-repo/issues)
- **Documentation:** Check `CLAUDE.md` in the repository root
- **Email:** support@creators-multiverse.com

---

**Last Updated:** 2025-10-30
**API Version:** 1.0.0
**Documentation Version:** 1.1.0
