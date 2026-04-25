# Software Requirements Specification (SRS)
# Instagram Post Creation Agent

## 1. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                      │
│              (main.py / Streamlit UI)                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Agent Core                          │
│            (PostCreationAgent)                       │
│                                                      │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ Style       │ │ Prompt       │ │ Post         │ │
│  │ Analyzer    │ │ Engineer     │ │ Composer     │ │
│  └─────────────┘ └──────────────┘ └──────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Image Generation Layer                  │
│                                                      │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────┐│
│  │Pollinat- │ │HuggingFace│ │Stability │ │Togeth-││
│  │ions.ai   │ │Inference  │ │AI        │ │er AI  ││
│  └──────────┘ └───────────┘ └──────────┘ └───────┘│
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Image Processing Layer                  │
│         (Pillow - resize, overlay, format)           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Instagram Publishing Layer              │
│           (Instagram Graph API via requests)         │
└─────────────────────────────────────────────────────┘
```

## 2. Module Specifications

### 2.1 Project Structure

```
PostCreationAgent/
├── ideas/                    # Reference images (existing)
├── docs/
│   ├── PRD.md
│   └── SRS.md
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point / CLI
│   ├── agent.py              # Core agent orchestrator
│   ├── style_analyzer.py     # Analyzes reference images for brand style
│   ├── prompt_engineer.py    # Enhances user prompts with style context
│   ├── image_generator.py    # Multi-provider image generation with fallback
│   ├── image_processor.py    # Post-processing (resize, overlay, format)
│   ├── instagram_publisher.py # Instagram Graph API integration
│   └── config.py             # Configuration and API keys
├── output/                   # Generated images stored here
├── .env.example              # Template for environment variables
├── requirements.txt
└── README.md
```

### 2.2 Module: `config.py`

**Purpose**: Centralized configuration management.

| Setting | Type | Description |
|---------|------|-------------|
| `HUGGINGFACE_API_KEY` | str | HuggingFace API token (free tier) |
| `STABILITY_API_KEY` | str | Stability AI API key (free tier) |
| `TOGETHER_API_KEY` | str | Together AI API key (free tier) |
| `INSTAGRAM_ACCESS_TOKEN` | str | Instagram Graph API long-lived token |
| `INSTAGRAM_ACCOUNT_ID` | str | Instagram Business account ID |
| `OUTPUT_DIR` | str | Path to store generated images |
| `REFERENCE_DIR` | str | Path to reference/idea images |
| `DEFAULT_IMAGE_SIZE` | tuple | Default output size (1080, 1080) |
| `MAX_RETRIES` | int | Max retries per provider (default: 3) |

### 2.3 Module: `style_analyzer.py`

**Purpose**: Analyze reference images to extract brand style characteristics.

**Class**: `StyleAnalyzer`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `analyze_images(image_dir)` | str (path) | StyleProfile | Analyzes all images in directory |
| `extract_colors(image)` | PIL.Image | list[tuple] | Extracts dominant color palette |
| `extract_style_description()` | None | str | Generates text description of visual style |
| `get_style_prompt()` | None | str | Returns style string for image generation prompts |

**Data Class**: `StyleProfile`
- `dominant_colors`: list of RGB tuples
- `color_palette`: list of hex color strings
- `brightness`: float (0-1)
- `contrast`: str (low/medium/high)
- `style_keywords`: list of descriptive keywords
- `style_prompt_suffix`: str (appended to generation prompts)

### 2.4 Module: `prompt_engineer.py`

**Purpose**: Transform user prompts into optimized image generation prompts.

**Class**: `PromptEngineer`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `enhance_prompt(user_prompt, style_profile, post_type)` | str, StyleProfile, str | str | Creates optimized prompt |
| `generate_caption(user_prompt)` | str | str | Generates Instagram caption |
| `generate_hashtags(user_prompt)` | str | list[str] | Generates relevant hashtags |

### 2.5 Module: `image_generator.py`

**Purpose**: Generate images using multiple free-tier AI providers with automatic fallback.

**Class**: `ImageGenerator`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `generate(prompt, count, size)` | str, int, tuple | list[PIL.Image] | Generates images with auto-fallback |
| `_generate_pollinations(prompt, size)` | str, tuple | PIL.Image | Pollinations.ai generation |
| `_generate_huggingface(prompt, size)` | str, tuple | PIL.Image | HuggingFace Inference API |
| `_generate_stability(prompt, size)` | str, tuple | PIL.Image | Stability AI generation |
| `_generate_together(prompt, size)` | str, tuple | PIL.Image | Together AI generation |

**Fallback Logic**:
```
for each image needed:
    for each provider in priority_order:
        try:
            image = provider.generate(prompt, size)
            if image is valid:
                return image
        except (RateLimitError, APIError):
            log warning, continue to next provider
    raise AllProvidersExhaustedError
```

### 2.6 Module: `image_processor.py`

**Purpose**: Post-process generated images for Instagram compliance.

**Class**: `ImageProcessor`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `resize_for_instagram(image, post_type)` | PIL.Image, str | PIL.Image | Resize to IG dimensions |
| `add_text_overlay(image, text, position, style)` | PIL.Image, str, ... | PIL.Image | Add text to image |
| `add_brand_watermark(image, logo_path)` | PIL.Image, str | PIL.Image | Add brand watermark |
| `save_image(image, filename)` | PIL.Image, str | str | Save to output dir, return path |
| `apply_color_filter(image, color_palette)` | PIL.Image, list | PIL.Image | Apply brand color grading |

**Instagram Dimensions**:
- Feed Post: 1080 x 1080 (1:1)
- Portrait: 1080 x 1350 (4:5)
- Story: 1080 x 1920 (9:16)
- Landscape: 1080 x 608 (1.91:1)

### 2.7 Module: `instagram_publisher.py`

**Purpose**: Publish images to Instagram via the Graph API.

**Class**: `InstagramPublisher`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `publish_single(image_path, caption)` | str, str | dict | Post single image |
| `publish_carousel(image_paths, caption)` | list, str | dict | Post carousel |
| `upload_image(image_path)` | str | str | Upload and get container ID |
| `verify_connection()` | None | bool | Test API credentials |

### 2.8 Module: `agent.py`

**Purpose**: Orchestrates the entire workflow.

**Class**: `PostCreationAgent`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `create_post(prompt, count, post_type)` | str, int, str | list[str] | Full pipeline execution |
| `preview_post(prompt, count, post_type)` | str, int, str | list[PIL.Image] | Generate without posting |
| `analyze_brand()` | None | StyleProfile | Analyze reference images |
| `publish(image_paths, caption)` | list, str | dict | Publish to Instagram |

### 2.9 Module: `main.py`

**Purpose**: CLI entry point with interactive menu.

**Commands**:
- `create` — Create and optionally publish a post
- `preview` — Generate images without publishing
- `analyze` — Analyze brand style from reference images
- `publish` — Publish previously generated images
- `config` — View/update configuration

## 3. External API Specifications

### 3.1 Pollinations.ai (Primary - No API Key)
- **Endpoint**: `https://image.pollinations.ai/prompt/{encoded_prompt}`
- **Method**: GET
- **Rate Limit**: Generous, no hard limit
- **Auth**: None required
- **Parameters**: width, height, seed, model, nologo

### 3.2 HuggingFace Inference API (Secondary)
- **Endpoint**: `https://api-inference.huggingface.co/models/{model_id}`
- **Method**: POST
- **Rate Limit**: ~1000 requests/day (free tier)
- **Auth**: Bearer token
- **Models**: `black-forest-labs/FLUX.1-schnell`, `stabilityai/stable-diffusion-xl-base-1.0`

### 3.3 Stability AI (Tertiary)
- **Endpoint**: `https://api.stability.ai/v2beta/stable-image/generate/core`
- **Method**: POST
- **Rate Limit**: 25 credits free
- **Auth**: Bearer token

### 3.4 Together AI (Quaternary)
- **Endpoint**: `https://api.together.xyz/v1/images/generations`
- **Method**: POST
- **Rate Limit**: $5 free credits
- **Auth**: Bearer token
- **Models**: `black-forest-labs/FLUX.1-schnell-Free`

### 3.5 Instagram Graph API
- **Base URL**: `https://graph.instagram.com/v21.0`
- **Auth**: OAuth 2.0 long-lived access token
- **Requirements**: Facebook Business account, Instagram Professional account, Meta App

## 4. Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| Language | Python 3.10+ |
| Response Time | < 60s per image generation |
| Storage | Generated images saved locally before upload |
| Error Handling | Graceful fallback between providers |
| Logging | Full pipeline logging with timestamps |
| Security | API keys stored in .env, never hardcoded |
| Image Quality | Minimum 1080px on shortest side |

## 5. Dependencies

```
requests>=2.31.0
Pillow>=10.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0     # For color clustering in style analysis
rich>=13.0.0            # For CLI formatting
```

## 6. Error Handling Strategy

| Error | Handler |
|-------|---------|
| API rate limit exceeded | Switch to next provider |
| All providers exhausted | Notify user, suggest retry later |
| Invalid image generated | Retry with modified prompt |
| Instagram API error | Log error, save image locally |
| Network timeout | Retry with exponential backoff |
| Invalid credentials | Clear error message with setup instructions |
