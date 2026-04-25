# Product Requirements Document (PRD)
# Instagram Post Creation Agent

## 1. Product Overview

An AI-powered agent that automates Instagram post creation for the company. The agent takes text prompts describing the desired post, generates branded images matching the company's visual theme, and publishes them directly to Instagram.

## 2. Problem Statement

Creating consistent, on-brand Instagram posts is time-consuming. The team needs a tool that:
- Understands the company's visual brand/theme from existing posts
- Generates professional images from text descriptions
- Supports batch generation (multiple images per request)
- Handles the full pipeline: prompt → image generation → Instagram publishing

## 3. Target Users

- Marketing team members
- Social media managers
- Company stakeholders who need quick post creation

## 4. Core Features

### 4.1 Prompt-Based Post Creation
- User provides a text prompt describing the post they want
- User specifies the number of images to generate
- Agent interprets the prompt and generates appropriate images

### 4.2 Brand Consistency (Style Learning)
- Agent analyzes existing company posts (reference images in `ideas/` folder)
- Extracts color themes, typography style, layout patterns, and visual identity
- Applies learned style to all generated images
- Maintains consistent brand aesthetics across all outputs

### 4.3 Multi-Model Image Generation with Fallback
- Uses multiple free-tier image generation APIs
- Automatic fallback: if one model's free tier is exhausted, switches to the next
- Supported providers (in priority order):
  1. **Pollinations.ai** — Completely free, no API key required
  2. **Hugging Face Inference API** — Free tier with FLUX/Stable Diffusion models
  3. **Stability AI** — Free tier available
  4. **Together AI** — Free tier with FLUX models

### 4.4 Instagram Publishing
- Direct posting to Instagram via the Instagram Graph API
- Requires Facebook Business account + Instagram Professional account
- Supports single image and carousel posts
- Auto-generates captions based on the prompt

### 4.5 Post-Ready Image Formatting
- All images formatted for Instagram (1080x1080 for feed, 1080x1920 for stories)
- Proper resolution and aspect ratio handling
- Optional text overlay on generated images

## 5. User Flow

```
1. User launches the agent (CLI or web interface)
2. User provides:
   - Text prompt describing the post
   - Number of images desired
   - Post type (feed/story/carousel)
   - Optional: specific style instructions
3. Agent generates images using AI models
4. Agent displays generated images for review
5. User approves or requests modifications
6. Agent posts approved images to Instagram
```

## 6. Technical Constraints

- Must use free-tier APIs only (no paid subscriptions required)
- Python-based implementation
- Must handle API rate limits gracefully
- Must store generated images locally before posting
- Must support Instagram's image requirements (JPEG, 1080px min)

## 7. Success Metrics

- Generated images match company brand theme (subjective review)
- End-to-end time from prompt to published post < 5 minutes
- Zero manual image editing needed for 80%+ of posts
- Successful Instagram posting without errors

## 8. Future Scope (v2+)

- Canva integration for template-based designs
- Scheduling posts for optimal engagement times
- Multi-platform support (Facebook, Twitter/X, LinkedIn)
- Analytics dashboard for post performance
- A/B testing with multiple image variants
