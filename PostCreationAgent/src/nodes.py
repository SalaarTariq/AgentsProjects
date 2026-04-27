from rich.console import Console
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from src.style_analyzer import StyleAnalyzer
from src.deep_style_analyzer import DeepStyleAnalyzer
from src.image_generator import ImageGenerator, AllProvidersExhaustedError
from src.image_processor import ImageProcessor
from src.post_designer import PostDesigner
from src.instagram_publisher import InstagramPublisher
from src import config

console = Console()

_generator = ImageGenerator()
_processor = ImageProcessor()
_publisher = InstagramPublisher()


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.7,
    )


def analyze_style_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Analyze Brand Style[/]")

    pixel_analyzer = StyleAnalyzer(config.REFERENCE_DIR)
    pixel_profile = pixel_analyzer.analyze_images()

    result: dict = {"style_profile": pixel_profile, "status": "style_analyzed"}

    if config.DEEP_STYLE_ENABLED:
        try:
            deep_analyzer = DeepStyleAnalyzer(
                config.REFERENCE_DIR, model_name=config.CLIP_MODEL_NAME
            )
            deep_profile = deep_analyzer.analyze_all()
            result["deep_style_profile"] = deep_profile
            result["analysis_results"] = deep_profile.per_image
            result["layout_template"] = deep_profile.layout_type
            result["aggregated_style"] = {
                "layout_type": deep_profile.layout_type,
                "typography_style": deep_profile.typography_style,
                "mood_keywords": deep_profile.mood_keywords,
                "composition_tags": deep_profile.composition_tags,
                "visual_hierarchy": deep_profile.visual_hierarchy,
                "color_palette": deep_profile.color_palette,
                "color_palette_description": deep_profile.color_palette_description,
                "contrast_level": deep_profile.contrast_level,
                "style_prompt_suffix": deep_profile.style_prompt_suffix,
            }
            result["status"] = "style_analyzed_deep"
            console.print(f"  [green]Deep style: {deep_profile.layout_type} / {deep_profile.typography_style}[/]")
            console.print(f"  [green]Mood: {', '.join(deep_profile.mood_keywords[:4])}[/]")
        except Exception as exc:
            console.print(f"  [yellow]Deep analysis failed ({exc}), using pixel-only.[/]")

    return result


def _build_style_context(state: AgentState) -> str:
    """Build the richest style context available from state."""
    agg = state.get("aggregated_style")
    if agg:
        lines = [
            f"\nBrand visual identity (extracted from reference posts):",
            f"  Layout pattern: {agg.get('layout_type', 'centered')}",
            f"  Typography: {agg.get('typography_style', 'sans-serif')}",
            f"  Visual hierarchy: {agg.get('visual_hierarchy', 'balanced')}",
            f"  Mood/theme: {', '.join(agg.get('mood_keywords', [])[:5])}",
            f"  Composition style: {', '.join(agg.get('composition_tags', [])[:5])}",
            f"  Color palette: {agg.get('color_palette_description', 'neutral tones')}",
            f"  Hex colors: {', '.join(agg.get('color_palette', [])[:4])}",
            f"  Contrast: {agg.get('contrast_level', 'medium')}",
        ]
        return "\n".join(lines)

    style = state.get("style_profile")
    if style:
        return (
            f"\nBrand style: {style.style_prompt_suffix}"
            f"\nColor palette: {', '.join(style.color_palette[:4])}"
            f"\nStyle keywords: {', '.join(style.style_keywords[:5])}"
        )

    return ""


def enhance_prompt_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Enhance Prompt (LLM)[/]")

    user_prompt = state["user_prompt"]
    post_type = state.get("post_type", "feed")

    style_context = _build_style_context(state)

    dimension_map = {
        "feed": "1080x1080 square (1:1)",
        "portrait": "1080x1350 vertical (4:5)",
        "story": "1080x1920 full-screen vertical (9:16)",
        "landscape": "1080x608 wide (1.91:1)",
    }
    dimensions = dimension_map.get(post_type, dimension_map["feed"])

    llm = _get_llm()
    prompt_msg = (
        f"You are an expert social media visual designer. Generate an optimized image generation prompt "
        f"for an AI image generator (like Stable Diffusion / FLUX).\n\n"
        f"User's request: {user_prompt}\n"
        f"Image format: Instagram {post_type} post, {dimensions}\n"
        f"{style_context}\n\n"
        f"Create a detailed image generation prompt that faithfully reproduces the brand's visual "
        f"identity described above. Specify the exact layout structure, color relationships, "
        f"typography placement, mood, and composition in the prompt. "
        f"Write ONLY the image generation prompt — no explanation. "
        f"End with: high quality, professional, sharp details, 4k"
    )

    try:
        response = llm.invoke(prompt_msg)
        enhanced = response.content.strip()
        console.print(f"  [dim]{enhanced[:150]}...[/]" if len(enhanced) > 150 else f"  [dim]{enhanced}[/]")
        return {"enhanced_prompt": enhanced, "status": "prompt_enhanced"}
    except Exception as e:
        console.print(f"  [yellow]LLM unavailable ({e}), using basic enhancement[/]")

        agg = state.get("aggregated_style")
        deep = state.get("deep_style_profile")
        style = state.get("style_profile")

        suffix = ""
        if deep:
            suffix = deep.style_prompt_suffix
        elif agg:
            suffix = agg.get("style_prompt_suffix", "")
        elif style:
            suffix = style.style_prompt_suffix

        basic = (
            f"{user_prompt}, instagram {post_type} post, {dimensions}, "
            f"professional design, high quality, sharp details"
        )
        if suffix:
            basic += f", {suffix}"
        return {"enhanced_prompt": basic, "status": "prompt_enhanced_fallback"}


def generate_caption_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Generate Caption (LLM)[/]")

    user_prompt = state["user_prompt"]

    llm = _get_llm()
    caption_msg = (
        f"You are a social media copywriter. Write an engaging Instagram caption for a post about:\n"
        f"{user_prompt}\n\n"
        f"Requirements:\n"
        f"- Engaging, conversational tone\n"
        f"- Include a call to action\n"
        f"- 2-3 sentences max\n"
        f"- Then add a blank line and 10-15 relevant hashtags\n"
        f"- Make hashtags specific and trending\n\n"
        f"Write ONLY the caption and hashtags, nothing else."
    )

    try:
        response = llm.invoke(caption_msg)
        caption = response.content.strip()
        console.print(f"  [dim]{caption[:100]}...[/]" if len(caption) > 100 else f"  [dim]{caption}[/]")
        return {"caption": caption, "status": "caption_generated"}
    except Exception as e:
        console.print(f"  [yellow]LLM unavailable ({e}), using basic caption[/]")
        from src.prompt_engineer import PromptEngineer
        eng = PromptEngineer()
        return {"caption": eng.build_full_caption(user_prompt), "status": "caption_generated_fallback"}


def generate_images_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Generate Images[/]")

    prompt = state.get("enhanced_prompt", state["user_prompt"])
    count = state.get("image_count", 1)
    post_type = state.get("post_type", "feed")
    size = config.INSTAGRAM_SIZES.get(post_type, config.INSTAGRAM_SIZES["feed"])

    try:
        images = _generator.generate(prompt, count, size)
        if not images:
            return {"generated_images": [], "error": "No images generated", "status": "generation_failed"}
        console.print(f"  [green]Generated {len(images)} image(s)[/]")
        return {"generated_images": images, "status": "images_generated"}
    except AllProvidersExhaustedError:
        return {"generated_images": [], "error": "All providers exhausted", "status": "generation_failed"}


def process_images_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Process Images[/]")

    images = state.get("generated_images", [])
    if not images:
        return {"processed_images": [], "status": "processing_skipped"}

    post_type = state.get("post_type", "feed")
    text_overlay = state.get("text_overlay")
    enhance = state.get("enhance", True)
    style = state.get("style_profile")

    processed = []
    for i, img in enumerate(images):
        console.print(f"  Processing image {i + 1}/{len(images)}...")
        img = _processor.resize_for_instagram(img, post_type)

        if enhance:
            img = _processor.enhance_image(img)

        if text_overlay:
            img = _processor.add_text_overlay(img, text_overlay)

        if style and style.color_palette:
            img = _processor.apply_color_filter(img, style.color_palette, strength=0.08)

        processed.append(img)

    console.print(f"  [green]Processed {len(processed)} image(s)[/]")
    return {"processed_images": processed, "status": "images_processed"}


def save_images_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Save Images[/]")

    images = state.get("processed_images", [])
    if not images:
        return {"saved_paths": [], "status": "save_skipped"}

    paths = _processor.save_batch(images, prefix="post")

    caption = state.get("caption", "")
    if caption:
        caption_path = config.OUTPUT_DIR / "latest_caption.txt"
        caption_path.write_text(caption)

    console.print(f"  [green]Saved {len(paths)} image(s) to {config.OUTPUT_DIR}[/]")
    return {"saved_paths": paths, "status": "images_saved"}


def publish_node(state: AgentState) -> dict:
    console.print("\n[bold cyan]>> Node: Publish to Instagram[/]")

    if not state.get("publish_to_instagram"):
        console.print("  [dim]Publishing skipped (not requested)[/]")
        return {"publish_result": {"success": False, "reason": "not_requested"}, "status": "publish_skipped"}

    if not _publisher.verify_connection():
        console.print("  [yellow]Instagram not configured[/]")
        return {"publish_result": {"success": False, "reason": "not_configured"}, "status": "publish_failed"}

    paths = state.get("saved_paths", [])
    if not paths:
        return {"publish_result": {"success": False, "reason": "no_images"}, "status": "publish_failed"}

    console.print("  [yellow]Note: Instagram API requires publicly hosted image URLs.[/]")
    console.print("  [dim]Images saved locally. Upload to a public server to publish.[/]")
    return {
        "publish_result": {"success": False, "reason": "needs_public_hosting"},
        "status": "publish_needs_hosting",
    }


def design_post_node(state: AgentState) -> dict:
    """Generate a minimal branded post from user-provided text lines."""
    console.print("\n[bold cyan]>> Node: Design Minimal Post[/]")

    lines = state.get("design_lines", [])
    if not lines:
        return {"error": "No design lines provided", "status": "design_skipped"}

    tagline = state.get("design_tagline")
    theme = state.get("design_theme", "light")
    highlight = state.get("design_highlight_line", -2)  # -2 = no highlight

    designer = PostDesigner(theme=theme)

    if highlight != -2:
        path = designer.generate_accent_post(
            lines=lines,
            highlight_line=highlight,
            tagline=tagline,
        )
    else:
        path = designer.generate(
            lines=lines,
            tagline=tagline,
        )

    console.print(f"  [green]Designed post saved: {path}[/]")
    return {"saved_paths": [path], "status": "post_designed"}


def summary_node(state: AgentState) -> dict:
    divider = "=" * 50
    console.print(f"\n[bold magenta]{divider}[/]")
    console.print("[bold magenta]  Pipeline Complete[/]")
    console.print(f"[bold magenta]{divider}[/]\n")

    paths = state.get("saved_paths", [])
    caption = state.get("caption", "")

    if paths:
        console.print(f"[bold green]Generated {len(paths)} image(s):[/]")
        for p in paths:
            console.print(f"  [cyan]{p}[/]")

    if caption:
        console.print(f"\n[bold]Caption:[/]")
        console.print(f"  {caption[:200]}")

    errors = state.get("error")
    if errors:
        console.print(f"\n[yellow]Warnings: {errors}[/]")

    return {"status": "complete"}
