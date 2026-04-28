from __future__ import annotations

from typing import Any, TypedDict, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.style_analyzer import StyleProfile
    from src.deep_style_analyzer import DeepStyleProfile


class AgentState(TypedDict, total=False):
    user_prompt: str
    post_type: str
    image_count: int
    text_overlay: str
    enhance: bool

    # Minimal post design (user-provided lines)
    design_lines: list[str]
    design_tagline: str
    design_theme: str  # light, dark, sage, cream, olive, forest, warm
    design_highlight_line: int  # index of line to accent-color, -2 for none

    # Pixel-level style (legacy KMeans colors)
    style_profile: StyleProfile

    # Deep learning style analysis (CLIP-based)
    deep_style_profile: DeepStyleProfile
    aggregated_style: dict[str, Any]
    analysis_results: list[dict[str, Any]]
    layout_template: str

    enhanced_prompt: str
    caption: str
    hashtags: list[str]

    generated_images: list  # list of PIL.Image
    processed_images: list  # list of PIL.Image
    saved_paths: list[Path]

    publish_to_instagram: bool
    publish_result: dict

    error: str
    status: str
