from typing import TypedDict
from pathlib import Path
from src.style_analyzer import StyleProfile


class AgentState(TypedDict, total=False):
    user_prompt: str
    post_type: str
    image_count: int
    text_overlay: str
    enhance: bool

    # Minimal post design (user-provided lines)
    design_lines: list[str]
    design_tagline: str
    design_theme: str  # "light" or "dark"
    design_highlight_line: int  # index of line to accent-color, -1 for none

    style_profile: StyleProfile
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
