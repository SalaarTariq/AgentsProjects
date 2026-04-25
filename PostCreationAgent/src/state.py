from typing import TypedDict
from pathlib import Path
from src.style_analyzer import StyleProfile


class AgentState(TypedDict, total=False):
    user_prompt: str
    post_type: str
    image_count: int
    text_overlay: str
    enhance: bool

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
