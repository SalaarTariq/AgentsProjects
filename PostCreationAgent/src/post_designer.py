"""Dynamic minimal-style Instagram post generator.

Accepts user-provided text lines and generates clean, branded posts
using the project's color theme. No hardcoded content.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime
from src import config

# ── Color themes derived from the brand color_theme.png ──────────────
# Each theme pairs a background with complementary text/accent/divider
# colors for readability and visual harmony.

THEMES: dict[str, dict[str, str]] = {
    # Warm cream (default light)
    "light": {
        "bg": "#ECEDE6",
        "text": "#2F3029",
        "accent": "#7E8F66",
        "accent_dark": "#5F6D4C",
        "accent_light": "#99AA7B",
        "divider": "#C4C5BE",
    },
    # Deep dark
    "dark": {
        "bg": "#0A0A0A",
        "text": "#ECEDE6",
        "accent": "#99AA7B",
        "accent_dark": "#7E8F66",
        "accent_light": "#B5C49A",
        "divider": "#3A3A3A",
    },
    # Sage green background with cream text
    "sage": {
        "bg": "#6B7D54",
        "text": "#ECEDE6",
        "accent": "#2F3029",
        "accent_dark": "#1A1C16",
        "accent_light": "#C4C5BE",
        "divider": "#5F6D4C",
    },
    # Warm beige / sand
    "cream": {
        "bg": "#D7D3C6",
        "text": "#2F3029",
        "accent": "#5F6D4C",
        "accent_dark": "#3D4632",
        "accent_light": "#7E8F66",
        "divider": "#B5B0A3",
    },
    # Dark olive background with cream text
    "olive": {
        "bg": "#2F3029",
        "text": "#ECEDE6",
        "accent": "#99AA7B",
        "accent_dark": "#7E8F66",
        "accent_light": "#B5C49A",
        "divider": "#5F6D4C",
    },
    # Deep forest green
    "forest": {
        "bg": "#3D4632",
        "text": "#ECEDE6",
        "accent": "#B5C49A",
        "accent_dark": "#99AA7B",
        "accent_light": "#D7D3C6",
        "divider": "#5F6D4C",
    },
    # Warm terracotta tones
    "warm": {
        "bg": "#C9B99A",
        "text": "#2F3029",
        "accent": "#5F6D4C",
        "accent_dark": "#3D4632",
        "accent_light": "#7E8F66",
        "divider": "#A89B7E",
    },
}

FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
FONT_INDEX_BOLD = 1
FONT_INDEX_LIGHT = 7
FONT_INDEX_MEDIUM = 10


def build_color_scheme(
    theme: str = "light",
    bg: str | None = None,
    text: str | None = None,
    accent: str | None = None,
    divider: str | None = None,
) -> dict:
    base = dict(THEMES.get(theme, THEMES["light"]))
    if bg:
        base["bg"] = bg
    if text:
        base["text"] = text
    if accent:
        base["accent"] = accent
    if divider:
        base["divider"] = divider
    return base


class PostDesigner:
    """Generates minimal Instagram posts from user-provided text."""

    def __init__(
        self,
        width: int = 1080,
        height: int = 1080,
        theme: str = "light",
        bg: str | None = None,
        text: str | None = None,
        accent: str | None = None,
        divider: str | None = None,
    ):
        self.width = width
        self.height = height
        self.colors = build_color_scheme(theme, bg, text, accent, divider)

    # ── shared rendering core ────────────────────────────────────────

    def _render_post(
        self,
        lines: list[str],
        tagline: str | None = None,
        font_size: int = 70,
        tagline_size: int = 24,
        line_spacing: int = 88,
        uppercase: bool = True,
        highlight_line: int = -2,
        output_prefix: str = "post",
        output_name: str | None = None,
    ) -> Path:
        if uppercase:
            lines = [line.upper() for line in lines]

        if highlight_line != -2 and highlight_line < 0:
            highlight_line = len(lines) + highlight_line

        img = Image.new("RGB", (self.width, self.height), self.colors["bg"])
        draw = ImageDraw.Draw(img)

        font_bold = ImageFont.truetype(FONT_PATH, font_size, index=FONT_INDEX_BOLD)
        font_light = ImageFont.truetype(FONT_PATH, tagline_size, index=FONT_INDEX_LIGHT)

        main_block_height = len(lines) * line_spacing - (line_spacing - font_size)

        has_tagline = tagline is not None and tagline.strip()
        divider_gap = 45
        divider_height = 1
        tagline_gap = 20

        if has_tagline:
            total_height = main_block_height + divider_gap + divider_height + tagline_gap + tagline_size
        else:
            total_height = main_block_height

        start_y = (self.height - total_height) // 2

        y = start_y
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font_bold)
            tw = bbox[2] - bbox[0]
            x = (self.width - tw) // 2
            color = self.colors["accent"] if i == highlight_line else self.colors["text"]
            draw.text((x, y), line, fill=color, font=font_bold)
            y += line_spacing

        if has_tagline:
            divider_y = start_y + (len(lines) - 1) * line_spacing + font_size + divider_gap
            divider_w = 60
            divider_x1 = (self.width - divider_w) // 2
            divider_x2 = divider_x1 + divider_w
            draw.line(
                [(divider_x1, divider_y), (divider_x2, divider_y)],
                fill=self.colors["divider"],
                width=1,
            )

            tagline_y = divider_y + tagline_gap
            bbox = draw.textbbox((0, 0), tagline, font=font_light)
            tw = bbox[2] - bbox[0]
            tx = (self.width - tw) // 2
            draw.text((tx, tagline_y), tagline, fill=self.colors["accent"], font=font_light)

        config.OUTPUT_DIR.mkdir(exist_ok=True)
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{output_prefix}_{timestamp}"

        output_path = config.OUTPUT_DIR / f"{output_name}.png"
        img.save(output_path, "PNG")
        return output_path

    # ── public API (delegates to _render_post) ───────────────────────

    def generate(
        self,
        lines: list[str],
        tagline: str | None = None,
        font_size: int = 70,
        tagline_size: int = 24,
        line_spacing: int = 88,
        uppercase: bool = True,
        output_name: str | None = None,
    ) -> Path:
        return self._render_post(
            lines=lines,
            tagline=tagline,
            font_size=font_size,
            tagline_size=tagline_size,
            line_spacing=line_spacing,
            uppercase=uppercase,
            output_prefix="post",
            output_name=output_name,
        )

    def generate_accent_post(
        self,
        lines: list[str],
        highlight_line: int = -1,
        tagline: str | None = None,
        font_size: int = 70,
        tagline_size: int = 24,
        line_spacing: int = 88,
        uppercase: bool = True,
        output_name: str | None = None,
    ) -> Path:
        return self._render_post(
            lines=lines,
            tagline=tagline,
            font_size=font_size,
            tagline_size=tagline_size,
            line_spacing=line_spacing,
            uppercase=uppercase,
            highlight_line=highlight_line,
            output_prefix="post_accent",
            output_name=output_name,
        )
