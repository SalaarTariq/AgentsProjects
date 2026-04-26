"""Dynamic minimal-style Instagram post generator.

Accepts user-provided text lines and generates clean, branded posts
using the project's color theme. No hardcoded content.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime
from src import config

# Brand color theme
BRAND = {
    "bg": "#ECEDE6",           # Warm cream background
    "text": "#2F3029",         # Dark olive text
    "accent": "#7E8F66",       # Sage green accent
    "accent_dark": "#5F6D4C",  # Darker olive accent
    "accent_light": "#99AA7B", # Light sage accent
    "divider": "#C4C5BE",      # Muted divider
}

# Dark variant
BRAND_DARK = {
    "bg": "#0A0A0A",
    "text": "#FFFFFF",
    "accent": "#7E8F66",
    "accent_dark": "#5F6D4C",
    "accent_light": "#99AA7B",
    "divider": "#3A3A3A",
}

FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
FONT_INDEX_BOLD = 1
FONT_INDEX_LIGHT = 7
FONT_INDEX_MEDIUM = 10


THEMES = {
    "light": BRAND,
    "dark": BRAND_DARK,
}


def build_color_scheme(
    theme: str = "light",
    bg: str | None = None,
    text: str | None = None,
    accent: str | None = None,
    divider: str | None = None,
) -> dict:
    base = dict(THEMES.get(theme, BRAND))
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
        """Generate a minimal post from user-provided text lines.

        Args:
            lines: Main text lines to display.
            tagline: Optional small tagline below divider.
            font_size: Size of main text.
            tagline_size: Size of tagline text.
            line_spacing: Vertical spacing between main lines.
            uppercase: Whether to uppercase the main text.
            output_name: Optional filename (without extension).

        Returns:
            Path to the saved image.
        """
        if uppercase:
            lines = [line.upper() for line in lines]

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
            tagline_h = tagline_size
            total_height = main_block_height + divider_gap + divider_height + tagline_gap + tagline_h
        else:
            total_height = main_block_height

        start_y = (self.height - total_height) // 2

        # Draw main text
        y = start_y
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_bold)
            tw = bbox[2] - bbox[0]
            x = (self.width - tw) // 2
            draw.text((x, y), line, fill=self.colors["text"], font=font_bold)
            y += line_spacing

        # Draw divider and tagline
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

        # Save
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"post_{timestamp}"

        output_path = config.OUTPUT_DIR / f"{output_name}.png"
        img.save(output_path, "PNG")
        return output_path

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
        """Generate a post with one line highlighted in accent color.

        Args:
            lines: Main text lines.
            highlight_line: Index of the line to highlight (default: last line).
            tagline: Optional tagline below divider.
            font_size: Size of main text.
            tagline_size: Size of tagline text.
            line_spacing: Vertical spacing between main lines.
            uppercase: Whether to uppercase main text.
            output_name: Optional filename.

        Returns:
            Path to the saved image.
        """
        if uppercase:
            lines = [line.upper() for line in lines]

        if highlight_line < 0:
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
            tagline_h = tagline_size
            total_height = main_block_height + divider_gap + divider_height + tagline_gap + tagline_h
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
            output_name = f"post_accent_{timestamp}"

        output_path = config.OUTPUT_DIR / f"{output_name}.png"
        img.save(output_path, "PNG")
        return output_path
