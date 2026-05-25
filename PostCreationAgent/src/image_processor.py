import colorsys
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from rich.console import Console
from src import config


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _pick_accent_color(palette: list[str]) -> tuple[int, int, int] | None:
    """Pick the most saturated, mid-brightness color from the palette.

    KMeans sorts palette by frequency, so palette[0] is usually the background.
    A flat blend toward the background washes every output toward beige/cream.
    Instead we want the actual brand accent — high saturation, not too dark/light.
    """
    best = None
    best_score = -1.0
    for hex_color in palette:
        try:
            r, g, b = _hex_to_rgb(hex_color)
        except (ValueError, IndexError):
            continue
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        # penalize extremes of lightness so we don't pick a saturated near-black
        lightness_penalty = abs(l - 0.5) * 0.6
        score = s - lightness_penalty
        if score > best_score:
            best_score = score
            best = (r, g, b)
    return best

console = Console()


class ImageProcessor:
    def resize_for_instagram(self, image: Image.Image, post_type: str = "feed") -> Image.Image:
        target = config.INSTAGRAM_SIZES.get(post_type, config.INSTAGRAM_SIZES["feed"])
        return self._smart_resize(image, target)

    def _smart_resize(self, image: Image.Image, target: tuple) -> Image.Image:
        tw, th = target
        target_ratio = tw / th
        iw, ih = image.size
        image_ratio = iw / ih

        if abs(image_ratio - target_ratio) < 0.05:
            return image.resize(target, Image.LANCZOS)

        if image_ratio > target_ratio:
            new_h = ih
            new_w = int(ih * target_ratio)
            left = (iw - new_w) // 2
            image = image.crop((left, 0, left + new_w, ih))
        else:
            new_w = iw
            new_h = int(iw / target_ratio)
            top = (ih - new_h) // 2
            image = image.crop((0, top, iw, top + new_h))

        return image.resize(target, Image.LANCZOS)

    def add_text_overlay(
        self,
        image: Image.Image,
        text: str,
        position: str = "bottom",
        font_size: int = 48,
        color: str = "white",
        bg_opacity: int = 160,
    ) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        w, h = img.size
        padding = 20

        if position == "bottom":
            x = (w - text_w) // 2
            y = h - text_h - padding * 3
        elif position == "top":
            x = (w - text_w) // 2
            y = padding * 2
        elif position == "center":
            x = (w - text_w) // 2
            y = (h - text_h) // 2
        else:
            x = (w - text_w) // 2
            y = h - text_h - padding * 3

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
            fill=(0, 0, 0, bg_opacity),
        )
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, fill=color, font=font)

        return img

    def apply_color_filter(self, image: Image.Image, color_palette: list[str], strength: float = 0.08) -> Image.Image:
        if not color_palette:
            return image

        accent = _pick_accent_color(color_palette)
        if accent is None:
            return image

        overlay = Image.new("RGB", image.size, accent)
        return Image.blend(image, overlay, strength)

    def enhance_image(self, image: Image.Image, sharpness: float = 1.2, contrast: float = 1.1, saturation: float = 1.1) -> Image.Image:
        img = ImageEnhance.Sharpness(image).enhance(sharpness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img

    def save_image(self, image: Image.Image, filename: str | None = None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"post_{timestamp}.jpg"

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filename += ".jpg"

        output_path = config.OUTPUT_DIR / filename
        config.OUTPUT_DIR.mkdir(exist_ok=True)

        if filename.lower().endswith(".png"):
            image.save(output_path, "PNG", optimize=True)
        else:
            image.save(output_path, "JPEG", quality=95, optimize=True)

        console.print(f"  [dim]Saved: {output_path}[/]")
        return output_path

    def save_batch(self, images: list[Image.Image], prefix: str = "post") -> list[Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = []
        for i, img in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i + 1}.jpg"
            path = self.save_image(img, filename)
            paths.append(path)
        return paths
