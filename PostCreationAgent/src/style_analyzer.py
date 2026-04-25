from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from rich.console import Console

console = Console()


@dataclass
class StyleProfile:
    dominant_colors: list = field(default_factory=list)
    color_palette: list = field(default_factory=list)
    avg_brightness: float = 0.5
    contrast_level: str = "medium"
    style_keywords: list = field(default_factory=list)
    style_prompt_suffix: str = ""


class StyleAnalyzer:
    def __init__(self, reference_dir: str | Path):
        self.reference_dir = Path(reference_dir)
        self.profile = StyleProfile()

    def analyze_images(self, max_images: int = 30) -> StyleProfile:
        console.print("[bold cyan]Analyzing reference images for brand style...[/]")
        image_paths = self._get_image_paths()
        if not image_paths:
            console.print("[yellow]No reference images found. Using default style.[/]")
            return self._default_profile()

        image_paths = image_paths[:max_images]
        all_colors = []
        brightnesses = []
        contrasts = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                img_small = img.resize((150, 150))
                pixels = np.array(img_small).reshape(-1, 3)
                all_colors.append(pixels)
                brightnesses.append(np.mean(pixels) / 255.0)
                pixel_std = np.std(pixels.astype(float))
                contrasts.append(pixel_std)
                if (i + 1) % 10 == 0:
                    console.print(f"  Processed {i + 1}/{len(image_paths)} images")
            except Exception as e:
                console.print(f"  [dim]Skipping {path.name}: {e}[/]")

        if not all_colors:
            return self._default_profile()

        combined_pixels = np.vstack(all_colors)
        sample_size = min(50000, len(combined_pixels))
        indices = np.random.choice(len(combined_pixels), sample_size, replace=False)
        sampled = combined_pixels[indices]

        n_clusters = 6
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(sampled)

        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(-counts)

        dominant_colors = [tuple(int(v) for v in centers[i]) for i in sorted_indices]
        color_palette = [self._rgb_to_hex(c) for c in dominant_colors]

        avg_brightness = float(np.mean(brightnesses))
        avg_contrast = float(np.mean(contrasts))

        if avg_contrast < 40:
            contrast_level = "low"
        elif avg_contrast < 70:
            contrast_level = "medium"
        else:
            contrast_level = "high"

        style_keywords = self._derive_keywords(avg_brightness, contrast_level, dominant_colors)
        style_prompt_suffix = self._build_style_prompt(style_keywords, color_palette, avg_brightness)

        self.profile = StyleProfile(
            dominant_colors=dominant_colors,
            color_palette=color_palette,
            avg_brightness=avg_brightness,
            contrast_level=contrast_level,
            style_keywords=style_keywords,
            style_prompt_suffix=style_prompt_suffix,
        )

        console.print(f"[green]Style analysis complete![/]")
        console.print(f"  Colors: {', '.join(color_palette[:4])}")
        console.print(f"  Brightness: {avg_brightness:.2f} | Contrast: {contrast_level}")
        console.print(f"  Keywords: {', '.join(style_keywords[:5])}")

        return self.profile

    def _get_image_paths(self) -> list[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        paths = []
        for f in sorted(self.reference_dir.iterdir()):
            if f.suffix.lower() in extensions and f.is_file():
                paths.append(f)
        return paths

    def _rgb_to_hex(self, rgb: tuple) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _derive_keywords(self, brightness: float, contrast: str, colors: list) -> list:
        keywords = []

        if brightness > 0.65:
            keywords.extend(["bright", "light", "airy"])
        elif brightness < 0.35:
            keywords.extend(["dark", "moody", "dramatic"])
        else:
            keywords.extend(["balanced", "natural"])

        if contrast == "high":
            keywords.append("bold")
        elif contrast == "low":
            keywords.extend(["soft", "muted"])

        if colors:
            r, g, b = colors[0]
            if r > 180 and g < 100 and b < 100:
                keywords.append("warm-red")
            elif b > 180 and r < 100:
                keywords.append("cool-blue")
            elif g > 180 and r < 100:
                keywords.append("green-natural")
            elif r > 150 and g > 100 and b < 80:
                keywords.append("warm-golden")

        keywords.extend(["professional", "social-media", "instagram"])
        return keywords

    def _build_style_prompt(self, keywords: list, palette: list, brightness: float) -> str:
        parts = []

        tone = "bright and clean" if brightness > 0.6 else "dark and dramatic" if brightness < 0.35 else "balanced tones"
        parts.append(f"Style: {tone}")

        color_desc = f"color palette using {', '.join(palette[:3])}"
        parts.append(color_desc)

        relevant = [k for k in keywords if k not in ("professional", "social-media", "instagram")]
        if relevant:
            parts.append(f"aesthetic: {', '.join(relevant[:4])}")

        parts.append("professional social media design, high quality, clean layout")
        return ", ".join(parts)

    def _default_profile(self) -> StyleProfile:
        return StyleProfile(
            dominant_colors=[(41, 41, 41), (255, 255, 255), (0, 122, 255)],
            color_palette=["#292929", "#ffffff", "#007aff"],
            avg_brightness=0.5,
            contrast_level="medium",
            style_keywords=["modern", "clean", "professional", "social-media"],
            style_prompt_suffix="modern clean professional design, balanced tones, high quality social media post",
        )
