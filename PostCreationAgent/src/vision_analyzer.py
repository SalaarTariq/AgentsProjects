from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

from src import config

logger = logging.getLogger(__name__)
console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ANALYSIS_PROMPT = """\
You are a senior visual designer analyzing a social media post image.

Study this image carefully and return a JSON object with these exact keys:

{
  "layout_type": "<one of: centered, grid, asymmetric, split, full-bleed, text-only, collage, framed>",
  "typography_style": "<one of: sans-serif, serif, script, display, handwritten, mixed, none>",
  "typography_weight": "<one of: light, regular, bold, heavy, mixed, none>",
  "composition_tags": ["<list of 3-6 tags, e.g. 'flat-lay', 'portrait', 'text-heavy', 'photo-dominant', 'minimalist', 'layered', 'gradient-bg', 'product-shot'>"],
  "mood_keywords": ["<list of 3-5 mood descriptors, e.g. 'energetic', 'calm', 'luxurious', 'playful', 'corporate'>"],
  "color_palette_description": "<short natural-language sentence about the color relationships, e.g. 'Warm earth tones with a teal accent against cream backgrounds'>",
  "text_placement": "<one of: top, center, bottom, left, right, overlay, scattered, none>",
  "visual_hierarchy": "<one of: image-first, text-first, balanced, icon-driven>",
  "brand_elements": ["<list of 0-4 recurring brand elements, e.g. 'logo watermark', 'consistent border', 'signature color bar', 'rounded corners'>"]
}

Rules:
- Return ONLY valid JSON. No markdown fences, no explanation.
- Use the exact keys above; do not add or rename keys.
- If the image has no text, set typography_style and typography_weight to "none" and text_placement to "none".
- Be specific and observational, not generic.
"""

VALID_KEYS = {
    "layout_type",
    "typography_style",
    "typography_weight",
    "composition_tags",
    "mood_keywords",
    "color_palette_description",
    "text_placement",
    "visual_hierarchy",
    "brand_elements",
}


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.name.encode())
    h.update(str(path.stat().st_size).encode())
    h.update(str(int(path.stat().st_mtime)).encode())
    return h.hexdigest()


def _most_common(values: list[str], default: str = "unknown") -> str:
    if not values:
        return default
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _flatten_union(lists: list[list[str]]) -> list[str]:
    """Deduplicated union preserving frequency order."""
    counter: Counter[str] = Counter()
    for lst in lists:
        counter.update(lst)
    return [item for item, _ in counter.most_common()]


class VisionStyleAnalyzer:
    """Analyzes reference social media images using a vision-language model
    to extract structured style descriptors (layout, typography, mood, etc.).

    Results are cached to disk so repeated runs with the same images are instant.
    """

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        model_name: str | None = None,
        cache_dir: str | Path | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.reference_dir / ".style_cache"
        self.model = ChatGoogleGenerativeAI(
            model=model_name or config.LLM_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_all(self, max_images: int = 15) -> dict[str, Any]:
        """Analyze reference images and return an aggregated style descriptor.

        Returns a dict with the same keys as the per-image analysis plus
        a ``per_image`` list containing individual results and a
        ``style_prompt_suffix`` string for direct injection into generation prompts.
        """
        image_paths = self._get_image_paths()[:max_images]

        if not image_paths:
            console.print("[yellow]No reference images found for vision analysis.[/]")
            return self._default_result()

        cached = self._load_cache(image_paths)
        if cached is not None:
            console.print("[green]Vision style cache hit — skipping re-analysis.[/]")
            return cached

        console.print(
            f"[bold cyan]Analyzing {len(image_paths)} reference image(s) with vision model...[/]"
        )

        per_image: list[dict[str, Any]] = []
        for i, path in enumerate(image_paths):
            console.print(f"  [{i + 1}/{len(image_paths)}] {path.name}")
            result = self._analyze_single(path)
            if result is not None:
                per_image.append(result)

        if not per_image:
            console.print("[yellow]All image analyses failed. Returning defaults.[/]")
            return self._default_result()

        aggregated = self._aggregate(per_image)
        self._save_cache(image_paths, aggregated)

        console.print(f"[green]Vision analysis complete — {len(per_image)} image(s) processed.[/]")
        logger.info("Aggregated style: %s", json.dumps(aggregated, indent=2))
        return aggregated

    # ------------------------------------------------------------------
    # Single-image analysis
    # ------------------------------------------------------------------

    def _analyze_single(self, image_path: Path) -> dict[str, Any] | None:
        try:
            raw_bytes = image_path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", image_path, exc)
            console.print(f"    [dim]Skipped (read error): {exc}[/]")
            return None

        suffix = image_path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        b64 = base64.b64encode(raw_bytes).decode()

        message = HumanMessage(
            content=[
                {"type": "text", "text": ANALYSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ]
        )

        try:
            response = self.model.invoke([message])
        except Exception as exc:
            logger.warning("VLM call failed for %s: %s", image_path.name, exc)
            console.print(f"    [yellow]VLM error: {exc}[/]")
            return None

        return self._parse_response(response.content, image_path.name)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str, source: str = "") -> dict[str, Any] | None:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed for %s: %s — raw: %s", source, exc, text[:300])
            console.print(f"    [yellow]Invalid JSON from model (skipped)[/]")
            return None

        if not isinstance(data, dict):
            logger.warning("Expected dict, got %s for %s", type(data).__name__, source)
            return None

        missing = VALID_KEYS - data.keys()
        if missing:
            logger.debug("Response for %s missing keys: %s", source, missing)

        for key in ("composition_tags", "mood_keywords", "brand_elements"):
            if key in data and not isinstance(data[key], list):
                data[key] = [str(data[key])]

        for key in ("layout_type", "typography_style", "typography_weight",
                     "text_placement", "visual_hierarchy", "color_palette_description"):
            if key in data and not isinstance(data[key], str):
                data[key] = str(data[key])

        return data

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        aggregated: dict[str, Any] = {
            "layout_type": _most_common(
                [r["layout_type"] for r in results if r.get("layout_type")],
                default="centered",
            ),
            "typography_style": _most_common(
                [r["typography_style"] for r in results if r.get("typography_style")],
                default="sans-serif",
            ),
            "typography_weight": _most_common(
                [r["typography_weight"] for r in results if r.get("typography_weight")],
                default="regular",
            ),
            "composition_tags": _flatten_union(
                [r.get("composition_tags", []) for r in results]
            ),
            "mood_keywords": _flatten_union(
                [r.get("mood_keywords", []) for r in results]
            ),
            "color_palette_description": _most_common(
                [r["color_palette_description"] for r in results if r.get("color_palette_description")],
                default="neutral tones",
            ),
            "text_placement": _most_common(
                [r["text_placement"] for r in results if r.get("text_placement")],
                default="center",
            ),
            "visual_hierarchy": _most_common(
                [r["visual_hierarchy"] for r in results if r.get("visual_hierarchy")],
                default="balanced",
            ),
            "brand_elements": _flatten_union(
                [r.get("brand_elements", []) for r in results]
            ),
            "per_image": results,
            "image_count": len(results),
        }

        aggregated["style_prompt_suffix"] = self._build_style_prompt(aggregated)
        return aggregated

    @staticmethod
    def _build_style_prompt(agg: dict[str, Any]) -> str:
        parts = [
            f"Layout: {agg['layout_type']}",
            f"Typography: {agg['typography_style']} {agg['typography_weight']}",
            f"Mood: {', '.join(agg['mood_keywords'][:5])}",
            f"Composition: {', '.join(agg['composition_tags'][:5])}",
            f"Colors: {agg['color_palette_description']}",
            f"Text placement: {agg['text_placement']}",
            f"Visual hierarchy: {agg['visual_hierarchy']}",
        ]
        if agg.get("brand_elements"):
            parts.append(f"Brand elements: {', '.join(agg['brand_elements'][:4])}")
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_fingerprint(self, image_paths: list[Path]) -> str:
        combined = "|".join(sorted(_file_hash(p) for p in image_paths))
        return hashlib.sha256(combined.encode()).hexdigest()

    def _cache_path(self) -> Path:
        return self.cache_dir / "vision_style_cache.json"

    def _load_cache(self, image_paths: list[Path]) -> dict[str, Any] | None:
        cp = self._cache_path()
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
        except (json.JSONDecodeError, OSError):
            logger.debug("Cache file unreadable, will re-analyze.")
            return None

        if data.get("fingerprint") != self._cache_fingerprint(image_paths):
            logger.debug("Cache fingerprint mismatch, will re-analyze.")
            return None

        return data.get("result")

    def _save_cache(self, image_paths: list[Path], result: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self._cache_fingerprint(image_paths),
            "image_count": len(image_paths),
            "result": result,
        }
        try:
            self._cache_path().write_text(json.dumps(payload, indent=2))
            logger.debug("Vision style cache saved to %s", self._cache_path())
        except OSError as exc:
            logger.warning("Failed to write cache: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_image_paths(self) -> list[Path]:
        if not self.reference_dir.is_dir():
            return []
        return sorted(
            p for p in self.reference_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _default_result() -> dict[str, Any]:
        return {
            "layout_type": "centered",
            "typography_style": "sans-serif",
            "typography_weight": "bold",
            "composition_tags": ["minimalist", "clean", "professional"],
            "mood_keywords": ["modern", "professional", "clean"],
            "color_palette_description": "Neutral tones with a single brand accent color",
            "text_placement": "center",
            "visual_hierarchy": "balanced",
            "brand_elements": [],
            "per_image": [],
            "image_count": 0,
            "style_prompt_suffix": (
                "Layout: centered; Typography: sans-serif bold; "
                "Mood: modern, professional, clean; "
                "Composition: minimalist, clean, professional; "
                "Colors: Neutral tones with a single brand accent color; "
                "Text placement: center; Visual hierarchy: balanced"
            ),
        }
