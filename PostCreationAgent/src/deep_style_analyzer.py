"""Local deep-learning style analyzer using CLIP zero-shot classification.

Runs entirely offline after the initial model download — no per-request API
costs.  Extracts layout type, typography style, mood/theme, composition tags,
and color palette from reference social-media post images.

Requires: torch, transformers, Pillow, numpy, scikit-learn
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.console import Console
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── CLIP candidate labels ────────────────────────────────────────────

LAYOUT_LABELS = [
    "centered layout",
    "grid layout",
    "asymmetric layout",
    "split layout",
    "full-bleed layout",
    "text-only post",
    "collage layout",
    "framed layout",
]

TYPOGRAPHY_LABELS = [
    "serif typography",
    "sans-serif typography",
    "script typography",
    "bold sans-serif typography",
    "display typography",
    "handwritten typography",
    "no visible text",
]

MOOD_LABELS = [
    "professional",
    "playful",
    "luxurious",
    "minimal",
    "earthy",
    "vibrant",
    "elegant",
    "bold",
    "calm",
    "energetic",
]

COMPOSITION_LABELS = [
    "flat-lay composition",
    "rule-of-thirds composition",
    "negative space composition",
    "symmetrical composition",
    "diagonal composition",
    "layered composition",
    "product-shot composition",
    "portrait composition",
    "text-heavy design",
    "photo-dominant design",
]

HIERARCHY_LABELS = [
    "image-first visual hierarchy",
    "text-first visual hierarchy",
    "balanced visual hierarchy",
    "icon-driven visual hierarchy",
]


@dataclass
class DeepStyleProfile:
    layout_type: str = "centered"
    typography_style: str = "sans-serif"
    mood_keywords: list[str] = field(default_factory=list)
    composition_tags: list[str] = field(default_factory=list)
    visual_hierarchy: str = "balanced"
    dominant_colors: list[tuple[int, int, int]] = field(default_factory=list)
    color_palette: list[str] = field(default_factory=list)
    color_palette_description: str = ""
    avg_brightness: float = 0.5
    contrast_level: str = "medium"
    style_embedding: list[float] = field(default_factory=list)
    style_prompt_suffix: str = ""
    per_image: list[dict[str, Any]] = field(default_factory=list)
    image_count: int = 0


# ── Helpers ───────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.name.encode())
    h.update(str(path.stat().st_size).encode())
    h.update(str(int(path.stat().st_mtime)).encode())
    return h.hexdigest()


def _most_common(values: list[str], default: str = "unknown") -> str:
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _flatten_union(lists: list[list[str]]) -> list[str]:
    counter: Counter[str] = Counter()
    for lst in lists:
        counter.update(lst)
    return [item for item, _ in counter.most_common()]


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _describe_color_palette(
    colors: list[tuple[int, int, int]], brightness: float
) -> str:
    if not colors:
        return "neutral tones"

    def _hue_name(r: int, g: int, b: int) -> str:
        if max(r, g, b) - min(r, g, b) < 30:
            return "gray" if r < 200 else "white" if r > 230 else "neutral"
        if r >= g and r >= b:
            return "warm red" if g < 100 else "warm golden"
        if g >= r and g >= b:
            return "green" if r < 100 else "olive"
        return "cool blue" if r < 100 else "purple"

    hue_names = [_hue_name(*c) for c in colors[:4]]
    unique = list(dict.fromkeys(hue_names))
    tone = "bright" if brightness > 0.6 else "dark" if brightness < 0.35 else "balanced"
    return f"{tone} palette with {', '.join(unique[:3])} tones"


# ── Main class ────────────────────────────────────────────────────────

class DeepStyleAnalyzer:
    """CLIP-based zero-shot style analyzer that runs locally.

    Loads ``openai/clip-vit-base-patch32`` from HuggingFace Transformers on
    first use.  Supports GPU when available, falls back to CPU.
    """

    _model = None
    _processor = None
    _device = None

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.model_name = model_name
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.reference_dir / ".deep_cache"
        )

    # ── Model loading (singleton, lazy) ──────────────────────────────

    def _ensure_model(self) -> None:
        if DeepStyleAnalyzer._model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[bold cyan]Loading CLIP model on {device}...[/]")
        logger.info("Loading %s on %s", self.model_name, device)

        DeepStyleAnalyzer._processor = CLIPProcessor.from_pretrained(self.model_name)
        DeepStyleAnalyzer._model = CLIPModel.from_pretrained(self.model_name).to(device)
        DeepStyleAnalyzer._model.eval()
        DeepStyleAnalyzer._device = device

        console.print("[green]CLIP model loaded.[/]")

    # ── Public API ───────────────────────────────────────────────────

    def analyze_all(self, max_images: int = 15) -> DeepStyleProfile:
        image_paths = self._get_image_paths()[:max_images]

        if not image_paths:
            console.print("[yellow]No reference images found for deep analysis.[/]")
            return self._default_profile()

        cached = self._load_cache(image_paths)
        if cached is not None:
            console.print("[green]Deep style cache hit — skipping re-analysis.[/]")
            return cached

        self._ensure_model()

        console.print(
            f"[bold cyan]Deep-analyzing {len(image_paths)} reference image(s)...[/]"
        )

        per_image: list[dict[str, Any]] = []
        embeddings: list[np.ndarray] = []

        for i, path in enumerate(image_paths):
            console.print(f"  [{i + 1}/{len(image_paths)}] {path.name}")
            result = self._analyze_single(path)
            if result is not None:
                per_image.append(result["labels"])
                embeddings.append(result["embedding"])

        if not per_image:
            console.print("[yellow]All deep analyses failed. Returning defaults.[/]")
            return self._default_profile()

        profile = self._aggregate(per_image, embeddings, image_paths)
        self._save_cache(image_paths, profile)

        console.print(
            f"[green]Deep analysis complete — {len(per_image)} image(s) processed.[/]"
        )
        return profile

    # ── Single image ─────────────────────────────────────────────────

    def _analyze_single(self, image_path: Path) -> dict[str, Any] | None:
        import torch

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open %s: %s", image_path, exc)
            console.print(f"    [dim]Skipped (read error): {exc}[/]")
            return None

        model = DeepStyleAnalyzer._model
        processor = DeepStyleAnalyzer._processor
        device = DeepStyleAnalyzer._device

        try:
            labels: dict[str, Any] = {}

            labels["layout_type"] = self._zero_shot(img, LAYOUT_LABELS)
            labels["typography_style"] = self._zero_shot(img, TYPOGRAPHY_LABELS)
            labels["visual_hierarchy"] = self._zero_shot(img, HIERARCHY_LABELS)

            labels["mood_keywords"] = self._zero_shot_topk(
                img, MOOD_LABELS, k=3, threshold=0.08
            )
            labels["composition_tags"] = self._zero_shot_topk(
                img, COMPOSITION_LABELS, k=3, threshold=0.08
            )

            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
            emb_np = embedding.cpu().numpy().flatten()
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-9)

            return {"labels": labels, "embedding": emb_np}

        except Exception as exc:
            logger.warning("CLIP inference failed for %s: %s", image_path.name, exc)
            console.print(f"    [yellow]Inference error: {exc}[/]")
            return None

    def _zero_shot(self, img: Image.Image, candidates: list[str]) -> str:
        """Return the single best-matching label."""
        import torch

        processor = DeepStyleAnalyzer._processor
        model = DeepStyleAnalyzer._model
        device = DeepStyleAnalyzer._device

        text_prompts = [f"a social media post with {c}" for c in candidates]
        inputs = processor(
            text=text_prompts, images=img, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().numpy()
        best = int(np.argmax(probs))
        return candidates[best]

    def _zero_shot_topk(
        self,
        img: Image.Image,
        candidates: list[str],
        k: int = 3,
        threshold: float = 0.08,
    ) -> list[str]:
        """Return top-k labels that exceed the probability threshold."""
        import torch

        processor = DeepStyleAnalyzer._processor
        model = DeepStyleAnalyzer._model
        device = DeepStyleAnalyzer._device

        text_prompts = [f"a social media post with {c}" for c in candidates]
        inputs = processor(
            text=text_prompts, images=img, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=-1).cpu().numpy()

        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        return [candidates[i] for i, p in ranked[:k] if p >= threshold]

    # ── Color extraction (pixel-level, reusing KMeans approach) ──────

    def _extract_colors(
        self, image_paths: list[Path], n_clusters: int = 6
    ) -> tuple[
        list[tuple[int, int, int]], list[str], float, str
    ]:
        all_pixels: list[np.ndarray] = []
        brightnesses: list[float] = []
        contrasts: list[float] = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                small = img.resize((150, 150))
                pixels = np.array(small).reshape(-1, 3)
                all_pixels.append(pixels)
                brightnesses.append(float(np.mean(pixels) / 255.0))
                contrasts.append(float(np.std(pixels.astype(float))))
            except Exception:
                continue

        if not all_pixels:
            return (
                [(41, 41, 41), (255, 255, 255), (0, 122, 255)],
                ["#292929", "#ffffff", "#007aff"],
                0.5,
                "medium",
            )

        combined = np.vstack(all_pixels)
        sample_size = min(50_000, len(combined))
        rng = np.random.default_rng(42)
        sampled = combined[rng.choice(len(combined), sample_size, replace=False)]

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(sampled)

        centers = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        order = np.argsort(-counts)

        dominant = [tuple(int(v) for v in centers[i]) for i in order]
        palette = [_rgb_to_hex(c) for c in dominant]

        avg_b = float(np.mean(brightnesses))
        avg_c = float(np.mean(contrasts))
        contrast = "low" if avg_c < 40 else "high" if avg_c >= 70 else "medium"

        return dominant, palette, avg_b, contrast

    # ── Aggregation ──────────────────────────────────────────────────

    def _aggregate(
        self,
        per_image: list[dict[str, Any]],
        embeddings: list[np.ndarray],
        image_paths: list[Path],
    ) -> DeepStyleProfile:
        layout = _most_common(
            [r["layout_type"] for r in per_image], default="centered layout"
        )
        typo = _most_common(
            [r["typography_style"] for r in per_image], default="sans-serif typography"
        )
        hierarchy = _most_common(
            [r["visual_hierarchy"] for r in per_image],
            default="balanced visual hierarchy",
        )

        moods = _flatten_union([r.get("mood_keywords", []) for r in per_image])
        comps = _flatten_union([r.get("composition_tags", []) for r in per_image])

        # strip trailing descriptor words for cleaner output
        layout_clean = layout.replace(" layout", "").replace(" post", "")
        typo_clean = typo.replace(" typography", "")
        hierarchy_clean = hierarchy.replace(" visual hierarchy", "")
        moods_clean = [m.replace(" mood", "") for m in moods]
        comps_clean = [
            c.replace(" composition", "").replace(" design", "") for c in comps
        ]

        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-9)

        dominant, palette, brightness, contrast = self._extract_colors(image_paths)
        color_desc = _describe_color_palette(dominant, brightness)

        profile = DeepStyleProfile(
            layout_type=layout_clean,
            typography_style=typo_clean,
            mood_keywords=moods_clean,
            composition_tags=comps_clean,
            visual_hierarchy=hierarchy_clean,
            dominant_colors=dominant,
            color_palette=palette,
            color_palette_description=color_desc,
            avg_brightness=brightness,
            contrast_level=contrast,
            style_embedding=avg_emb.tolist(),
            per_image=per_image,
            image_count=len(per_image),
        )
        profile.style_prompt_suffix = self._build_style_prompt(profile)
        return profile

    @staticmethod
    def _build_style_prompt(p: DeepStyleProfile) -> str:
        parts = [
            f"Layout: {p.layout_type}",
            f"Typography: {p.typography_style}",
            f"Mood: {', '.join(p.mood_keywords[:5])}",
            f"Composition: {', '.join(p.composition_tags[:5])}",
            f"Colors: {p.color_palette_description}",
            f"Color hex: {', '.join(p.color_palette[:4])}",
            f"Visual hierarchy: {p.visual_hierarchy}",
            f"Brightness: {'bright' if p.avg_brightness > 0.6 else 'dark' if p.avg_brightness < 0.35 else 'balanced'}",
            f"Contrast: {p.contrast_level}",
        ]
        return "; ".join(parts)

    # ── Cache ────────────────────────────────────────────────────────

    def _cache_fingerprint(self, image_paths: list[Path]) -> str:
        combined = "|".join(sorted(_file_hash(p) for p in image_paths))
        return hashlib.sha256(combined.encode()).hexdigest()

    def _cache_path(self) -> Path:
        return self.cache_dir / "deep_style_cache.json"

    def _load_cache(self, image_paths: list[Path]) -> DeepStyleProfile | None:
        cp = self._cache_path()
        if not cp.exists():
            return None
        try:
            data = json.loads(cp.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("fingerprint") != self._cache_fingerprint(image_paths):
            return None
        raw = data.get("result")
        if raw is None:
            return None
        return self._dict_to_profile(raw)

    def _save_cache(
        self, image_paths: list[Path], profile: DeepStyleProfile
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self._cache_fingerprint(image_paths),
            "image_count": len(image_paths),
            "result": self._profile_to_dict(profile),
        }
        try:
            self._cache_path().write_text(json.dumps(payload, indent=2))
        except OSError as exc:
            logger.warning("Failed to write deep cache: %s", exc)

    @staticmethod
    def _profile_to_dict(p: DeepStyleProfile) -> dict[str, Any]:
        return {
            "layout_type": p.layout_type,
            "typography_style": p.typography_style,
            "mood_keywords": p.mood_keywords,
            "composition_tags": p.composition_tags,
            "visual_hierarchy": p.visual_hierarchy,
            "dominant_colors": [list(c) for c in p.dominant_colors],
            "color_palette": p.color_palette,
            "color_palette_description": p.color_palette_description,
            "avg_brightness": p.avg_brightness,
            "contrast_level": p.contrast_level,
            "style_embedding": p.style_embedding,
            "style_prompt_suffix": p.style_prompt_suffix,
            "per_image": p.per_image,
            "image_count": p.image_count,
        }

    @staticmethod
    def _dict_to_profile(d: dict[str, Any]) -> DeepStyleProfile:
        return DeepStyleProfile(
            layout_type=d.get("layout_type", "centered"),
            typography_style=d.get("typography_style", "sans-serif"),
            mood_keywords=d.get("mood_keywords", []),
            composition_tags=d.get("composition_tags", []),
            visual_hierarchy=d.get("visual_hierarchy", "balanced"),
            dominant_colors=[tuple(c) for c in d.get("dominant_colors", [])],
            color_palette=d.get("color_palette", []),
            color_palette_description=d.get("color_palette_description", ""),
            avg_brightness=d.get("avg_brightness", 0.5),
            contrast_level=d.get("contrast_level", "medium"),
            style_embedding=d.get("style_embedding", []),
            style_prompt_suffix=d.get("style_prompt_suffix", ""),
            per_image=d.get("per_image", []),
            image_count=d.get("image_count", 0),
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_image_paths(self) -> list[Path]:
        if not self.reference_dir.is_dir():
            return []
        return sorted(
            p
            for p in self.reference_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _default_profile() -> DeepStyleProfile:
        p = DeepStyleProfile(
            layout_type="centered",
            typography_style="sans-serif",
            mood_keywords=["professional", "minimal", "clean"],
            composition_tags=["negative space", "balanced"],
            visual_hierarchy="balanced",
            dominant_colors=[(41, 41, 41), (255, 255, 255), (0, 122, 255)],
            color_palette=["#292929", "#ffffff", "#007aff"],
            color_palette_description="Neutral tones with a single brand accent color",
            avg_brightness=0.5,
            contrast_level="medium",
            style_embedding=[],
            per_image=[],
            image_count=0,
        )
        p.style_prompt_suffix = DeepStyleAnalyzer._build_style_prompt(p)
        return p
