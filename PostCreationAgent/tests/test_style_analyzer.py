"""StyleAnalyzer must return the same palette for the same reference images.

The palette is not just reported — its *order* is load-bearing. _build_style_prompt
interpolates palette[:3] into the text handed to the image model, _derive_keywords
branches on colors[0], and nodes._build_style_context passes color_palette[:4] into
the prompt. So a palette that reshuffles between runs quietly changes the images the
pipeline produces from an unchanged brand folder.
"""

import random
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.style_analyzer import StyleAnalyzer

BRAND = [
    (236, 237, 230), (47, 48, 41), (126, 143, 102),
    (212, 211, 198), (153, 170, 123), (61, 70, 50),
]


@pytest.fixture(scope="module")
def brand_dir():
    """Six block-structured images — real clusters plus light grain, and enough
    pixels (6 x 300 x 300) to exceed the 50k subsample cap the analyzer applies."""
    d = Path(tempfile.mkdtemp())
    rng = random.Random(11)
    for i in range(6):
        im = Image.new("RGB", (300, 300), BRAND[i % len(BRAND)])
        draw = ImageDraw.Draw(im)
        for j, colour in enumerate(BRAND):
            draw.rectangle([j * 50, i * 20, j * 50 + 45, i * 20 + 120], fill=colour)
        px = im.load()
        for _ in range(3000):
            x, y = rng.randrange(300), rng.randrange(300)
            r, g, b = px[x, y]
            px[x, y] = (min(255, r + 12), min(255, g + 12), min(255, b + 12))
        im.save(d / f"ref{i}.png")
    return d


class TestPaletteReproducibility:
    def test_repeated_runs_agree(self, brand_dir):
        palettes = {tuple(StyleAnalyzer(brand_dir).analyze_images().color_palette)
                    for _ in range(3)}
        assert len(palettes) == 1, f"palette drifted across runs: {palettes}"

    def test_palette_order_is_stable(self, brand_dir):
        first = StyleAnalyzer(brand_dir).analyze_images().color_palette
        second = StyleAnalyzer(brand_dir).analyze_images().color_palette
        assert first == second, "same colours but reshuffled — the style prompt would change"

    def test_derived_prompt_is_stable(self, brand_dir):
        """What actually reaches the image model must not move between runs."""
        a = StyleAnalyzer(brand_dir).analyze_images()
        b = StyleAnalyzer(brand_dir).analyze_images()
        assert a.style_prompt_suffix == b.style_prompt_suffix
        assert a.style_keywords == b.style_keywords

    def test_palette_still_tracks_the_actual_colours(self, brand_dir):
        """Determinism must not come at the cost of clustering the wrong thing."""
        profile = StyleAnalyzer(brand_dir).analyze_images()
        expected = {"#{:02x}{:02x}{:02x}".format(*c) for c in BRAND}
        assert len(profile.color_palette) == 6
        # Every reported colour should sit near one of the six brand colours.
        for hexval in profile.color_palette:
            r, g, b = (int(hexval[i:i + 2], 16) for i in (1, 3, 5))
            assert min(abs(r - cr) + abs(g - cg) + abs(b - cb)
                       for cr, cg, cb in BRAND) < 40, f"{hexval} is not a brand colour"
        assert len(set(profile.color_palette) & expected) >= 4
