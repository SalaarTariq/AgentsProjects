"""Tests for PostDesigner and theme system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.post_designer import PostDesigner, THEMES, build_color_scheme


class TestThemes:
    def test_all_expected_themes_exist(self):
        expected = {"light", "dark", "sage", "cream", "olive", "forest", "warm"}
        assert set(THEMES.keys()) == expected

    def test_each_theme_has_required_keys(self):
        required = {"bg", "text", "accent", "accent_dark", "accent_light", "divider"}
        for name, theme in THEMES.items():
            assert set(theme.keys()) == required, f"Theme '{name}' missing keys"

    def test_all_colors_are_valid_hex(self):
        for name, theme in THEMES.items():
            for key, color in theme.items():
                assert color.startswith("#"), f"{name}.{key} = {color} not hex"
                assert len(color) == 7, f"{name}.{key} = {color} wrong length"
                int(color[1:], 16)  # should not raise

    def test_light_theme_has_dark_text_on_light_bg(self):
        light = THEMES["light"]
        bg_lum = _hex_luminance(light["bg"])
        text_lum = _hex_luminance(light["text"])
        assert bg_lum > text_lum, "Light theme: bg should be lighter than text"

    def test_dark_theme_has_light_text_on_dark_bg(self):
        dark = THEMES["dark"]
        bg_lum = _hex_luminance(dark["bg"])
        text_lum = _hex_luminance(dark["text"])
        assert text_lum > bg_lum, "Dark theme: text should be lighter than bg"

    def test_sufficient_contrast_all_themes(self):
        for name, theme in THEMES.items():
            bg_lum = _hex_luminance(theme["bg"])
            text_lum = _hex_luminance(theme["text"])
            ratio = _contrast_ratio(bg_lum, text_lum)
            assert ratio >= 3.0, f"Theme '{name}' contrast ratio {ratio:.1f} too low"


class TestBuildColorScheme:
    def test_default_returns_light(self):
        scheme = build_color_scheme()
        assert scheme["bg"] == THEMES["light"]["bg"]

    def test_theme_selection(self):
        scheme = build_color_scheme(theme="sage")
        assert scheme["bg"] == THEMES["sage"]["bg"]

    def test_override_bg(self):
        scheme = build_color_scheme(theme="light", bg="#FF0000")
        assert scheme["bg"] == "#FF0000"
        assert scheme["text"] == THEMES["light"]["text"]

    def test_override_all(self):
        scheme = build_color_scheme(bg="#111", text="#222", accent="#333", divider="#444")
        assert scheme["bg"] == "#111"
        assert scheme["text"] == "#222"
        assert scheme["accent"] == "#333"
        assert scheme["divider"] == "#444"

    def test_invalid_theme_falls_back_to_light(self):
        scheme = build_color_scheme(theme="nonexistent")
        assert scheme["bg"] == THEMES["light"]["bg"]


class TestPostDesigner:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = patch("src.config.OUTPUT_DIR", Path(self._tmpdir))
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_generate_creates_png(self):
        designer = PostDesigner(theme="light")
        path = designer.generate(lines=["Hello", "World"])
        assert path.exists()
        assert path.suffix == ".png"

    def test_generate_with_custom_name(self):
        designer = PostDesigner(theme="dark")
        path = designer.generate(lines=["Test"], output_name="custom_test")
        assert path.name == "custom_test.png"

    def test_generate_correct_dimensions(self):
        from PIL import Image
        designer = PostDesigner(width=1080, height=1080, theme="cream")
        path = designer.generate(lines=["Dimension test"])
        img = Image.open(path)
        assert img.size == (1080, 1080)

    def test_accent_post_creates_png(self):
        designer = PostDesigner(theme="sage")
        path = designer.generate_accent_post(
            lines=["Line one", "Line two", "Highlighted"],
            highlight_line=-1,
        )
        assert path.exists()

    def test_all_themes_render_without_error(self):
        for theme_name in THEMES:
            designer = PostDesigner(theme=theme_name)
            path = designer.generate(lines=["Theme test"], output_name=f"theme_{theme_name}")
            assert path.exists(), f"Theme '{theme_name}' failed to render"

    def test_tagline_renders(self):
        designer = PostDesigner(theme="forest")
        path = designer.generate(lines=["Main"], tagline="A tagline here")
        assert path.exists()

    def test_uppercase_default(self):
        designer = PostDesigner(theme="light")
        path = designer.generate(lines=["lowercase text"])
        assert path.exists()

    def test_no_uppercase(self):
        designer = PostDesigner(theme="warm")
        path = designer.generate(lines=["Mixed Case"], uppercase=False)
        assert path.exists()


# ── Helpers ──────────────────────────────────────────────────────────

def _hex_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    def _linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def _contrast_ratio(lum1: float, lum2: float) -> float:
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)
