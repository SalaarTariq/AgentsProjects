"""Every interactive control needs a visible focus indicator (WCAG 2.4.7).

The composer's textarea set `outline: none` and the stylesheet carried no
:focus rule at all, so a keyboard user tabbing into the question field — the
primary control of the whole app — got no indication of where they were.
"""

import re
from pathlib import Path

import pytest

CSS = Path(__file__).resolve().parent.parent / "static" / "styles.css"


def _relative_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


@pytest.fixture(scope="module")
def css():
    return CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declarations(css):
    """The stylesheet with /* comments */ stripped.

    Prose about a property is not a declaration of it — the comment explaining
    why the textarea drops its outline would otherwise be counted as another
    suppression.
    """
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


@pytest.fixture(scope="module")
def tokens(css):
    """The :root custom properties, so the tests use the real palette."""
    root = css[css.index(":root {") : css.index("}", css.index(":root {"))]
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", root))


def _rule_body(css: str, selector: str) -> str:
    start = css.index(selector)
    return css[start : css.index("}", start)]


class TestContrastHelper:
    """The helper itself, against values the WCAG spec pins down."""

    def test_black_on_white_is_21(self):
        assert round(contrast_ratio("#000000", "#ffffff"), 1) == 21.0

    def test_identical_colours_are_1(self):
        assert round(contrast_ratio("#8a6d2c", "#8a6d2c"), 2) == 1.0


class TestFocusIndicatorExists:
    def test_stylesheet_has_focus_rules(self, css):
        assert ":focus-visible" in css or ":focus-within" in css

    def test_composer_shows_focus(self, css):
        """The textarea kills its own outline, so the wrapper must show it."""
        assert ".composer:focus-within" in css
        body = _rule_body(css, ".composer:focus-within")
        assert any(p in body for p in ("outline", "box-shadow", "border-color")), (
            "focus rule sets nothing visible"
        )

    def test_outline_none_is_compensated(self, declarations, css):
        """Any element dropping its outline needs a replacement indicator."""
        suppressors = declarations.count("outline: none")
        assert suppressors == 1, f"{suppressors} outline suppressions; only the composer is accounted for"
        assert ".composer:focus-within" in css

    @pytest.mark.parametrize(
        "selector",
        [".btn:focus-visible", ".mode-btn:focus-visible", ".prompt-card:focus-visible"],
    )
    def test_controls_have_a_keyboard_ring(self, css, selector):
        assert selector in css


class TestIndicatorIsPerceivable:
    """WCAG 1.4.11 asks 3:1 for a non-text indicator against what it sits on."""

    BACKGROUNDS = ["--surface", "--surface-2", "--bg"]

    @pytest.mark.parametrize("bg_token", BACKGROUNDS)
    def test_focus_colour_contrasts(self, tokens, bg_token):
        ratio = contrast_ratio(tokens["--gold"], tokens[bg_token])
        assert ratio >= 3.0, f"--gold on {bg_token} is only {ratio:.2f}:1"

    def test_focus_rules_use_the_gold_token(self, css):
        body = _rule_body(css, ".btn:focus-visible")
        assert "var(--gold)" in body
