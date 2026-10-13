"""Motion that runs indefinitely must respect prefers-reduced-motion.

Two animations in this stylesheet are declared `infinite`: the streaming cursor
blinks for as long as an answer takes to arrive, and the thinking dots cycle for
the whole wait. A streamed brief runs well past the five seconds WCAG 2.2.2
allows automatic motion to continue with no way to pause it, and the UI offers
no such control — so the system preference has to be honoured.

These tests derive the list of animated selectors from the stylesheet itself
rather than naming them, so a new animation added later is covered too.
"""

import re
from pathlib import Path

import pytest

CSS = Path(__file__).resolve().parent.parent / "static" / "styles.css"


@pytest.fixture(scope="module")
def css():
    return CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declarations(css):
    """Stylesheet with /* comments */ removed — prose is not a declaration."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


@pytest.fixture(scope="module")
def reduced_block(declarations):
    """The body of the @media (prefers-reduced-motion: reduce) block."""
    m = re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{", declarations)
    assert m, "no prefers-reduced-motion block in the stylesheet"
    i = m.end()
    depth, start = 1, i
    while depth:
        ch = declarations[i]
        depth += (ch == "{") - (ch == "}")
        i += 1
    return declarations[start : i - 1]


def _animated_selectors(declarations):
    """Selectors declaring an animation, paired with whether it is infinite."""
    found = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", declarations):
        selector, body = m.group(1).strip(), m.group(2)
        if selector.startswith("@") or "animation:" not in body:
            continue
        decl = re.search(r"animation:\s*([^;]+);", body).group(1)
        found.append((selector, "infinite" in decl))
    return found


class TestInventory:
    """Guard the premise: if these stop holding, the tests below mean nothing."""

    def test_stylesheet_declares_animations(self, declarations):
        assert _animated_selectors(declarations), "no animations found to check"

    def test_at_least_one_is_infinite(self, declarations):
        assert any(inf for _, inf in _animated_selectors(declarations))


class TestReducedMotionIsHonoured:
    def test_block_exists(self, reduced_block):
        assert reduced_block.strip()

    def test_every_infinite_animation_is_stopped(self, declarations, reduced_block):
        """Each selector running an infinite animation must be named and halted."""
        for selector, infinite in _animated_selectors(declarations):
            if not infinite:
                continue
            base = selector.split("::")[0].split(":")[0].strip()
            assert base in reduced_block, f"{selector} keeps animating under reduced motion"

    def test_animations_are_neutralised(self, reduced_block):
        assert "animation: none" in reduced_block or "animation-name: none" in reduced_block

    def test_iteration_count_is_capped(self, reduced_block):
        assert "animation-iteration-count: 1" in reduced_block

    def test_transitions_are_shortened(self, reduced_block):
        assert "transition-duration" in reduced_block

    def test_universal_selector_catches_future_motion(self, reduced_block):
        assert "*" in reduced_block, "nothing blanket-covers motion added later"


class TestStateIsStillReadable:
    """Stopping the motion must not remove the information it carried."""

    def test_streaming_cursor_keeps_its_glyph(self, declarations, reduced_block):
        base = re.search(r"\.streaming::after\s*\{([^}]*)\}", declarations).group(1)
        assert "content:" in base, "cursor relies on the animation for its glyph"
        assert "content:" not in reduced_block or "animation" in reduced_block

    def test_thinking_dots_keep_a_static_fallback(self, declarations):
        base = re.search(r"\.thinking::after\s*\{([^}]*)\}", declarations).group(1)
        assert re.search(r'content:\s*"[^"]+"', base), (
            "with the animation off the dots would render empty"
        )
