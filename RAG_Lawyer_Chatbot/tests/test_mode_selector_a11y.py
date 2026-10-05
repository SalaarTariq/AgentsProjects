"""The answer-mode selector declares role="radiogroup" — it has to behave like one.

The four modes change the answer substantially (IRAC brief vs quick answer vs
drafting vs compare), but the markup gave the group a radiogroup role while its
children stayed plain buttons: no role="radio", no aria-checked, no tabindex.
The selection existed only as a CSS class, so a screen reader announced "Answer
mode, radio group" followed by four indistinguishable buttons with no way to
tell which one was in force — a WCAG 4.1.2 (Name, Role, Value) failure.
"""

from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
INDEX = STATIC / "index.html"
APP_JS = STATIC / "app.js"


class _RadioGroupParser(HTMLParser):
    """Collect the element tags/attrs nested inside the radiogroup container."""

    def __init__(self):
        super().__init__()
        self.group_attrs = None
        self._depth = 0
        self.children = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self._depth == 0:
            if a.get("role") == "radiogroup":
                self.group_attrs = a
                self._depth = 1
            return
        if tag == "button":
            self.children.append(a)
        self._depth += 1

    def handle_endtag(self, tag):
        if self._depth > 0:
            self._depth -= 1


@pytest.fixture(scope="module")
def set_mode_src():
    """Just the body of setMode, so the assertions cannot match elsewhere."""
    src = APP_JS.read_text(encoding="utf-8")
    start = src.index("function setMode(")
    return src[start:src.index("\n}", start)]


@pytest.fixture(scope="module")
def group():
    p = _RadioGroupParser()
    p.feed(INDEX.read_text(encoding="utf-8"))
    assert p.group_attrs is not None, 'no role="radiogroup" found in index.html'
    assert p.children, "radiogroup has no buttons"
    return p


class TestRadioGroupMarkup:
    def test_group_has_an_accessible_name(self, group):
        assert group.group_attrs.get("aria-label") or group.group_attrs.get(
            "aria-labelledby"
        )

    def test_every_option_is_a_radio(self, group):
        missing = [b.get("data-mode") for b in group.children if b.get("role") != "radio"]
        assert not missing, f'buttons in a radiogroup without role="radio": {missing}'

    def test_every_option_declares_checked_state(self, group):
        missing = [
            b.get("data-mode") for b in group.children if "aria-checked" not in b
        ]
        assert not missing, f"options with no aria-checked: {missing}"

    def test_exactly_one_option_is_checked(self, group):
        checked = [b for b in group.children if b.get("aria-checked") == "true"]
        assert len(checked) == 1, f"{len(checked)} options marked checked"

    def test_roving_tabindex(self, group):
        """A radiogroup is one Tab stop; arrows move within it."""
        tabbable = [b for b in group.children if b.get("tabindex") == "0"]
        assert len(tabbable) == 1, f"{len(tabbable)} options in the tab order"
        assert all(
            b.get("tabindex") in ("0", "-1") for b in group.children
        ), "every option needs an explicit tabindex"

    def test_checked_option_is_the_tabbable_one(self, group):
        checked = next(b for b in group.children if b.get("aria-checked") == "true")
        assert checked.get("tabindex") == "0"

    def test_buttons_declare_their_type(self, group):
        """Without type=button a <button> defaults to submit if ever moved into a form."""
        assert all(b.get("type") == "button" for b in group.children)


class TestSelectionIsMaintainedInJs:
    """Source-level checks: setMode must update the ARIA state, not just a class.

    These assert the wiring exists. The behaviour itself was verified by running
    the real setMode against a DOM stand-in — see the commit message.
    """

    def test_set_mode_updates_aria_checked(self, set_mode_src):
        assert "aria-checked" in set_mode_src

    def test_set_mode_updates_tabindex(self, set_mode_src):
        assert "tabIndex" in set_mode_src

    def test_arrow_keys_are_handled(self):
        src = APP_JS.read_text(encoding="utf-8")
        for key in ("ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"):
            assert key in src, f"{key} not handled for the radiogroup"
