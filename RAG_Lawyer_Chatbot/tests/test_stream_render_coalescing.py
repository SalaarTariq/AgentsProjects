"""Streaming must not re-render the whole answer once per token.

Every `token` event re-ran marked.parse over the entire accumulated buffer and
replaced the message's innerHTML. Both costs grow with the answer, so the work
is quadratic in its length: measured against marked 12.0.2, a 2380-character
brief cost 25ms of parsing and a 4760-character one 95ms — double the answer,
roughly four times the work — before any of the DOM reflow a browser adds on
top. Paints are now coalesced into one per animation frame.

These are source-level assertions: the project has no JS test runner. The
behaviour was checked separately by extracting the real helpers from app.js and
driving them under node — see the commit message.
"""

from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"


@pytest.fixture(scope="module")
def src():
    return APP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def token_branch(src):
    """Just the `token` arm of handleSSEEvent."""
    start = src.index('} else if (event === "token") {')
    end = src.index('} else if (event === "done") {', start)
    return src[start:end]


@pytest.fixture(scope="module")
def done_branch(src):
    start = src.index('} else if (event === "done") {')
    end = src.index('} else if (event === "error") {', start)
    return src[start:end]


@pytest.fixture(scope="module")
def error_branch(src):
    start = src.index('} else if (event === "error") {')
    return src[start:src.index("\n    }", start)]


class TestTokensAreCoalesced:
    def test_token_schedules_rather_than_paints(self, token_branch):
        assert "scheduleAnswerPaint()" in token_branch

    def test_token_does_not_write_innerhtml_directly(self, token_branch):
        # The one permitted write is the clear on the very first token.
        writes = token_branch.count("body.innerHTML")
        assert writes == 1, f"{writes} direct innerHTML writes in the token path"
        assert 'body.innerHTML = ""' in token_branch

    def test_token_does_not_parse_markdown_directly(self, token_branch):
        assert "renderMarkdown" not in token_branch

    def test_scheduler_is_idempotent_within_a_frame(self, src):
        """A second token in the same frame must not queue a second paint."""
        start = src.index("function scheduleAnswerPaint")
        body = src[start:src.index("\n  }", start)]
        assert "renderHandle === null" in body
        assert "requestAnimationFrame" in body


class TestFinalStateIsNotLeftToAFrame:
    def test_done_paints_immediately(self, done_branch):
        assert "paintAnswer()" in done_branch

    def test_done_drops_any_pending_frame(self, done_branch):
        assert "cancelAnswerPaint()" in done_branch


class TestPendingPaintCannotClobberAMessage:
    """A queued frame firing after an error would overwrite it — verified."""

    def test_error_cancels_first(self, error_branch):
        assert "cancelAnswerPaint()" in error_branch
        assert error_branch.index("cancelAnswerPaint()") < error_branch.index(
            "body.innerHTML"
        ), "cancel must come before the message is written"

    def test_abort_path_cancels(self, src):
        # app.js has more than one `catch (err)`; the streaming one is the
        # block that handles AbortError.
        abort = src.index('err.name === "AbortError"')
        start = src.rindex("} catch (err) {", 0, abort)
        catch = src[start:src.index("} finally {", start)]
        assert "cancelAnswerPaint()" in catch
        assert catch.index("cancelAnswerPaint()") < catch.index("body.innerHTML")

    def test_cancel_actually_cancels(self, src):
        start = src.index("function cancelAnswerPaint")
        body = src[start:src.index("\n  }", start)]
        assert "cancelAnimationFrame" in body
        assert "renderHandle = null" in body
