"""Session history must match what the browser is still showing.

/api/ask/stream records the exchange after the token loop finishes. The Stop
button aborts the fetch, which closes the response generator and raises
GeneratorExit inside it — a BaseException, so it slips past `except Exception`.
The turn then went unrecorded while the client kept the partial answer on
screen under a "stopped" marker, and the next question was rewritten against a
history missing an exchange the user could still see.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as appmod  # noqa: E402

TOKENS = ["The ", "clause ", "governs ", "indemnity ", "in ", "full [1]."]


class _Doc:
    page_content = "Clause 4 governs indemnity."
    metadata = {"source": "contract.pdf", "page": 3, "doc_type": "contract", "citations": []}


@pytest.fixture
def session(monkeypatch):
    appmod._sessions.clear()
    s = appmod.get_or_create_session(None)
    s.store = object()

    def fake_stream(store, question, *, mode="brief", history=None):
        return iter(TOKENS), [_Doc()], f"rewritten {question}"

    monkeypatch.setattr(appmod, "stream_answer", fake_stream)
    return s


def drive(session, question, stop_after=None):
    """Run the endpoint, optionally hanging up after `stop_after` chunks."""
    req = appmod.AskRequest(session_id=session.id, question=question, mode="brief")
    gen = appmod.ask_stream(req).body_iterator
    chunks = []

    async def pump():
        n = 0
        async for chunk in gen:
            chunks.append(chunk)
            n += 1
            if stop_after is not None and n >= stop_after:
                await gen.aclose()      # what a client disconnect does
                return

    asyncio.run(pump())
    return chunks


def tokens_of(chunks):
    return "".join(
        json.loads(c.split("data: ", 1)[1])["t"]
        for c in chunks if c.startswith("event: token")
    )


class TestStoppedTurnIsRemembered:
    def test_partial_answer_is_recorded(self, session):
        chunks = drive(session, "What governs clause 4?", stop_after=3)
        seen = tokens_of(chunks)
        assert seen, "probe did not deliver any tokens before stopping"
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "What governs clause 4?"}
        # The server must remember exactly what the user was left looking at.
        assert session.history[1]["content"] == seen

    def test_follow_up_sees_the_stopped_turn(self, session, monkeypatch):
        seen_history = {}
        real = appmod.stream_answer

        def spy(store, question, *, mode="brief", history=None):
            seen_history["h"] = list(history or [])
            return real(store, question, mode=mode, history=history)

        monkeypatch.setattr(appmod, "stream_answer", spy)
        drive(session, "What governs clause 4?", stop_after=3)
        drive(session, "expand on that")
        assert len(seen_history["h"]) == 2, "follow-up lost the referent for 'that'"


class TestUnchangedPaths:
    def test_completed_turn_records_once(self, session):
        chunks = drive(session, "What governs clause 4?")
        assert session.history[1]["content"] == "".join(TOKENS)
        assert len(session.history) == 2, "a full run must not double-record"
        assert chunks[-1].startswith("event: done")

    def test_stream_error_is_not_recorded(self, session, monkeypatch):
        def exploding_stream(store, question, *, mode="brief", history=None):
            def gen():
                yield "partial "
                raise RuntimeError("upstream died")
            return gen(), [_Doc()], "rewritten"

        monkeypatch.setattr(appmod, "stream_answer", exploding_stream)
        chunks = drive(session, "What governs clause 4?")
        assert any(c.startswith("event: error") for c in chunks)
        # The client swaps the body for the error text and drops the partial,
        # so recording it would desync in the other direction.
        assert session.history == []

    def test_immediate_abort_records_nothing(self, session):
        drive(session, "What governs clause 4?", stop_after=1)  # only the meta frame
        assert session.history == []
