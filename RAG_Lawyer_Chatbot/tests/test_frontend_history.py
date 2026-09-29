"""A failed turn must not become part of the model's transcript.

frontend.py keeps errors in st.session_state.messages so they survive Streamlit
reruns and stay on screen. That same list is what gets handed to
answer_question as `history`, so an error was replayed to the model as a real
assistant turn saying "Error generating answer: ...". rewrite_query and
_build_messages both read it, meaning one timeout kept distorting every later
question in the conversation. app.py records nothing on failure; this pins the
Streamlit UI to the same rule while keeping the error visible to the user.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
FRONTEND = PROJECT_DIR / "frontend.py"


class FakeAnswer:
    answer = "Clause 4 governs indemnity [1]."
    sources = []


@pytest.fixture
def harness(monkeypatch):
    """Install the streamlit stub plus stub pipeline modules, then hand back a runner."""
    spec = importlib.util.spec_from_file_location(
        "streamlit", TESTS_DIR / "_streamlit_stub.py"
    )
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    st.reset()
    monkeypatch.setitem(sys.modules, "streamlit", st)

    state = {"fail": False, "seen_history": None, "calls": 0}

    def answer_question(store, question, *, mode="brief", history=None):
        state["calls"] += 1
        state["seen_history"] = list(history or [])
        if state["fail"]:
            raise RuntimeError("Connection to Groq timed out")
        return FakeAnswer()

    rp = types.ModuleType("rag_pipeline")
    rp.answer_question = answer_question
    vd = types.ModuleType("vector_database")
    vd.build_hybrid_store = lambda paths: object()
    monkeypatch.setitem(sys.modules, "rag_pipeline", rp)
    monkeypatch.setitem(sys.modules, "vector_database", vd)

    st.session_state["store"] = object()

    def run(question, fail=False):
        state["fail"] = fail
        st.SCRIPT["chat_input"] = question
        s = importlib.util.spec_from_file_location("frontend", FRONTEND)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)

    return types.SimpleNamespace(st=st, state=state, run=run)


class TestFailedTurnIsNotReplayed:
    def test_error_stays_on_screen(self, harness):
        harness.run("What governs clause 4?", fail=True)
        msgs = harness.st.session_state["messages"]
        assert len(msgs) == 2
        assert msgs[1]["content"].startswith("Error generating answer:")

    def test_error_is_kept_out_of_model_history(self, harness):
        harness.run("What governs clause 4?", fail=True)
        harness.run("expand on that")
        assert harness.state["seen_history"] == [], (
            "the model was handed the failed exchange: "
            f"{harness.state['seen_history']}"
        )

    def test_no_fabricated_assistant_turn(self, harness):
        harness.run("What governs clause 4?", fail=True)
        harness.run("expand on that")
        replayed = " ".join(m["content"] for m in harness.state["seen_history"])
        assert "Error generating answer" not in replayed


class TestSuccessfulHistoryStillFlows:
    def test_completed_turn_is_replayed(self, harness):
        harness.run("What governs clause 4?")
        harness.run("expand on that")
        hist = harness.state["seen_history"]
        assert [m["role"] for m in hist] == ["user", "assistant"]
        assert hist[0]["content"] == "What governs clause 4?"
        assert hist[1]["content"] == FakeAnswer.answer

    def test_good_turns_survive_a_failure_between_them(self, harness):
        harness.run("What governs clause 4?")          # good
        harness.run("and clause 9?", fail=True)        # blip
        harness.run("expand on that")                  # follow-up
        hist = harness.state["seen_history"]
        contents = [m["content"] for m in hist]
        assert "What governs clause 4?" in contents, "a good turn was lost"
        assert FakeAnswer.answer in contents
        assert not any("Error generating answer" in c for c in contents)
        assert "and clause 9?" not in contents, "the failed question was replayed"
