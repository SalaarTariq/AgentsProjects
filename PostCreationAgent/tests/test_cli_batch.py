"""One failing prompt must not take the rest of the batch down with it.

cmd_batch calls agent.create_post in a bare loop. The prompts in a batch are
independent and the user queued them deliberately, but any exception escaping
one call abandoned every prompt behind it, skipped the summary, and — because
main()'s interactive loop has no handler either — ended the whole CLI session.
"""

import types

import pytest

from src import main as cli


class RecordingAgent:
    """Stand-in for PostCreationAgent that fails on nominated prompts."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.attempted = []
        self.succeeded = []

    def create_post(self, prompt, count=1, post_type="feed"):
        self.attempted.append(prompt)
        if prompt in self.fail_on:
            raise RuntimeError(f"provider died on {prompt}")
        self.succeeded.append(prompt)
        return [f"/out/{prompt}.jpg"]


@pytest.fixture
def answers(monkeypatch):
    """Feed cmd_batch its prompts, then the post type."""

    def _set(prompts, post_type="feed"):
        scripted = iter(list(prompts) + ["", post_type])
        monkeypatch.setattr(
            cli, "Prompt", types.SimpleNamespace(ask=lambda *a, **k: next(scripted))
        )

    return _set


class TestBatchSurvivesAFailure:
    def test_remaining_prompts_still_run(self, answers):
        answers(["alpha", "beta", "gamma"])
        agent = RecordingAgent(fail_on={"beta"})
        cli.cmd_batch(agent)
        assert agent.attempted == ["alpha", "beta", "gamma"]
        assert agent.succeeded == ["alpha", "gamma"]

    def test_does_not_raise(self, answers):
        """Escaping here would also kill the interactive loop in main()."""
        answers(["alpha", "beta"])
        cli.cmd_batch(RecordingAgent(fail_on={"beta"}))  # must not raise

    def test_first_prompt_failing_does_not_abort(self, answers):
        answers(["alpha", "beta"])
        agent = RecordingAgent(fail_on={"alpha"})
        cli.cmd_batch(agent)
        assert agent.succeeded == ["beta"]

    def test_every_prompt_failing_is_survivable(self, answers):
        answers(["alpha", "beta"])
        agent = RecordingAgent(fail_on={"alpha", "beta"})
        cli.cmd_batch(agent)
        assert agent.attempted == ["alpha", "beta"]
        assert agent.succeeded == []

    def test_failure_is_reported(self, answers, capsys):
        answers(["alpha", "beta"])
        cli.cmd_batch(RecordingAgent(fail_on={"beta"}))
        out = capsys.readouterr().out
        assert "failed" in out.lower()
        assert "beta" in out


class TestUnchangedBehaviour:
    def test_clean_batch_runs_every_prompt(self, answers):
        answers(["alpha", "beta", "gamma"])
        agent = RecordingAgent()
        cli.cmd_batch(agent)
        assert agent.succeeded == ["alpha", "beta", "gamma"]

    def test_empty_queue_is_a_no_op(self, answers):
        answers([])
        agent = RecordingAgent()
        cli.cmd_batch(agent)
        assert agent.attempted == []

    def test_post_type_is_passed_through(self, answers, monkeypatch):
        seen = {}

        class Agent(RecordingAgent):
            def create_post(self, prompt, count=1, post_type="feed"):
                seen["post_type"] = post_type
                return super().create_post(prompt, count, post_type)

        answers(["alpha"], post_type="story")
        cli.cmd_batch(Agent())
        assert seen["post_type"] == "story"
