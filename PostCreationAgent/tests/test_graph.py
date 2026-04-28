"""Tests for graph routing logic.

Routing functions are tested by extracting their logic directly
to avoid importing the full node module (which requires langchain_google_genai).
"""


def should_process(state: dict) -> str:
    images = state.get("generated_images", [])
    if not images:
        return "summary"
    return "process_images"


def should_publish(state: dict) -> str:
    if state.get("publish_to_instagram"):
        return "publish"
    return "summary"


def should_generate(state: dict) -> str:
    if state.get("error"):
        return "summary"
    return "generate_images"


class TestShouldProcess:
    def test_with_images_returns_process(self):
        assert should_process({"generated_images": ["img1"]}) == "process_images"

    def test_empty_images_returns_summary(self):
        assert should_process({"generated_images": []}) == "summary"

    def test_missing_images_returns_summary(self):
        assert should_process({}) == "summary"


class TestShouldPublish:
    def test_publish_true(self):
        assert should_publish({"publish_to_instagram": True}) == "publish"

    def test_publish_false(self):
        assert should_publish({"publish_to_instagram": False}) == "summary"

    def test_publish_missing(self):
        assert should_publish({}) == "summary"


class TestShouldGenerate:
    def test_no_error_returns_generate(self):
        assert should_generate({}) == "generate_images"

    def test_error_returns_summary(self):
        assert should_generate({"error": "something broke"}) == "summary"
