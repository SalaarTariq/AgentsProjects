"""Tests for graph routing logic.

Imports the real routing functions from src.routing so a regression in
the production code can't silently pass.
"""

from src.routing import should_process, should_publish, should_generate


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
