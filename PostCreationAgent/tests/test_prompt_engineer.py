"""Tests for PromptEngineer."""

from src.prompt_engineer import PromptEngineer
from src.style_analyzer import StyleProfile


class TestEnhancePrompt:
    def test_basic_enhancement(self):
        eng = PromptEngineer()
        result = eng.enhance_prompt("sunset over mountains")
        assert "sunset over mountains" in result
        assert "instagram" in result.lower()

    def test_post_type_dimension_hints(self):
        eng = PromptEngineer()
        for post_type, keyword in [
            ("feed", "1:1"),
            ("portrait", "4:5"),
            ("story", "9:16"),
            ("landscape", "1.91:1"),
        ]:
            result = eng.enhance_prompt("test", post_type)
            assert keyword in result, f"Missing {keyword} for {post_type}"

    def test_style_suffix_included(self):
        style = StyleProfile(style_prompt_suffix="warm tones, earthy palette")
        eng = PromptEngineer(style_profile=style)
        result = eng.enhance_prompt("a flower")
        assert "warm tones" in result


class TestCaption:
    def test_capitalize_first_letter(self):
        eng = PromptEngineer()
        assert eng.generate_caption("hello world") == "Hello world"

    def test_strips_trailing_period(self):
        eng = PromptEngineer()
        assert eng.generate_caption("test.") == "Test"


class TestHashtags:
    def test_generates_hashtags(self):
        eng = PromptEngineer()
        tags = eng.generate_hashtags("beautiful sunset photography landscape")
        assert all(t.startswith("#") for t in tags)

    def test_filters_stop_words(self):
        eng = PromptEngineer()
        tags = eng.generate_hashtags("the beautiful sunset in the sky")
        tag_words = [t.lstrip("#") for t in tags]
        assert "the" not in tag_words
        assert "in" not in tag_words

    def test_max_tags_respected(self):
        eng = PromptEngineer()
        tags = eng.generate_hashtags("one two three four five six", max_tags=3)
        assert len(tags) <= 3

    def test_includes_defaults(self):
        eng = PromptEngineer()
        tags = eng.generate_hashtags("hello")
        tag_set = set(tags)
        assert "#instagram" in tag_set or len(tags) >= 3


class TestFullCaption:
    def test_build_full_caption(self):
        eng = PromptEngineer()
        result = eng.build_full_caption("beautiful sunset")
        assert "\n\n" in result
        assert "#" in result
