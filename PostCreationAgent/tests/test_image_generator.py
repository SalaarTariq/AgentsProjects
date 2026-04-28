"""Tests for ImageGenerator (non-network)."""

from src.image_generator import ImageGenerator


class TestClosestAspectRatio:
    def test_square(self):
        assert ImageGenerator._closest_aspect_ratio((1080, 1080)) == "1:1"

    def test_portrait(self):
        assert ImageGenerator._closest_aspect_ratio((1080, 1350)) == "4:5"

    def test_landscape_wide(self):
        result = ImageGenerator._closest_aspect_ratio((1920, 1080))
        assert result == "16:9"

    def test_story(self):
        result = ImageGenerator._closest_aspect_ratio((1080, 1920))
        assert result == "9:16"


class TestProviderList:
    def test_always_has_at_least_one_provider(self):
        gen = ImageGenerator()
        assert len(gen.providers) >= 1

    def test_pollinations_is_default(self):
        gen = ImageGenerator()
        names = [p["name"] for p in gen.providers]
        assert "pollinations" in names

    def test_reset_failed_providers(self):
        gen = ImageGenerator()
        gen._failed_providers.add("pollinations")
        gen.reset_failed_providers()
        assert len(gen._failed_providers) == 0
