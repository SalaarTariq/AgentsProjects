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
        gen._failed_providers["pollinations"] = 0.0
        gen.reset_failed_providers()
        assert len(gen._failed_providers) == 0

    def test_provider_cooldown_expires(self):
        import time
        from src import image_generator as ig_mod

        gen = ImageGenerator()
        # Mark provider as failed 1 second after the cooldown horizon — should re-enable.
        gen._failed_providers["pollinations"] = time.time() - (ig_mod.PROVIDER_COOLDOWN_SECONDS + 1)
        assert gen._is_provider_on_cooldown("pollinations") is False
        assert "pollinations" not in gen._failed_providers

    def test_provider_cooldown_active(self):
        gen = ImageGenerator()
        import time
        gen._failed_providers["pollinations"] = time.time()
        assert gen._is_provider_on_cooldown("pollinations") is True
