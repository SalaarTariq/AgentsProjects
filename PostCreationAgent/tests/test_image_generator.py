"""Tests for ImageGenerator (non-network)."""

import pytest
from PIL import Image

from src.image_generator import (
    AllProvidersExhaustedError,
    ImageGenerator,
    ProviderExhaustedError,
)


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


class TestBatchResilience:
    """A provider dying partway through a batch must not discard earlier images.

    Free tiers rate-limit mid-run routinely, and generate_images_node already
    distinguishes an empty result from AllProvidersExhaustedError — so the
    batch is expected to come back short rather than not at all.
    """

    @staticmethod
    def _with_provider(fn):
        gen = ImageGenerator()
        gen.providers = [{"name": "stub", "fn": fn}]
        return gen

    @staticmethod
    def _dies_after(n):
        """Provider that serves n images, then is permanently rate-limited."""
        served = {"n": 0}

        def provider(prompt, size, seed_offset=0):
            served["n"] += 1
            if served["n"] <= n:
                return Image.new("RGB", (8, 8), "red")
            raise ProviderExhaustedError("stub rate limit")

        return provider

    def test_partial_batch_is_kept(self):
        gen = self._with_provider(self._dies_after(4))
        images = gen.generate("prompt", count=5, size=(8, 8))
        assert len(images) == 4

    def test_failure_on_last_image_still_returns_the_rest(self):
        gen = self._with_provider(self._dies_after(1))
        images = gen.generate("prompt", count=2, size=(8, 8))
        assert len(images) == 1

    def test_generating_nothing_still_raises(self):
        gen = self._with_provider(self._dies_after(0))
        with pytest.raises(AllProvidersExhaustedError):
            gen.generate("prompt", count=3, size=(8, 8))

    def test_healthy_provider_fills_the_batch(self):
        gen = self._with_provider(lambda p, s, seed_offset=0: Image.new("RGB", (8, 8), "blue"))
        assert len(gen.generate("prompt", count=3, size=(8, 8))) == 3
