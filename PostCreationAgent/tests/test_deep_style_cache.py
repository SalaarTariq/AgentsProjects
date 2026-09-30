"""CLIP_MODEL must actually take effect.

DeepStyleAnalyzer holds the CLIP model in a class-level singleton and caches
analysis results on disk. Neither was keyed on the checkpoint, so setting
CLIP_MODEL was silently ignored twice over: the first analyzer to run pinned
the model for the whole process, and the on-disk cache handed back the previous
model's profile as a hit.
"""

import sys
import tempfile
import types
from pathlib import Path

import pytest

# Imported before the torch stub exists: sklearn/scipy probe sys.modules for
# torch at import time and a partial stub confuses them.
from src.deep_style_analyzer import DeepStyleAnalyzer


@pytest.fixture
def clip(monkeypatch):
    """Stub the lazily-imported torch/transformers and reset the singleton."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)

    requested: list[str] = []

    class FakeCLIP:
        def __init__(self, name): self.name = name
        def to(self, device): return self
        def eval(self): return self

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            requested.append(name)
            return cls(name)

    transformers = types.ModuleType("transformers")
    transformers.CLIPModel = FakeCLIP
    transformers.CLIPProcessor = FakeCLIP
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    for attr in ("_model", "_processor", "_device", "_loaded_model_name"):
        monkeypatch.setattr(DeepStyleAnalyzer, attr, None, raising=False)

    refs = Path(tempfile.mkdtemp())
    (refs / "a.png").write_bytes(b"fake")
    (refs / "b.png").write_bytes(b"fake2")
    return types.SimpleNamespace(requested=requested, refs=refs)


class TestModelSingleton:
    def test_switching_model_reloads(self, clip):
        DeepStyleAnalyzer(clip.refs, model_name="clip-BASE")._ensure_model()
        DeepStyleAnalyzer(clip.refs, model_name="clip-LARGE")._ensure_model()
        assert DeepStyleAnalyzer._model.name == "clip-LARGE"
        assert "clip-LARGE" in clip.requested

    def test_same_model_is_not_reloaded(self, clip):
        """The singleton must still do its job — loading CLIP is expensive."""
        DeepStyleAnalyzer(clip.refs, model_name="clip-BASE")._ensure_model()
        before = len(clip.requested)
        DeepStyleAnalyzer(clip.refs, model_name="clip-BASE")._ensure_model()
        DeepStyleAnalyzer(clip.refs, model_name="clip-BASE")._ensure_model()
        assert len(clip.requested) == before, "reloaded a model it already held"


class TestCacheKey:
    @staticmethod
    def _pair(refs, cache):
        a = DeepStyleAnalyzer(refs, model_name="clip-BASE", cache_dir=cache)
        b = DeepStyleAnalyzer(refs, model_name="clip-LARGE", cache_dir=cache)
        return a, b

    def test_different_models_do_not_share_a_cache_entry(self, clip, tmp_path):
        a, b = self._pair(clip.refs, tmp_path)
        paths = a._get_image_paths()
        profile = DeepStyleAnalyzer._default_profile()
        profile.layout_type = "written-by-BASE"
        a._save_cache(paths, profile)
        assert b._load_cache(paths) is None, "read back another model's profile"

    def test_fingerprint_includes_the_model(self, clip, tmp_path):
        a, b = self._pair(clip.refs, tmp_path)
        paths = a._get_image_paths()
        assert a._cache_fingerprint(paths) != b._cache_fingerprint(paths)

    def test_same_model_still_hits(self, clip, tmp_path):
        """Keying on the model must not disable caching altogether."""
        a = DeepStyleAnalyzer(clip.refs, model_name="clip-BASE", cache_dir=tmp_path)
        paths = a._get_image_paths()
        profile = DeepStyleAnalyzer._default_profile()
        profile.layout_type = "cached-layout"
        a._save_cache(paths, profile)

        again = DeepStyleAnalyzer(clip.refs, model_name="clip-BASE", cache_dir=tmp_path)
        hit = again._load_cache(paths)
        assert hit is not None and hit.layout_type == "cached-layout"

    def test_changed_images_still_miss(self, clip, tmp_path):
        a = DeepStyleAnalyzer(clip.refs, model_name="clip-BASE", cache_dir=tmp_path)
        paths = a._get_image_paths()
        a._save_cache(paths, DeepStyleAnalyzer._default_profile())
        (clip.refs / "c.png").write_bytes(b"new image")
        assert a._load_cache(a._get_image_paths()) is None
