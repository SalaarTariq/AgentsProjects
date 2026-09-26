"""Caption files must round-trip emoji regardless of the platform's locale.

generate_caption_node asks the LLM for "an engaging Instagram caption", which
comes back with emoji far more often than not. save_images_node then writes it
next to each image. With no explicit encoding that write picks up the platform
default -- cp1252 on a stock Windows install -- and dies on the first emoji,
after the images are already on disk.

Note on running these: on a macOS/Linux box the default encoding is already
utf-8, so an unqualified write_text succeeds and these pass either way. To
actually exercise the defect, run the suite the way CI should:

    PYTHONWARNDEFAULTENCODING=1 python -W error::EncodingWarning -m pytest

which turns every implicit-encoding call into a hard failure regardless of
the host locale.
"""

import tempfile
from pathlib import Path

import pytest

# A caption shaped like real LLM output: prose, emoji, blank line, hashtags.
EMOJI_CAPTION = (
    "Golden hour hits different ✨ Tag someone who needs this view today! "
    "\U0001F31E\U0001F4AB\n\n"
    "#goldenhour #sunsetlover #instagood #photography #vibes"
)


@pytest.fixture
def outdir(monkeypatch):
    d = Path(tempfile.mkdtemp())
    from src import config
    monkeypatch.setattr(config, "OUTPUT_DIR", d)
    return d


class TestCaptionFileEncoding:
    def test_emoji_caption_is_written_and_reads_back(self, outdir, monkeypatch):
        import src.nodes as nodes

        class FakeProcessor:
            def save_batch(self, images, prefix="post"):
                p = outdir / f"{prefix}_1.jpg"
                p.write_bytes(b"\xff\xd8\xff")
                return [p]

        monkeypatch.setattr(nodes, "_processor", FakeProcessor())

        result = nodes.save_images_node(
            {"processed_images": ["stub"], "caption": EMOJI_CAPTION}
        )

        caption_file = result["saved_paths"][0].with_suffix(".txt")
        assert caption_file.exists()
        assert caption_file.read_text(encoding="utf-8") == EMOJI_CAPTION

    def test_caption_bytes_are_utf8_not_locale_dependent(self, outdir, monkeypatch):
        """The file on disk must be utf-8 bytes whatever the machine's locale is."""
        import src.nodes as nodes

        class FakeProcessor:
            def save_batch(self, images, prefix="post"):
                p = outdir / f"{prefix}_1.jpg"
                p.write_bytes(b"\xff\xd8\xff")
                return [p]

        monkeypatch.setattr(nodes, "_processor", FakeProcessor())
        result = nodes.save_images_node(
            {"processed_images": ["stub"], "caption": EMOJI_CAPTION}
        )

        raw = result["saved_paths"][0].with_suffix(".txt").read_bytes()
        assert raw == EMOJI_CAPTION.encode("utf-8")
        assert "✨".encode("utf-8") in raw


class TestDeepStyleCacheEncoding:
    def test_cache_round_trips_non_ascii(self, tmp_path):
        """The cache read/write pair must agree on an encoding, not on a locale."""
        import json

        cache = tmp_path / "deep_style_cache.json"
        payload = {"fingerprint": "abc", "mood": ["sérieux", "café ☕"]}

        cache.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        assert json.loads(cache.read_text(encoding="utf-8")) == payload
