"""Tests for ImageProcessor."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.image_processor import ImageProcessor


class TestSmartResize:
    def setup_method(self):
        self.proc = ImageProcessor()

    def test_square_to_square(self):
        img = Image.new("RGB", (500, 500), "red")
        result = self.proc.resize_for_instagram(img, "feed")
        assert result.size == (1080, 1080)

    def test_wide_to_square_crops_width(self):
        img = Image.new("RGB", (2000, 1000), "blue")
        result = self.proc.resize_for_instagram(img, "feed")
        assert result.size == (1080, 1080)

    def test_tall_to_square_crops_height(self):
        img = Image.new("RGB", (1000, 2000), "green")
        result = self.proc.resize_for_instagram(img, "feed")
        assert result.size == (1080, 1080)

    def test_portrait_size(self):
        img = Image.new("RGB", (800, 1000), "white")
        result = self.proc.resize_for_instagram(img, "portrait")
        assert result.size == (1080, 1350)

    def test_story_size(self):
        img = Image.new("RGB", (500, 900), "black")
        result = self.proc.resize_for_instagram(img, "story")
        assert result.size == (1080, 1920)

    def test_landscape_size(self):
        img = Image.new("RGB", (1200, 600), "gray")
        result = self.proc.resize_for_instagram(img, "landscape")
        assert result.size == (1080, 608)


class TestEnhanceImage:
    def test_enhancement_returns_same_size(self):
        proc = ImageProcessor()
        img = Image.new("RGB", (100, 100), "red")
        result = proc.enhance_image(img)
        assert result.size == (100, 100)

    def test_custom_parameters(self):
        proc = ImageProcessor()
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        result = proc.enhance_image(img, sharpness=2.0, contrast=1.5, saturation=0.5)
        assert result.size == (100, 100)


class TestColorFilter:
    def test_apply_with_palette(self):
        proc = ImageProcessor()
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = proc.apply_color_filter(img, ["#FF0000", "#00FF00"], strength=0.1)
        assert result.size == (50, 50)

    def test_empty_palette_returns_unchanged(self):
        proc = ImageProcessor()
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = proc.apply_color_filter(img, [])
        assert result is img


class TestTextOverlay:
    def test_overlay_returns_correct_size(self):
        proc = ImageProcessor()
        img = Image.new("RGB", (500, 500), "white")
        result = proc.add_text_overlay(img, "Test text")
        assert result.size == (500, 500)

    def test_overlay_positions(self):
        proc = ImageProcessor()
        img = Image.new("RGB", (500, 500), "white")
        for pos in ("top", "bottom", "center"):
            result = proc.add_text_overlay(img, "Hello", position=pos)
            assert result.size == (500, 500)


class TestSave:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = patch("src.config.OUTPUT_DIR", Path(self._tmpdir))
        self._patcher.start()
        self.proc = ImageProcessor()

    def teardown_method(self):
        self._patcher.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_jpg(self):
        img = Image.new("RGB", (100, 100), "red")
        path = self.proc.save_image(img, "test.jpg")
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_save_png(self):
        img = Image.new("RGB", (100, 100), "blue")
        path = self.proc.save_image(img, "test.png")
        assert path.exists()
        assert path.suffix == ".png"

    def test_save_auto_name(self):
        img = Image.new("RGB", (100, 100), "green")
        path = self.proc.save_image(img)
        assert path.exists()

    def test_save_batch(self):
        images = [Image.new("RGB", (100, 100), c) for c in ("red", "green", "blue")]
        paths = self.proc.save_batch(images)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
