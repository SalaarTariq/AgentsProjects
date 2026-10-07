"""Pressing 's' twice must leave two snapshots, not overwrite the first.

Both visualisers built the filename as "<prefix>_<int(unix seconds)>.png". That
only resolves to the second, so at 30 fps two presses a few frames apart produced
the same name: the second write replaced the first while the HUD and stdout both
reported "Saved <name>". The same collision also clobbered snapshots left over
from an earlier run that happened to land on the same second.
"""

import os

import pytest

import HandGesture
import HandMagic

MODULES = pytest.mark.parametrize(
    "mod", [HandGesture, HandMagic], ids=["HandGesture", "HandMagic"]
)

NOW = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@MODULES
class TestUniqueNames:
    def test_first_press_uses_the_plain_name(self, mod):
        assert mod.next_screenshot_path("shot", NOW) == f"shot_{int(NOW)}.png"

    def test_presses_in_the_same_second_do_not_collide(self, mod):
        """Three presses a few frames apart — all within one second."""
        names = []
        for offset in (0.0, 0.10, 0.25):
            name = mod.next_screenshot_path("shot", NOW + offset)
            open(name, "wb").write(b"frame")      # the save that would follow
            names.append(name)
        assert len(set(names)) == 3, f"collided: {names}"
        assert sorted(os.listdir(".")) == sorted(names)

    def test_existing_file_from_an_earlier_run_is_not_clobbered(self, mod):
        first = f"shot_{int(NOW)}.png"
        open(first, "wb").write(b"earlier run")
        second = mod.next_screenshot_path("shot", NOW)
        assert second != first
        assert open(first, "rb").read() == b"earlier run"

    def test_counter_keeps_climbing(self, mod):
        for _ in range(5):
            open(mod.next_screenshot_path("shot", NOW), "wb").write(b"x")
        assert len(os.listdir(".")) == 5

    def test_prefix_is_honoured(self, mod):
        assert mod.next_screenshot_path("other", NOW).startswith("other_")


@MODULES
class TestFailedSaveIsReported:
    """imwrite returns False rather than raising, so the result must be checked."""

    def test_save_block_checks_the_return_value(self, mod):
        import inspect

        src = inspect.getsource(mod.main)
        assert "cv2.imwrite" in src
        assert "save_ok" in src, "imwrite result is ignored"

    def test_failure_message_differs_from_success(self, mod):
        import inspect

        src = inspect.getsource(mod.main)
        assert "Could not save" in src
