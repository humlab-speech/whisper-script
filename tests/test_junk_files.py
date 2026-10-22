"""Tests for system-junk filtering in run_whisper.py (AppleDouble ._ files etc.).

Stdlib-only: run with  .venv/bin/python tests/test_junk_files.py  from the whisper-script dir.
"""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_whisper  # noqa: E402

EXTENSIONS = (".mp3", ".flac", ".wav")


class IsJunkFileTests(unittest.TestCase):
    def test_apple_and_windows_artifacts_are_junk(self):
        junk = [
            "._ZOOM0020.mp3",  # macOS AppleDouble resource fork
            ".__ZOOM0020.mp3",  # AppleDouble older variant
            ".DS_Store",
            ".hidden.mp3",
            ".Trashes",
            "~$ZOOM0020.mp3",  # Windows Office lock file
        ]
        for name in junk:
            with self.subTest(name=name):
                self.assertTrue(run_whisper.is_junk_file(name))

    def test_real_audio_files_are_not_junk(self):
        real = ["ZOOM0020.mp3", "track-01.flac", "interview.wav", "my.song.mp3", "track._mp3"]
        for name in real:
            with self.subTest(name=name):
                self.assertFalse(run_whisper.is_junk_file(name))


class FindConvertibleFilesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.root.joinpath("real.mp3").write_bytes(b"x")
        self.root.joinpath("._real.mp3").write_bytes(b"x")  # AppleDouble twin of real.mp3
        self.root.joinpath(".__older.mp3").write_bytes(b"x")
        self.root.joinpath(".hidden.mp3").write_bytes(b"x")
        self.root.joinpath(".DS_Store").write_bytes(b"x")
        self.root.joinpath("~$lock.mp3").write_bytes(b"x")
        self.root.joinpath("notes.txt").write_bytes(b"x")  # right junk, wrong extension anyway
        (self.root / "sub").mkdir()
        self.root.joinpath("sub/deep.flac").write_bytes(b"x")
        self.root.joinpath("sub/._deep.flac").write_bytes(b"x")
        (self.root / ".Trashes").mkdir()
        self.root.joinpath(".Trashes/deleted.mp3").write_bytes(b"x")

    def test_recursive_skips_all_junk(self):
        found = run_whisper.find_convertible_files(self.root, EXTENSIONS, recursive=True)
        names = {p.relative_to(self.root).as_posix() for p in found}
        self.assertEqual(names, {"real.mp3", "sub/deep.flac"})

    def test_non_recursive_skips_all_junk(self):
        found = run_whisper.find_convertible_files(self.root, EXTENSIONS, recursive=False)
        names = {p.name for p in found}
        self.assertEqual(names, {"real.mp3"})


class CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def text(self):
        return "\n".join(f"{r.levelname}:{r.getMessage()}" for r in self.records)


class ConvertSafetyNetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "._ghost.mp3"
        self.source.write_bytes(b"\x00\x05\x16\x07")  # AppleDouble magic header
        self.target = self.tmp / "out" / "ghost.wav"
        self.handler = CaptureLogs()
        logging.getLogger().addHandler(self.handler)
        self.saved_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)

    def tearDown(self):
        logging.getLogger().removeHandler(self.handler)
        logging.getLogger().setLevel(self.saved_level)

    def _convert(self, returncode, stderr):
        fake = SimpleNamespace(returncode=returncode, stderr=stderr, stdout=b"")
        with mock.patch.object(run_whisper.subprocess, "run", return_value=fake) as run:
            result = run_whisper.convert_audio_to_wav_file(self.source, self.target)
        return result, run.call_args[0][0]

    def test_junk_ffmpeg_failure_downgraded_to_warning(self):
        stderr = b"[mp3 @ 0x1] Invalid data found when processing input\n" b"misdetection possible !!\n"
        result, _cmd = self._convert(1, stderr)
        self.assertFalse(result)
        logged = self.handler.text()
        self.assertIn("WARNING", logged)
        self.assertIn("Skipped", logged)
        self.assertNotIn("ERROR", logged)

    def test_real_ffmpeg_failure_still_logs_error(self):
        result, _cmd = self._convert(1, b"Unknown encoder 'opus_surprise'")
        self.assertFalse(result)
        logged = self.handler.text()
        self.assertIn("ERROR", logged)
        self.assertIn("FFmpeg error", logged)

    def test_ffmpeg_command_suppresses_version_banner(self):
        _result, cmd = self._convert(0, b"")
        self.assertIn("-hide_banner", cmd)
        i = cmd.index("-loglevel")
        self.assertEqual(cmd[i + 1], "error")
        # both global options must appear before the first input file
        self.assertLess(cmd.index("-hide_banner"), cmd.index("-i"))
        self.assertLess(i, cmd.index("-i"))

    def test_successful_conversion_returns_true(self):
        result, _cmd = self._convert(0, b"")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
