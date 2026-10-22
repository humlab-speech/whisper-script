"""Tests for system-junk filtering in the whisper-script pipeline.

Uses stdlib unittest only (no pytest). Run from the whisper-script dir with the
script's own venv (importing run_whisper pulls in tqdm):

    .venv/bin/python tests/test_junk_files.py -v
"""

import logging
import shutil
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


class IsJunkDirTests(unittest.TestCase):
    def test_known_os_metadata_dirs_are_junk(self):
        for name in (".Trashes", ".Spotlight-V100", ".fseventsd", ".DS_Store", "._payload", "~$work"):
            with self.subTest(name=name):
                self.assertTrue(run_whisper.is_junk_dir(name))

    def test_hidden_user_dirs_are_not_junk(self):
        # Directory policy is an explicit deny-list: unknown hidden dirs may hold
        # user data and must never be silently dropped from a scan.
        for name in (".recordings", ".private", "2024.V1", "10.23 meeting"):
            with self.subTest(name=name):
                self.assertFalse(run_whisper.is_junk_dir(name))


class CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def at_level(self, levelno):
        return [r for r in self.records if r.levelno == levelno]

    def at_or_above(self, levelno):
        return [r for r in self.records if r.levelno >= levelno]


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
        (self.root / ".recordings").mkdir()  # unknown hidden dir = user data, must survive
        self.root.joinpath(".recordings/song.mp3").write_bytes(b"x")
        (self.root / "2024.V1").mkdir()  # dotted-but-not-hidden dir component
        self.root.joinpath("2024.V1/meeting.wav").write_bytes(b"x")
        self.handler = CaptureLogs()
        logging.getLogger().addHandler(self.handler)

    def tearDown(self):
        logging.getLogger().removeHandler(self.handler)

    def test_recursive_skips_junk_keeps_user_data(self):
        found = run_whisper.find_convertible_files(self.root, EXTENSIONS, recursive=True)
        names = {p.relative_to(self.root).as_posix() for p in found}
        self.assertEqual(
            names,
            {
                "real.mp3",
                "sub/deep.flac",
                ".recordings/song.mp3",
                "2024.V1/meeting.wav",
            },
        )

    def test_non_recursive_skips_all_junk(self):
        found = run_whisper.find_convertible_files(self.root, EXTENSIONS, recursive=False)
        names = {p.name for p in found}
        self.assertEqual(names, {"real.mp3"})

    def test_scan_reports_junk_skips_at_info_level(self):
        run_whisper.find_convertible_files(self.root, EXTENSIONS, recursive=True)
        infos = [r.getMessage() for r in self.handler.at_level(logging.INFO)]
        self.assertTrue(
            any("system/junk" in m for m in infos),
            f"expected an INFO junk tally, got: {infos}",
        )

    def test_dot_scan_root_itself_is_never_filtered(self):
        dot_root = Path(tempfile.mkdtemp()) / ".private"
        dot_root.mkdir()
        dot_root.joinpath("song.mp3").write_bytes(b"x")
        dot_root.joinpath("._song.mp3").write_bytes(b"x")
        found = run_whisper.find_convertible_files(dot_root, EXTENSIONS, recursive=True)
        self.assertEqual([p.name for p in found], ["song.mp3"])


class ConvertSafetyNetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.junk_source = self.tmp / "._ghost.mp3"
        self.junk_source.write_bytes(b"\x00\x05\x16\x07")  # real AppleDouble magic header
        self.real_source = self.tmp / "interview.mp3"
        self.real_source.write_bytes(b"ID3fakeaudio")
        self.target = self.tmp / "out" / "ghost.wav"
        self.handler = CaptureLogs()
        logging.getLogger().addHandler(self.handler)
        self.saved_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)

    def tearDown(self):
        logging.getLogger().removeHandler(self.handler)
        logging.getLogger().setLevel(self.saved_level)

    def _convert(self, source, returncode=1, stderr=b"boom"):
        fake = SimpleNamespace(returncode=returncode, stderr=stderr, stdout=b"")
        with mock.patch.object(run_whisper.subprocess, "run", return_value=fake) as run:
            result = run_whisper.convert_audio_to_wav_file(source, self.target)
        return result, run.call_args[0][0]

    def test_junk_ffmpeg_failure_downgraded_to_single_warning(self):
        stderr = b"[mp3 @ 0x1] Invalid data found when processing input\nError opening input file.\n"
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_bytes(b"partial")  # ffmpeg left a partial file: cleanup must fire
        result, _cmd = self._convert(self.junk_source, 1, stderr)
        self.assertFalse(result)
        warnings = self.handler.at_level(logging.WARNING)
        self.assertEqual(len(warnings), 1, f"expected exactly one WARNING, got {self.handler.records}")
        self.assertIn("Skipped", warnings[0].getMessage())
        self.assertIn(str(self.junk_source), warnings[0].getMessage())
        self.assertFalse(self.target.exists(), "partial output file should be cleaned up")
        self.assertEqual(self.handler.at_or_above(logging.ERROR), [])

    def test_corrupt_real_file_is_not_labeled_junk(self):
        # Same ffmpeg message, real filename: truncation must stay a full ERROR,
        # because the user action is recovery, not ignoring.
        stderr = b"[mp3 @ 0x1] Invalid data found when processing input\n"
        result, _cmd = self._convert(self.real_source, 1, stderr)
        self.assertFalse(result)
        self.assertEqual(self.handler.at_level(logging.WARNING), [])
        errors = [r.getMessage() for r in self.handler.at_or_above(logging.ERROR)]
        self.assertTrue(any("FFmpeg error" in m for m in errors), errors)
        self.assertTrue(any("Invalid data found" in m for m in errors), "full stderr must stay in ERROR log")

    def test_junk_name_with_real_error_stays_error(self):
        result, _cmd = self._convert(self.junk_source, 1, b"Unknown encoder 'opus_surprise'")
        self.assertFalse(result)
        errors = [r.getMessage() for r in self.handler.at_or_above(logging.ERROR)]
        self.assertTrue(any("FFmpeg error" in m for m in errors), errors)

    def test_empty_stderr_does_not_crash_warning_branch(self):
        result, _cmd = self._convert(self.junk_source, returncode=0, stderr=b"   ")
        self.assertTrue(result)  # rc=0 success path unaffected by blank stderr

    def test_ffmpeg_command_suppresses_version_banner(self):
        _result, cmd = self._convert(self.real_source, returncode=0, stderr=b"")
        self.assertIn("-hide_banner", cmd)
        i = cmd.index("-loglevel")
        self.assertEqual(cmd[i + 1], "error")
        self.assertLess(cmd.index("-hide_banner"), cmd.index("-i"))
        self.assertLess(i, cmd.index("-i"))


@unittest.skipUnless(shutil.which("ffmpeg"), "real ffmpeg not available")
class RealFfmpegTests(unittest.TestCase):
    """Guard the flags x patterns interaction the mocks cannot cover:
    real ffmpeg on a real AppleDouble file must produce an *error-level* line
    that JUNK_FFMPEG_PATTERNS actually matches under -loglevel error."""

    def test_appledouble_file_downgrades_to_single_warning(self):
        tmp = Path(tempfile.mkdtemp())
        source = tmp / "._ghost.mp3"
        # Valid AppleDouble header + padding: ffmpeg sees pure garbage
        source.write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 512)
        target = tmp / "ghost.wav"

        handler = CaptureLogs()
        logging.getLogger().addHandler(handler)
        try:
            ok = run_whisper.convert_audio_to_wav_file(source, target)
        finally:
            logging.getLogger().removeHandler(handler)

        self.assertFalse(ok)
        loud = handler.at_or_above(logging.WARNING)
        self.assertEqual(len(loud), 1, f"expected one log line for the junk file, got: {loud}")
        self.assertEqual(loud[0].levelno, logging.WARNING)
        self.assertIn("Skipped", loud[0].getMessage())
        self.assertNotIn("ffmpeg version", loud[0].getMessage(), "version banner must be suppressed")


class ProcessSourceFileIntegrationTests(unittest.TestCase):
    def test_dry_run_over_scanned_files_never_sees_junk(self):
        root = Path(tempfile.mkdtemp())
        raw = root / "raw_audio"
        raw.mkdir()
        raw.joinpath("real.mp3").write_bytes(b"x")
        raw.joinpath("._real.mp3").write_bytes(b"x")
        (raw / ".Trashes").mkdir()
        raw.joinpath(".Trashes/deleted.mp3").write_bytes(b"x")

        files = run_whisper.find_convertible_files(raw, EXTENSIONS, recursive=True)
        args = SimpleNamespace(force_convert=False, dry_run=True, max_chunk_duration=None)
        summary = {
            "files_converted": 0,
            "conversion_skipped_exists": 0,
            "conversion_failed": 0,
        }
        wavs, path_map, chunk_map = [], {}, {}
        for f in files:
            run_whisper.process_source_file(f, raw, root / "converted_wavs", args, summary, wavs, path_map, chunk_map)

        self.assertEqual([p.name for p in wavs], ["real.wav"])
        self.assertEqual(summary["conversion_failed"], 0)


if __name__ == "__main__":
    unittest.main()
