#!/usr/bin/env python3
"""Strip speaker tags ([SPEAKER_XX]: ) from .srt, .txt, and .vtt files.

Reads the given files (or all matching files in a folder), removes
diarization speaker labels, and writes cleaned copies alongside the
originals with a ``.no_speakers`` suffix.  The original files are never
modified.  Files that already have ``.no_speakers`` in the name and files
without any speaker tags are skipped.

Usage:
    python strip_speakers.py output/recording.srt
    python strip_speakers.py output/my_transcription/
    python strip_speakers.py file1.srt file2.txt output/folder/
    python strip_speakers.py output/ --output-dir /tmp/cleaned/
"""

import argparse
import re
import sys
from pathlib import Path

# Matches [SPEAKER_00]:  (with the optional trailing space)
_SPEAKER_RE = re.compile(r"\[SPEAKER_\d+\]:\s?")
_EXTENSIONS = {".srt", ".txt", ".vtt"}


def strip_speakers(text: str) -> str:
    """Remove all ``[SPEAKER_XX]: `` tags from *text*."""
    return _SPEAKER_RE.sub("", text)


def _write_stripped(src: Path, dest_dir: Path) -> Path | None:
    """Strip speaker tags from *src* and write to *dest_dir*.

    Returns the destination path, or ``None`` if the file had no speaker tags.
    """
    content = src.read_text(encoding="utf-8")
    if not _SPEAKER_RE.search(content):
        return None
    cleaned = strip_speakers(content)
    dest = dest_dir / f"{src.stem}.no_speakers{src.suffix}"
    dest.write_text(cleaned, encoding="utf-8")
    return dest


def _collect_targets(paths: list[str]) -> list[Path]:
    """Expand a mix of files and directories into a flat list of targets."""
    targets: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in sorted(_EXTENSIONS):
                targets.extend(sorted(path.rglob(f"*{ext}")))
        elif path.is_file():
            targets.append(path)
        else:
            print(f"Not found: {path}", file=sys.stderr)
    # Exclude files that are already stripped variants
    return [t for t in targets if ".no_speakers" not in t.stem]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip [SPEAKER_XX] tags from transcription files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="File(s) or folder(s) to process (.srt, .txt, .vtt)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="Write stripped files here (default: same directory as source)",
    )
    args = parser.parse_args()

    targets = _collect_targets(args.paths)
    if not targets:
        sys.exit("No .srt/.txt/.vtt files found")

    count = 0
    for src in targets:
        dest_dir = Path(args.output_dir) if args.output_dir else src.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _write_stripped(src, dest_dir)
        if dest:
            print(f"  {src} → {dest.name}")
            count += 1
        else:
            print(f"  {src}  (no speaker tags, skipped)")

    print(f"\n{count} file(s) written")


if __name__ == "__main__":
    main()
