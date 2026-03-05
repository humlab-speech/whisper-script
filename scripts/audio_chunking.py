#!/usr/bin/env python3

import json
import logging
import re
import subprocess
from pathlib import Path


def parse_duration(duration_str: str) -> int:
    """Parse duration string like '20m', '1h30m', '30s' into seconds."""
    duration_str = duration_str.strip().lower()
    total_seconds = 0

    # Match patterns like 1h, 20m, 30s
    pattern = r"(\d+)([hms])"
    matches = re.findall(pattern, duration_str)

    if not matches:
        raise ValueError(f"Invalid duration format: {duration_str}. Use formats like '20m', '1h30m', '30s'")

    for value, unit in matches:
        value = int(value)
        if unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "s":
            total_seconds += value

    return total_seconds


def get_audio_duration(audio_file: Path) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logging.error(f"Could not determine duration of {audio_file}: {e}")
        raise


def split_audio_into_chunks(
    source_file: Path,
    target_dir: Path,
    max_chunk_duration_seconds: int,
    chunk_naming_pattern: str = "{stem}_chunk_{index:03d}{suffix}",
) -> list[Path]:
    """
    Split audio file into chunks using ffmpeg.

    Args:
        source_file: Path to source audio file
        target_dir: Directory to write chunks to
        max_chunk_duration_seconds: Maximum duration per chunk in seconds
        chunk_naming_pattern: Format string for chunk filenames

    Returns:
        List of created chunk file paths, in order
    """
    if not source_file.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_file}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Get total duration
    total_duration = get_audio_duration(source_file)
    num_chunks = (int(total_duration) + max_chunk_duration_seconds - 1) // max_chunk_duration_seconds

    if num_chunks <= 1:
        logging.info(
            f"Audio duration ({total_duration:.1f}s) <= max chunk duration"
            f" ({max_chunk_duration_seconds}s). No splitting needed."
        )
        return [source_file]

    logging.info(f"Splitting {source_file.name} into {num_chunks} chunks of ~{max_chunk_duration_seconds}s each")

    chunk_files = []

    for i in range(num_chunks):
        start_time = i * max_chunk_duration_seconds
        chunk_name = chunk_naming_pattern.format(stem=source_file.stem, index=i, suffix=source_file.suffix)
        chunk_file = target_dir / chunk_name

        cmd = [
            "ffmpeg",
            "-i",
            str(source_file),
            "-ss",
            str(start_time),
            "-t",
            str(max_chunk_duration_seconds),
            "-c",
            "copy",  # Copy codec for speed
            "-y",
            str(chunk_file),
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            chunk_files.append(chunk_file)
            logging.info(f"Created chunk {i}: {chunk_file.name}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create chunk {i}: {e}")
            raise

    return chunk_files


def merge_srt_files(srt_files: list[Path], output_file: Path, gap_ms: int = 0) -> None:
    """
    Merge multiple SRT files into one, adjusting timestamps so they're continuous.

    Args:
        srt_files: List of SRT file paths in order
        output_file: Path to write merged SRT
        gap_ms: Gap in milliseconds between chunks (default 0)
    """
    try:
        import pysrt
    except ImportError:
        logging.error("pysrt not installed. Install with: pip install pysrt")
        raise

    merged = pysrt.SubRipFile()
    cumulative_ms = 0
    next_index = 1

    for srt_file in srt_files:
        if not srt_file.exists():
            logging.warning(f"SRT file not found, skipping: {srt_file}")
            continue

        try:
            subs = pysrt.open(str(srt_file))

            # Shift timestamps
            for sub in subs:
                # Convert pysrt time to milliseconds, add offset, convert back
                start_ms = int(sub.start.ordinal) + cumulative_ms
                end_ms = int(sub.end.ordinal) + cumulative_ms

                sub.start = pysrt.SubRipTime(milliseconds=start_ms)
                sub.end = pysrt.SubRipTime(milliseconds=end_ms)
                sub.index = next_index
                next_index += 1
                merged.append(sub)

            # Update cumulative time: end of last subtitle + gap
            if subs:
                cumulative_ms = int(subs[-1].end.ordinal) + gap_ms
                logging.info(f"Merged {srt_file.name} (shifted by {cumulative_ms - gap_ms}ms)")
        except Exception as e:
            logging.error(f"Error merging SRT file {srt_file}: {e}")
            raise

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(output_file))
    logging.info(f"Saved merged SRT to {output_file}")


def merge_txt_files(txt_files: list[Path], output_file: Path, separator: str = "\n") -> None:
    """
    Merge multiple TXT files into one.

    Args:
        txt_files: List of TXT file paths in order
        output_file: Path to write merged TXT
        separator: Text to insert between files (default newline)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out:
        for i, txt_file in enumerate(txt_files):
            if not txt_file.exists():
                logging.warning(f"TXT file not found, skipping: {txt_file}")
                continue

            try:
                content = txt_file.read_text(encoding="utf-8")
                out.write(content)
                if i < len(txt_files) - 1:
                    out.write(separator)
                logging.info(f"Merged {txt_file.name}")
            except Exception as e:
                logging.error(f"Error merging TXT file {txt_file}: {e}")
                raise

    logging.info(f"Saved merged TXT to {output_file}")


def load_chunk_map(chunk_map_file: Path) -> dict:
    """Load the chunk mapping JSON file."""
    if not chunk_map_file.exists():
        return {}
    with open(chunk_map_file, "r") as f:
        return json.load(f)


def save_chunk_map(chunk_map_file: Path, mapping: dict) -> None:
    """Save the chunk mapping to JSON file."""
    chunk_map_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunk_map_file, "w") as f:
        json.dump(mapping, f, indent=2)


def merge_results_for_original_files(
    transcriptions_base: Path,
    chunk_map_file: Path,
) -> None:
    """
    After all transcriptions are done, merge results back to original file names.

    Args:
        transcriptions_base: Root transcriptions directory
        chunk_map_file: Path to the chunk mapping JSON
    """
    chunk_map = load_chunk_map(chunk_map_file)
    if not chunk_map:
        logging.info("No chunk mapping found; skipping merge phase.")
        return

    logging.info("Starting post-processing merge of chunked transcription results...")

    for original_name, chunk_info in chunk_map.items():
        chunks = chunk_info.get("chunks", [])

        if not chunks:
            continue

        # Find SRT and TXT files for each chunk
        srt_files = []
        txt_files = []

        for chunk in chunks:
            chunk_stem = Path(chunk).stem
            # Search recursively under transcriptions_base for matching files
            for srt in transcriptions_base.rglob(f"{chunk_stem}.srt"):
                srt_files.append(srt)
            for txt in transcriptions_base.rglob(f"{chunk_stem}.txt"):
                txt_files.append(txt)

        if srt_files or txt_files:
            # Determine output directory from first SRT/TXT file found
            # (they should all be in the same directory structure)
            output_dir = None
            if srt_files:
                # Get the parent directory of the first SRT file
                output_dir = srt_files[0].parent
            elif txt_files:
                # Get the parent directory of the first TXT file
                output_dir = txt_files[0].parent

            if output_dir:
                # Merge SRT
                if srt_files:
                    srt_files.sort()  # Ensure order
                    merged_srt = output_dir / f"{original_name}.srt"
                    try:
                        merge_srt_files(srt_files, merged_srt)
                    except Exception as e:
                        logging.error(f"Failed to merge SRT files for {original_name}: {e}")

                # Merge TXT
                if txt_files:
                    txt_files.sort()  # Ensure order
                    merged_txt = output_dir / f"{original_name}.txt"
                    try:
                        merge_txt_files(txt_files, merged_txt)
                    except Exception as e:
                        logging.error(f"Failed to merge TXT files for {original_name}: {e}")

                logging.info(f"Completed merge for original file: {original_name} in {output_dir}")
