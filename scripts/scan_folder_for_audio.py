import argparse
import csv
import json
import os
import subprocess
import sys

from tqdm import tqdm  # For the progress bar


def check_ffmpeg_installed():
    """
    Checks if ffmpeg (and ffprobe) is installed and accessible.
    Exits the script if not found.
    """
    try:
        process = subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if process.returncode != 0:
            tqdm.write("Error: ffprobe (part of ffmpeg) not found or not working correctly.", file=sys.stderr)
            tqdm.write("Please ensure ffmpeg is installed and in your system's PATH.", file=sys.stderr)
            if process.stderr:
                tqdm.write(f"Stderr: {process.stderr.decode('utf-8', errors='ignore')}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        tqdm.write("Error: ffprobe (part of ffmpeg) command not found.", file=sys.stderr)
        tqdm.write("Please ensure ffmpeg is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)


def get_audio_duration_ffmpeg(filepath):
    """
    Tries to get the duration of an audio file in seconds using ffprobe.
    Returns duration (float) or None if it can't be determined.
    Prints errors using tqdm.write for better console handling with the progress bar.
    """
    command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filepath]
    try:
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, encoding="utf-8"
        )
        metadata = json.loads(process.stdout)
        duration_str = None
        if "format" in metadata and "duration" in metadata["format"]:
            duration_str = metadata["format"]["duration"]
        elif "streams" in metadata:
            for stream in metadata["streams"]:
                if stream.get("codec_type") == "audio" and "duration" in stream:
                    duration_str = stream["duration"]
                    break

        if duration_str is not None:
            try:
                return round(float(duration_str), 2)
            except ValueError:
                tqdm.write(
                    f"Warning: Could not parse duration '{duration_str}' as float for {os.path.basename(filepath)}",
                    file=sys.stderr,
                )
                return None
        else:
            # This might happen for non-media files or very corrupted ones
            # tqdm.write(f"Warning: Could not find duration info in ffprobe output for
            # {os.path.basename(filepath)}", file=sys.stderr)
            return None

    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else str(e)
        tqdm.write(f"ffprobe error for {os.path.basename(filepath)}: {stderr_msg}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        tqdm.write(f"Error parsing ffprobe JSON output for {os.path.basename(filepath)}", file=sys.stderr)
        return None
    except Exception as e:
        tqdm.write(f"An unexpected error occurred for {os.path.basename(filepath)}: {e}", file=sys.stderr)
        return None


def format_duration_display(total_seconds):
    """Formats total seconds into HH:MM:SS.ss string."""
    if total_seconds is None:
        return "00:00:00.00"
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def scan_and_process_files(folder_path, file_extension):
    """
    First scans for all matching files, then processes them with a progress bar.
    Returns a list of dictionaries, each containing file info.
    """
    audio_files_data = []
    filepaths_to_process = []
    aggregated_duration_seconds = 0.0
    files_with_errors = 0

    # Normalize extension
    if not file_extension.startswith("."):
        file_extension = "." + file_extension
    file_extension = file_extension.lower()

    abs_folder_path = os.path.abspath(folder_path)
    tqdm.write("Scanning for files...")
    for root, _, files in os.walk(abs_folder_path):
        for filename in files:
            if filename.lower().endswith(file_extension):
                filepaths_to_process.append(os.path.join(root, filename))

    if not filepaths_to_process:
        tqdm.write(f"No files found with extension '{file_extension}'.")
        return []

    tqdm.write(f"Found {len(filepaths_to_process)} files to process.")

    # Process files with progress bar
    # Initial postfix data
    postfix_data = {"Total Dur": format_duration_display(0), "Errors": 0}
    with tqdm(total=len(filepaths_to_process), unit="file", desc="Processing", postfix=postfix_data) as pbar:
        for full_path in filepaths_to_process:
            filename = os.path.basename(full_path)
            short_filename = (filename[:35] + "...") if len(filename) > 38 else filename
            pbar.set_description_str(f"File: {short_filename}")

            relative_path = os.path.relpath(full_path, abs_folder_path)
            duration = get_audio_duration_ffmpeg(full_path)

            if duration is not None:
                aggregated_duration_seconds += duration
            else:
                files_with_errors += 1

            audio_files_data.append(
                {
                    "FileName": filename,
                    "RelativePath": relative_path,
                    "Duration (s)": duration if duration is not None else "N/A",
                }
            )

            pbar.set_postfix_str(
                f"Total Dur: {format_duration_display(aggregated_duration_seconds)}, Errors: {files_with_errors}",
                refresh=True,
            )
            pbar.update(1)

    tqdm.write(
        f"\nProcessing complete. Total aggregated duration: {format_duration_display(aggregated_duration_seconds)}"
    )
    if files_with_errors > 0:
        tqdm.write(f"Encountered errors while processing {files_with_errors} file(s). Check messages above.")
    return audio_files_data


def save_to_csv(data, output_filepath):
    """
    Saves the collected data to a CSV file.
    """
    if not data:
        # This case is now handled before calling save_to_csv if no files are processed.
        # tqdm.write("No audio files data to save.", file=sys.stderr)
        return

    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            tqdm.write(f"Created output directory: {output_dir}")
        except OSError as e:
            tqdm.write(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
            return

    try:
        with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["FileName", "RelativePath", "Duration (s)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        tqdm.write(f"Successfully saved report to {output_filepath}")
    except IOError as e:
        tqdm.write(f"Error writing CSV file {output_filepath}: {e}", file=sys.stderr)


def main():
    check_ffmpeg_installed()

    parser = argparse.ArgumentParser(
        description="Parse a folder for audio files, get duration using ffmpeg, show progress, and save to CSV."
    )
    parser.add_argument("folder", help="The folder to scan for audio files.")
    parser.add_argument("extension", help="The file extension to look for (e.g., mp3, wav, .flac). Case-insensitive.")
    parser.add_argument(
        "-o",
        "--output",
        default="audio_report.csv",
        help="The name of the output CSV file (default: audio_report.csv).",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        tqdm.write(f"Error: Folder '{args.folder}' not found.", file=sys.stderr)
        sys.exit(1)

    tqdm.write(f"Target folder: {os.path.abspath(args.folder)}")
    tqdm.write(f"File extension: {args.extension}")
    tqdm.write(f"Output CSV: {os.path.abspath(args.output)}")

    audio_data = scan_and_process_files(args.folder, args.extension)

    if audio_data:
        output_filepath = os.path.abspath(args.output)
        save_to_csv(audio_data, output_filepath)
    else:
        tqdm.write("No audio data processed or found. CSV file not created.")


if __name__ == "__main__":
    main()
