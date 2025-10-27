#!/usr/bin/env python3

import argparse
import logging
import os  # Already imported, but making it explicit for clarity
import shutil  # For checking ffmpeg
import subprocess
from pathlib import Path

from tqdm import tqdm  # Add tqdm import

import WhisperTranscriber
from WhisperTranscriber.configuration_reader import ConfigurationReader

# --- Configuration ---
# Default extensions for conversion (can be overridden by command line)
DEFAULT_CONVERT_EXTENSIONS = (
    "wma,wmv,mp3,mp4,mkv,aac,flac,ogg,m4a,avi,mov,flv,mpeg,mpg,webm"  # Added common non-wav types
)
TARGET_WAV_EXTENSION = ".wav"

RAW_AUDIO_DIR_NAME = "raw_audio"
CONVERTED_WAVS_DIR_NAME = "converted_wavs"
TRANSCRIPTIONS_DIR_NAME = "transcriptions"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def check_ffmpeg() -> bool:
    """Checks if ffmpeg is installed and accessible."""
    if not shutil.which("ffmpeg"):
        logging.error("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
        return False
    logging.info("ffmpeg found.")
    return True


def find_convertible_files(scan_directory: Path, target_extensions: tuple, recursive: bool) -> list[Path]:
    """Finds files matching target_extensions in scan_directory."""
    found_files = []
    normalized_extensions = tuple(ext.lower() for ext in target_extensions)

    if not scan_directory.is_dir():
        logging.warning(f"Scan directory {scan_directory} does not exist or is not a directory.")
        return found_files

    logging.info(f"Scanning {scan_directory} for {normalized_extensions} files (recursive: {recursive})...")

    iterator = scan_directory.rglob("*") if recursive else scan_directory.glob("*")

    for item in iterator:
        if item.is_file():
            if item.suffix.lower() in normalized_extensions:
                found_files.append(item)

    logging.info(f"Found {len(found_files)} files matching extensions.")
    return found_files


def convert_audio_to_wav_file(source_file: Path, target_wav_file: Path) -> bool:
    """Converts an audio/video file to WAV format (16kHz, 16-bit, mono) using ffmpeg."""
    if not source_file.exists():
        logging.error(f"Source file for conversion does not exist: {source_file}")
        return False

    try:
        target_wav_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Converting '{source_file.name}' to '{target_wav_file}'...")
        process = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(source_file),
                "-acodec",
                "pcm_s16le",  # Audio codec: PCM 16-bit little-endian
                "-ac",
                "1",  # Number of audio channels: 1 (mono)
                "-ar",
                "16000",  # Audio sample rate: 16 kHz
                "-vn",  # No video output
                "-f",
                "wav",  # Output format: WAV
                "-y",  # Overwrite output file if it exists
                str(target_wav_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,  # Don't raise exception for non-zero exit, check manually
        )

        if process.returncode != 0:
            logging.error(f"FFmpeg error converting {source_file.name}. Return code: {process.returncode}")
            logging.error(f"FFmpeg stderr: {process.stderr.decode('utf-8', errors='ignore')}")
            if target_wav_file.exists():  # Clean up partially created file
                try:
                    target_wav_file.unlink()
                except OSError as e:
                    logging.error(f"Could not remove partially converted file {target_wav_file}: {e}")
            return False

        logging.info(f"Successfully converted '{source_file.name}' to '{target_wav_file.name}'.")
        return True

    except Exception as e:
        logging.error(f"Error during conversion of {source_file.name}: {e}")
        if target_wav_file.exists():  # Clean up partially created file
            try:
                target_wav_file.unlink()
            except OSError as ose:
                logging.error(f"Could not remove partially converted file {target_wav_file} after exception: {ose}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert audio files and transcribe them using Whisper.")
    parser.add_argument(
        "project_dir", type=Path, help=f"Path to the project directory. Expects a '{RAW_AUDIO_DIR_NAME}' subdirectory."
    )
    parser.add_argument("--configuration", type=str, default=None, help="Path to the Whisper configuration JSON file.")
    parser.add_argument(
        "--run-description",
        type=str,
        default=None,
        help="Run only the configuration with this specific description from the JSON file.",
    )
    parser.add_argument(
        "--convert-extensions",
        type=str,
        default=DEFAULT_CONVERT_EXTENSIONS,
        help="Comma-separated list of file extensions to look for in raw_audio for conversion (e.g., 'mp3,m4a,wav').",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Force conversion of audio files even if target WAV files already exist.",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Optional tag to create a subfolder within the transcriptions directory."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help=f"Disable recursive search in '{RAW_AUDIO_DIR_NAME}'. Default is to search recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what actions would be taken without actually converting or transcribing.",
    )
    parser.add_argument(
        "--no-config-logs",
        action="store_true",
        help="Disable creation of configuration log files in the output directory.",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker diarization. Requires a HuggingFace token set in the HF_TOKEN environment variable.",
    )
    parser.set_defaults(recursive=True)

    args = parser.parse_args()

    if args.dry_run:
        logging.info("--- DRY RUN MODE ENABLED --- No actual files will be changed or processed. ---")

    # Initialize summary counters
    summary = {
        "project_dir": str(args.project_dir),
        "raw_audio_dir": None,
        "converted_wavs_dir": None,
        "transcriptions_base_dir": None,
        "source_files_found": 0,
        "files_converted": 0,
        "files_copied_as_wav": 0,
        "conversion_skipped_exists": 0,
        "conversion_failed": 0,
        "wav_files_for_transcription": 0,
        "transcriptions_submitted": 0,
        "transcription_errors": 0,
        "configs_to_run": 0,
        "run_description_filter": args.run_description,
        "diarization_enabled": args.enable_diarization,
    }

    if not check_ffmpeg():
        return  # Exit if ffmpeg is not available

    project_path = args.project_dir.resolve()
    raw_audio_path = project_path / RAW_AUDIO_DIR_NAME
    converted_wavs_root_path = project_path / CONVERTED_WAVS_DIR_NAME
    transcriptions_root_path = project_path / TRANSCRIPTIONS_DIR_NAME

    summary["raw_audio_dir"] = str(raw_audio_path)
    summary["converted_wavs_dir"] = str(converted_wavs_root_path)

    if args.tag:
        transcriptions_output_base = transcriptions_root_path / args.tag
    else:
        transcriptions_output_base = transcriptions_root_path

    summary["transcriptions_base_dir"] = str(transcriptions_output_base)

    logging.info(f"Project directory: {project_path}")
    logging.info(f"Raw audio directory: {raw_audio_path}")
    logging.info(f"Converted WAVs directory: {converted_wavs_root_path}")
    logging.info(f"Transcriptions base directory: {transcriptions_output_base}")

    if not raw_audio_path.is_dir():
        logging.error(
            f"'{RAW_AUDIO_DIR_NAME}' directory not found in {project_path}. Please create it and add audio files."
        )
        return

    try:
        if not args.dry_run:
            converted_wavs_root_path.mkdir(parents=True, exist_ok=True)
            transcriptions_output_base.mkdir(parents=True, exist_ok=True)
        else:
            logging.info(f"[DRY RUN] Would ensure directory exists: {converted_wavs_root_path}")
            logging.info(f"[DRY RUN] Would ensure directory exists: {transcriptions_output_base}")
    except OSError as e:
        logging.error(f"Could not create required directories: {e}")
        return

    # --- Load Whisper Configurations ---
    if args.configuration is None:
        logging.info("No configuration file specified, using project default!")
        configurations = ConfigurationReader(
            WhisperTranscriber.demo_config  # Assuming this is a valid default path or object
        ).get_configurations()
    else:
        config_file_path = Path(args.configuration)
        if not config_file_path.exists():
            # Try to resolve relative to script's 'configurations' dir
            script_dir_configs = Path(__file__).parent / "configurations" / config_file_path.name
            if not config_file_path.name.endswith(".json"):  # add .json if missing
                script_dir_configs = script_dir_configs.with_suffix(".json")

            if script_dir_configs.exists():
                config_file_path = script_dir_configs
                logging.info(f"Using configuration file: {config_file_path}")
            else:
                logging.error(f"Configuration file {args.configuration} (or {script_dir_configs}) does not exist.")
                return
        configurations = ConfigurationReader(str(config_file_path)).get_configurations()

    if not configurations:
        logging.error("No valid Whisper configurations loaded. Exiting.")
        return

    if args.run_description:
        original_config_count = len(configurations)
        configurations = [c for c in configurations if c.config.get("description") == args.run_description]
        if not configurations:
            logging.error(
                f"No configuration found with description: '{args.run_description}'. "
                f"Available descriptions in {config_file_path}:"
            )
            # Log available descriptions from the originally loaded configurations if needed
            # (requires re-reading or storing them). For now, just error out.
            return
        logging.info(
            f"Filtered to 1 configuration based on --run-description '{args.run_description}' "
            f"(out of {original_config_count})."
        )

    summary["configs_to_run"] = len(configurations)

    # --- File Discovery and Conversion ---
    parsed_extensions = tuple(f".{ext.strip().lstrip('.')}" for ext in args.convert_extensions.split(","))

    source_files_to_process = find_convertible_files(raw_audio_path, parsed_extensions, args.recursive)
    summary["source_files_found"] = len(source_files_to_process)

    if not source_files_to_process:
        logging.info(f"No files found in '{raw_audio_path}' matching extensions: {parsed_extensions}")
        return

    wav_files_for_transcription = []
    logging.info(f"Found {len(source_files_to_process)} source files for potential conversion/processing.")

    source_file_to_original_relative_path_map = {}  # For logging

    for source_file in source_files_to_process:
        relative_to_raw = source_file.relative_to(raw_audio_path)
        target_wav_path = (converted_wavs_root_path / relative_to_raw).with_suffix(TARGET_WAV_EXTENSION)

        source_file_to_original_relative_path_map[str(target_wav_path)] = relative_to_raw

        if source_file.suffix.lower() == TARGET_WAV_EXTENSION:
            # If the source is already a WAV, decide whether to copy or use directly
            if target_wav_path.exists() and not args.force_convert:
                logging.info(f"Target WAV {target_wav_path} already exists (source was WAV). Skipping copy.")
                summary["conversion_skipped_exists"] += 1
            else:
                if not args.dry_run:
                    try:
                        target_wav_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target_wav_path)  # Copy if source is WAV
                        logging.info(f"Copied source WAV {source_file.name} to {target_wav_path}")
                        summary["files_copied_as_wav"] += 1
                    except Exception as e:
                        logging.error(f"Could not copy source WAV {source_file.name} to {target_wav_path}: {e}")
                        summary["conversion_failed"] += 1
                        continue  # Skip this file
                else:
                    logging.info(f"[DRY RUN] Would copy source WAV {source_file.name} to {target_wav_path}")
                    summary["files_copied_as_wav"] += 1  # Count as if it happened for dry run summary
            wav_files_for_transcription.append(target_wav_path)
        else:  # Needs conversion
            if target_wav_path.exists() and not args.force_convert:
                logging.info(f"Skipping conversion, WAV already exists: {target_wav_path}")
                summary["conversion_skipped_exists"] += 1
                wav_files_for_transcription.append(target_wav_path)
            else:
                if not args.dry_run:
                    if convert_audio_to_wav_file(source_file, target_wav_path):
                        summary["files_converted"] += 1
                        wav_files_for_transcription.append(target_wav_path)
                    else:
                        logging.warning(f"Failed to convert {source_file.name}, it will be skipped for transcription.")
                        summary["conversion_failed"] += 1
                else:
                    logging.info(f"[DRY RUN] Would convert {source_file.name} to {target_wav_path}")
                    summary["files_converted"] += 1  # Count as if it happened
                    wav_files_for_transcription.append(target_wav_path)  # Assume conversion success for dry run

    summary["wav_files_for_transcription"] = len(wav_files_for_transcription)
    if not wav_files_for_transcription:
        logging.info("No WAV files available for transcription after conversion/copying phase.")
        return

    logging.info(
        f"Proceeding to transcribe {len(wav_files_for_transcription)} WAV files "
        f"across {len(configurations)} configurations."
    )

    # --- Transcription Loop ---
    # Wrap the outer loop (configurations) and inner loop (wav_files) with tqdm for progress
    total_transcription_tasks = len(configurations) * len(wav_files_for_transcription)
    if total_transcription_tasks == 0:
        logging.info("No transcription tasks to perform.")
    else:
        logging.info(f"Total transcription tasks to perform: {total_transcription_tasks}")

    with tqdm(total=total_transcription_tasks, desc="Transcribing Files", unit="task", disable=args.dry_run) as pbar:
        for configuration_obj in configurations:  # Renamed to avoid conflict
            logging.info(f"Using Whisper Configuration: {configuration_obj}")
            for wav_file_to_transcribe in wav_files_for_transcription:
                config_desc = configuration_obj.config.get("description", "N/A")
                logging.info(f"Preparing to transcribe: {wav_file_to_transcribe.name} with config '{config_desc}'")

                relative_to_converted_root = wav_file_to_transcribe.relative_to(converted_wavs_root_path)
                relative_dir_for_transcription = relative_to_converted_root.parent

                original_relative_path_for_log = source_file_to_original_relative_path_map.get(
                    str(wav_file_to_transcribe), "Unknown_Original_Path"
                )

                logging.debug(f"  Output base for transcription: {transcriptions_output_base}")
                logging.debug(f"  Relative subdir for this file: {relative_dir_for_transcription}")
                logging.debug(f"  Original source relative path (for logging): {original_relative_path_for_log}")

                if not args.dry_run:
                    try:
                        # Create a copy of configuration_obj.config to avoid modifying the original
                        config_copy = configuration_obj.config.copy() if hasattr(configuration_obj, "config") else {}

                        # Set diarization parameters if requested
                        if args.enable_diarization:
                            hf_token = os.environ.get("HF_TOKEN")
                            if not hf_token:
                                logging.warning(
                                    "Diarization enabled but HF_TOKEN environment variable not set. "
                                    "Diarization might not work correctly."
                                )

                            # Update config with diarization settings
                            config_copy["enable_diarization"] = True
                            if hf_token:
                                config_copy["huggingface_token"] = hf_token

                            logging.debug("Speaker diarization enabled for this transcription task.")

                        # If we modified the config, create a new Configuration object
                        if args.enable_diarization and hasattr(configuration_obj, "config"):
                            from WhisperTranscriber.configuration import (  # Keep import local if only used here
                                Configuration,
                            )

                            modified_config_obj = Configuration(config_copy)  # Create new object with modified config
                            transcriber_to_use = modified_config_obj
                        else:
                            transcriber_to_use = configuration_obj

                        transcriber_to_use.transcribe(
                            filename=str(wav_file_to_transcribe),
                            output_base_path=str(transcriptions_output_base),
                            relative_audio_subdir=str(relative_dir_for_transcription),
                            original_relative_path_to_raw=str(original_relative_path_for_log),
                            disable_config_logs=args.no_config_logs,
                        )
                        config_desc = configuration_obj.config.get("description", "N/A")
                        logging.info(
                            f"Successfully submitted {wav_file_to_transcribe.name} "
                            f"for transcription with config '{config_desc}'."
                        )
                        summary["transcriptions_submitted"] += 1
                    except Exception as e:
                        config_desc = configuration_obj.config.get("description", "N/A")
                        logging.error(
                            f"Error during transcription call for {wav_file_to_transcribe.name} "
                            f"with config '{config_desc}': {e}"
                        )
                        logging.exception("Traceback:")
                        summary["transcription_errors"] += 1
                    finally:
                        pbar.update(1)  # Update progress bar after each task (success or fail)
                else:
                    config_desc = configuration_obj.config.get("description")
                    logging.info(
                        f"[DRY RUN] Would transcribe {wav_file_to_transcribe.name} with config '{config_desc}'"
                    )
                    output_structure_path = transcriptions_output_base / relative_dir_for_transcription
                    logging.info(f"[DRY RUN]   Output would be in a structure under: {output_structure_path}")
                    if args.enable_diarization:
                        logging.info("[DRY RUN]   Speaker diarization would be enabled")
                    if not args.no_config_logs:
                        logging.info("[DRY RUN]   Would create config log file in the output directory")
                    else:
                        logging.info("[DRY RUN]   Config log file creation is disabled")
                    summary["transcriptions_submitted"] += 1  # Count as if it happened
                    pbar.update(1)  # Also update progress bar for dry run tasks

    logging.info("All processing finished.")

    # --- Print Summary Report ---
    logging.info("--- Execution Summary ---")
    if args.dry_run:
        logging.info("--- NOTE: This was a DRY RUN. No actual file operations were performed. ---")

    logging.info(f"Project Directory: {summary['project_dir']}")
    logging.info(f"  Raw Audio Source: {summary['raw_audio_dir']}")
    logging.info(f"  Converted WAVs Target: {summary['converted_wavs_dir']}")
    logging.info(f"  Transcriptions Base: {summary['transcriptions_base_dir']}")

    logging.info(f"Configurations to run: {summary['configs_to_run']}")
    if summary["run_description_filter"]:
        logging.info(f"  (Filtered by description: '{summary['run_description_filter']}')")

    if summary["diarization_enabled"]:
        hf_token_status = "available" if os.environ.get("HF_TOKEN") else "not found"
        logging.info(f"  Speaker diarization was enabled (HuggingFace token: {hf_token_status})")

    logging.info(f"Source files found in raw_audio: {summary['source_files_found']}")
    logging.info(f"  Files converted to WAV: {summary['files_converted']}")
    logging.info(f"  Source WAVs copied to converted_wavs: {summary['files_copied_as_wav']}")
    logging.info(f"  Conversions/Copies skipped (target WAV existed): {summary['conversion_skipped_exists']}")
    logging.info(f"  Conversion/Copying failed: {summary['conversion_failed']}")

    logging.info(f"Total WAV files prepared for transcription: {summary['wav_files_for_transcription']}")
    logging.info(f"  Transcription tasks submitted/simulated: {summary['transcriptions_submitted']}")
    logging.info(f"  Transcription errors encountered: {summary['transcription_errors']}")
    logging.info("--- End of Summary ---")


if __name__ == "__main__":
    main()
