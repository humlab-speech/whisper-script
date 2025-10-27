#!/usr/bin/python3

import argparse
import logging
import os
import shutil  # For checking ffmpeg availability
import subprocess
import sys
from pathlib import Path  # For easier path manipulation
from typing import List, Set, Tuple  # Added Set for extensions

# Import tqdm for the progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm library not found.")
    print("Please install it using: pip install tqdm")
    sys.exit(1)

# --- Configuration ---
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # Use TqdmLoggingHandler if available for better integration, otherwise StreamHandler
        logging.StreamHandler(sys.stdout)
    ],
)

# Default extensions (can be overridden by command line)
DEFAULT_EXTENSIONS = "wma,wmv,mp3,mp4,mkv,aac,flac,ogg,m4a,wav,avi,mov,flv,mpeg,mpg,webm"
DEFAULT_OUTPUT_DIR_NAME = "converted_wavs"  # Use just the name for default

# --- Core Functions ---


def check_ffmpeg() -> None:
    """Checks if ffmpeg is installed and accessible."""
    if not shutil.which("ffmpeg"):
        logging.error("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
        sys.exit(1)
    logging.info("ffmpeg found.")


def find_files_to_convert(
    input_path: Path,
    extensions_to_match: Set[str],  # Use a set for efficient lookup
    output_dir: Path,
    case_sensitive: bool,
) -> List[Tuple[Path, Path]]:
    """
    Finds all files matching the extensions in the input path and
    determines their corresponding output paths, preserving structure.
    Skips the output directory if it's inside the input directory.
    Returns a list of (input_file, output_file) path tuples.
    """
    files_to_process = []
    # Resolve paths early for consistent comparisons
    input_path_abs = input_path.resolve()
    output_dir_abs = output_dir.resolve()

    logging.info(f"Scanning '{input_path_abs}'...")
    logging.info(f"Output will be placed in '{output_dir_abs}'.")
    if case_sensitive:
        logging.info("Using case-sensitive extension matching.")
    else:
        logging.info("Using case-insensitive extension matching (default).")

    if input_path_abs.is_file():
        # Handle single file input
        file_suffix = input_path_abs.suffix
        suffix_to_check = file_suffix if case_sensitive else file_suffix.lower()

        if suffix_to_check in extensions_to_match:
            # Place single file directly in output dir root, maintaining name
            output_file = output_dir_abs / f"{input_path_abs.stem}.wav"
            # Ensure output directory exists for single file conversion
            output_dir_abs.mkdir(parents=True, exist_ok=True)
            files_to_process.append((input_path_abs, output_file))
        else:
            logging.warning(
                f"Input file '{input_path_abs}' does not match target extensions {list(extensions_to_match)}. Skipping."
            )

    elif input_path_abs.is_dir():
        # Handle directory input
        output_dir_relative_name = output_dir_abs.name  # Needed for os.walk check

        for root, dirs, files in os.walk(input_path_abs, topdown=True):
            current_dir = Path(root)
            current_dir_abs = current_dir.resolve()

            # --- Crucial Check: Prevent descending into the output directory ---
            # Check if the resolved output directory is EXACTLY one of the subdirectories
            # os.walk is about to traverse.
            if output_dir_abs == current_dir_abs / output_dir_relative_name:
                logging.info(f"Skipping scan of output directory found within input: '{output_dir_abs}'")
                # Remove the output directory from the list of directories to visit
                # This requires output_dir to be a direct child of current_dir
                if output_dir_relative_name in dirs:
                    dirs.remove(output_dir_relative_name)
                # Also handle case where output path *is* the current path (less likely but possible)
                elif output_dir_abs == current_dir_abs:
                    logging.warning("Scanning started within the output directory itself? Skipping further scan here.")
                    dirs[:] = []  # Don't descend further from here

            # Skip processing files if the current directory *is* the output directory
            # This handles cases where output_dir might be nested deeper or is the root
            if current_dir_abs == output_dir_abs:
                continue  # Don't process files *in* the output directory

            for file in files:
                input_file = current_dir / file
                file_suffix = input_file.suffix
                suffix_to_check = file_suffix if case_sensitive else file_suffix.lower()

                if suffix_to_check in extensions_to_match:
                    # Calculate relative path from input_dir base
                    relative_path = input_file.parent.relative_to(input_path_abs)
                    # Construct output subdirectory path
                    output_subdir = output_dir_abs / relative_path
                    # Construct final output file path
                    output_file = output_subdir / f"{input_file.stem}.wav"
                    files_to_process.append((input_file, output_file))
    else:
        logging.error(f"Input path '{input_path}' is not a valid file or directory.")
        # No files to process, return empty list

    return files_to_process


def convert_file(input_file: Path, output_file: Path) -> bool:
    """
    Converts a single audio/video file to 16kHz mono 16-bit PCM WAV using ffmpeg.
    Returns True on success or if file already exists, False on failure.
    """
    # Create the output directory for the specific file if it doesn't exist
    # This is done here instead of globally to handle structure creation lazily
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {output_file.parent}: {e}")
        return False  # Cannot proceed if directory creation fails

    if output_file.exists():
        logging.debug(f"Output file '{output_file}' already exists. Skipping conversion.")
        return True  # Consider existing file as a 'success' for progress

    output_relative = output_file.relative_to(
        output_file.parent.parent if output_file.parent != output_file.parent.parent else output_file.parent
    )
    logging.info(f"Converting '{input_file.name}' -> '{output_relative}'")  # Show relative output path nicely

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_file),  # Input file
                "-acodec",
                "pcm_s16le",  # Audio codec: PCM signed 16-bit little-endian
                "-ac",
                "1",  # Audio channels: 1 (mono)
                "-ar",
                "16000",  # Audio sample rate: 16000 Hz
                "-vn",  # No video
                "-loglevel",
                "error",  # Suppress verbose ffmpeg output, only show errors
                "-y",  # Overwrite output without asking (we check existence above)
                str(output_file),  # Output file
            ],
            check=True,  # Raise exception on non-zero exit code
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode stdout/stderr as text
            encoding="utf-8",  # Specify encoding
            errors="replace",  # Handle potential encoding errors in ffmpeg output
        )
        logging.debug(f"Successfully converted '{input_file.name}'")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting '{input_file.name}': ffmpeg exited with code {e.returncode}")
        # Log stderr, as it usually contains the ffmpeg error message
        if e.stderr:
            logging.error(f"FFmpeg stderr:\n{e.stderr.strip()}")
        if e.stdout:  # Log stdout too, might be useful
            logging.error(f"FFmpeg stdout:\n{e.stdout.strip()}")
        # Optionally remove partially created file on error
        if output_file.exists():
            try:
                output_file.unlink()
                logging.warning(f"Removed incomplete output file: '{output_file}'")
            except OSError as rm_err:
                logging.error(f"Failed to remove incomplete file '{output_file}': {rm_err}")
        return False
    except FileNotFoundError:  # Handle case where ffmpeg command itself fails (e.g. input missing during run)
        logging.error(f"Error converting '{input_file.name}': Input file likely disappeared or ffmpeg error.")
        return False
    except Exception as e:  # Catch other potential errors during subprocess execution
        logging.error(f"An unexpected error occurred during conversion of '{input_file.name}': {e}")
        return False


def main():
    """Main function to parse arguments and orchestrate conversion."""
    parser = argparse.ArgumentParser(
        description="Convert audio/video files to 16 kHz mono 16-bit PCM WAV format using ffmpeg. "
        "Scans for files first, allowing the output directory to be inside the input directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults in help
    )
    parser.add_argument(
        "input", type=Path, help="Input file or directory containing source files."  # Use Path object directly
    )
    parser.add_argument(
        "--extensions",
        "-e",
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated list of source file extensions to convert.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,  # Use Path object
        # Default is relative to the *input* path if input is a directory,
        # or relative to the input file's parent otherwise. Set later.
        default=None,
        help=f"Output directory to store converted WAV files. Maintains structure relative to input base. "
        f"Defaults to '{DEFAULT_OUTPUT_DIR_NAME}' inside the input directory (or its parent if input is a file).",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match file extensions case-sensitively (default is case-insensitive).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically confirm potentially unsafe operations (like output inside input).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level).")

    args = parser.parse_args()

    # --- Configure Logging Level ---
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")

    # --- Pre-checks ---
    check_ffmpeg()

    input_path_raw: Path = args.input
    if not input_path_raw.exists():
        logging.error(f"Input path does not exist: '{input_path_raw}'")
        sys.exit(1)

    # Resolve input path *after* existence check
    input_path: Path = input_path_raw.resolve()

    # --- Determine Output Path ---
    if args.output:
        # If output is specified, use it and resolve it
        output_path: Path = args.output.resolve()
    else:
        # Default output path logic
        if input_path.is_dir():
            output_path = input_path / DEFAULT_OUTPUT_DIR_NAME
        elif input_path.is_file():
            output_path = input_path.parent / DEFAULT_OUTPUT_DIR_NAME
        else:
            # Should not happen due to existence check, but defensively:
            logging.error(f"Input path '{input_path}' is neither a file nor a directory after resolving.")
            sys.exit(1)
        # No need to resolve output_path here as it's built from resolved input_path

    # --- Safety check: Output directory within input directory ---
    # This check is most relevant when input is a directory
    # Use is_relative_to (Python 3.9+) for robust check
    is_output_inside_input = False
    try:
        # Check if output_path is the same as or inside input_path
        if input_path.is_dir() and output_path.is_relative_to(input_path):
            is_output_inside_input = True
    except ValueError:
        # is_relative_to raises ValueError if paths are on different drives (Windows)
        # or cannot be related. Assume they are not inside in this case.
        pass
    except AttributeError:
        # Fallback for Python < 3.9
        # This is less robust, especially with symlinks or '..'
        try:
            output_path.relative_to(input_path)
            if input_path.is_dir():  # Double check input is dir for this logic
                is_output_inside_input = True
        except ValueError:
            pass  # Not relative

    if is_output_inside_input:
        logging.warning("-" * 40)
        logging.warning(f"Output directory '{output_path}'")
        logging.warning(f"is inside the input directory '{input_path}'.")
        logging.warning("The script will attempt to skip scanning the output directory.")
        logging.warning("-" * 40)
        if not args.yes:
            try:
                confirm = input("Continue? (y/N): ")
                if confirm.lower() != "y":
                    logging.info("Operation cancelled by user.")
                    sys.exit(0)
            except EOFError:  # Handle non-interactive environments
                logging.warning("Cannot get confirmation in non-interactive mode. Use -y to proceed automatically.")
                sys.exit(1)

    # --- Prepare Extensions ---
    raw_extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    if not raw_extensions:
        logging.error("No valid file extensions provided via --extensions.")
        sys.exit(1)

    # Prepare the set of extensions to match against, handling case sensitivity
    if args.case_sensitive:
        # Add leading dot, keep case
        extensions_to_match = {f".{ext}" for ext in raw_extensions}
    else:
        # Add leading dot, convert to lowercase
        extensions_to_match = {f".{ext.lower()}" for ext in raw_extensions}

    logging.info(f"Targeting extensions: {', '.join(sorted(list(extensions_to_match)))}")

    # --- Find Files (Pre-computation Step) ---
    logging.info("Scanning for files to convert (this may take a while for large directories)...")
    try:
        files_to_convert = find_files_to_convert(input_path, extensions_to_match, output_path, args.case_sensitive)
    except Exception as e:
        logging.error(f"An error occurred during file scanning: {e}", exc_info=args.verbose)
        sys.exit(1)

    if not files_to_convert:
        logging.info("No files matching the specified extensions were found to convert.")
        sys.exit(0)

    logging.info(f"Found {len(files_to_convert)} file(s) marked for potential conversion.")
    logging.debug("Files to process:")
    for infile, outfile in files_to_convert:
        logging.debug(f"  '{infile}' -> '{outfile}'")

    # --- Process Files ---
    successful_conversions = 0
    failed_conversions = 0
    skipped_existing = 0  # Keep track of skips vs actual conversions

    # Create the main output directory if it doesn't exist and we have files
    # Do this *after* finding files to avoid creating empty dirs
    if files_to_convert:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create base output directory '{output_path}': {e}")
            sys.exit(1)

    print("-" * 30)  # Separator before progress bar
    # Initialize tqdm progress bar
    with tqdm(total=len(files_to_convert), unit="file", desc="Converting", leave=True, ncols=100, ascii=True) as pbar:
        for input_file, output_file in files_to_convert:
            # Check existence again just before conversion (less likely needed, but safe)
            if output_file.exists():
                logging.debug(f"Skipping already existing file: '{output_file}'")
                skipped_existing += 1
                successful_conversions += 1  # Count skip as success for overall count
                pbar.update(1)
                pbar.set_postfix_str("skipped existing", refresh=True)
                continue

            # Perform actual conversion
            if convert_file(input_file, output_file):
                successful_conversions += 1
                pbar.set_postfix_str("converted", refresh=True)
            else:
                failed_conversions += 1
                pbar.set_postfix_str("failed", refresh=True)
                # Optionally add a small pause or different display for failures
            pbar.update(1)

    # --- Summary ---
    print("-" * 30)  # Separator after progress bar
    logging.info("Conversion process finished.")
    logging.info(f"Total files scanned: {len(files_to_convert)}")
    logging.info(f"Successfully converted: {successful_conversions - skipped_existing}")
    logging.info(f"Skipped (already exist): {skipped_existing}")
    logging.info(f"Failed conversions: {failed_conversions}")
    if failed_conversions > 0:
        logging.warning("Check logs above for details on failed conversions.")
    logging.info(f"Output saved to: '{output_path}'")
    print("-" * 30)

    if failed_conversions > 0:
        sys.exit(1)  # Exit with error code if any conversions failed


if __name__ == "__main__":
    main()
