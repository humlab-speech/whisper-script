#!/usr/bin/env python3

import WhisperTranscriber
from WhisperTranscriber.configuration_reader import ConfigurationReader
import os
import argparse
import tempfile
import subprocess
import io


def find_audio_files(directory, extension=".wav", recursive=True):
    found_files = []
    if recursive:
        for root, _, files in os.walk(directory):
            for f_name in files:
                if f_name.lower().endswith(extension.lower()):
                    found_files.append(os.path.join(root, f_name))
    else:
        for f_name in os.listdir(directory):
            file_path = os.path.join(directory, f_name)
            if os.path.isfile(file_path) and f_name.lower().endswith(extension.lower()):
                found_files.append(file_path)
    return found_files
    
def convert_to_wav(file: str) -> io.BytesIO:
    # Create a buffer to store the converted audio
    converted_audio = io.BytesIO()
    
    # Ensure the file exists
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    
    # Convert the audio file to WAV format
    try:
        # Run ffmpeg and pipe the output to stdout
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-i", file,          # Input file
                "-acodec", "pcm_s16le",  # Audio codec: PCM 16-bit little-endian
                "-ac", "1",          # Number of audio channels: 1 (mono)
                "-ar", "16000",      # Audio sample rate: 16 kHz
                "-f", "wav",         # Output format: WAV
                "pipe:1"             # Output to stdout
            ],
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.PIPE   # Capture stderr (optional, for debugging)
        )
        
        # Read the output from ffmpeg's stdout
        output_audio_data, error = process.communicate()

        # Check if ffmpeg encountered an error
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {error.decode('utf-8')}")

        # Write the output audio data to the buffer
        converted_audio.write(output_audio_data)
        converted_audio.seek(0)  # Reset the buffer's position to the beginning

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error converting {file}: {e}")
    
    return converted_audio


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files in a directory."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input file or directory containing audio files.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        required=False,
        default=None,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Base path to the output directory."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive search in subdirectories. Default is to search recursively.",
    )
    parser.set_defaults(recursive=True)

    args = parser.parse_args()

    input_path = args.input_path
    config_file_path = args.configuration
    user_specified_output_base_path = args.output # Renamed for clarity

    print("Settings received: ", input_path, config_file_path, user_specified_output_base_path)
    print(f"Recursive search: {args.recursive}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    if config_file_path is None:
        print("No configuration file specified, using project default!")
        configurations = ConfigurationReader(
            WhisperTranscriber.demo_config
        ).get_configurations()
    else:
        if not os.path.exists(config_file_path):
            # It is not an absolute path, look instead in the configuration directory, and add .json if not present:
            if not config_file_path.endswith(".json"):
                config_file_path += ".json"
            # Get directory of the currently executing script:
            script_directory = os.path.dirname(os.path.realpath(__file__))
            config_file_path = os.path.join(
                script_directory, "configurations", config_file_path
            )
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(
                    f"Configuration file {config_file_path} does not exist."
                )
        configurations = ConfigurationReader(config_file_path).get_configurations()

    if not os.path.exists(user_specified_output_base_path):
        os.makedirs(user_specified_output_base_path)

    audio_files = []
    if os.path.isdir(input_path):
        audio_files = find_audio_files(input_path, ".wav", recursive=args.recursive)
    elif os.path.isfile(input_path):
        audio_files = [input_path]
        # Check for common audio types, ffmpeg will handle actual conversion capability
        if not input_path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
             print(f"Warning: Input file {input_path} may not be a common audio type. Processing will attempt conversion.")
    else:
        raise FileNotFoundError(f"Input path {input_path} is not a valid file or directory.")


    if len(audio_files) == 0:
        print("No audio files found matching criteria.")
        return

    print(
        f"Starting transcription. Found {len(audio_files)} audio files."
    )

    for configuration in configurations:
        print("Configuration: ", configuration)
        for audio_file in audio_files:
            print("Transcribing audio file: ", audio_file)

            # Determine the base directory from which to calculate the relative path
            if os.path.isdir(input_path):
                reference_input_dir = input_path
            else: # input_path is a file
                reference_input_dir = os.path.dirname(input_path)
            
            # Get the relative path of the audio file's directory with respect to the reference_input_dir
            relative_dir_of_audio_file = os.path.relpath(os.path.dirname(audio_file), start=reference_input_dir)
            
            # Call transcribe with the main output path from args and the relative subdirectory
            configuration.transcribe(
                audio_file,
                output_base_path=user_specified_output_base_path, # Root output path from arguments
                relative_audio_subdir=relative_dir_of_audio_file # Relative path like "subfolder1/hello/you" or "."
            )


if __name__ == "__main__":
    main()
