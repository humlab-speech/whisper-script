#!/usr/bin/env python3

import WhisperTranscriber
from WhisperTranscriber.configuration_reader import ConfigurationReader
import os
import argparse


def find_audio_files(directory, extension=".wav"):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extension.lower())
    ]


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

    args = parser.parse_args()

    input_path = args.input_path
    config_file_path = args.configuration
    output_base_path = args.output

    print("Settings received: ", input_path, config_file_path, output_base_path)

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
            config_file_path = os.path.join(script_directory, 'configuration', config_file_path)
            if not os.path.exists(config_file_path):
                raise FileNotFoundError(
                    f"Configuration file {config_file_path} does not exist."
                )
        configurations = ConfigurationReader(config_file_path).get_configurations()

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    # Check if it is a directory; if so, get all files within the directory:
    if os.path.isdir(input_path):
        # We currently only look for .wav files in the exact directory, not in subdirectories:
        audio_files = find_audio_files(input_path, ".wav")
    else:
        # It exists and is a file, so just use that:
        audio_files = [input_path]

    if len(audio_files) == 0:
        print("No audio files found in the specified directory.")
        return  # No need to continue if no audio files are found.

    print(
        "Starting transcribation, a total of ", len(audio_files), " audio files found."
    )

    for configuration in configurations:
        print("Configuration: ", configuration)
        for audio_file in audio_files:
            print("Transcribing audio file: ", audio_file)
            configuration.transcribe(audio_file, output_base_path=output_base_path)


if __name__ == "__main__":
    main()
