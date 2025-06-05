"""
A module to handle the configuration for the WhisperTranscriber.

Classes:
    Configuration: A class to handle the configuration for the WhisperTranscriber.
"""

import datetime
import json
import os
import re  # Added for sanitizing folder names

from WhisperTranscriber.transcriber import Transcriber


class Configuration:
    """
    A class to handle the configuration for the WhisperTranscriber.

    Attributes:
        config (dict): A dictionary containing the configuration parameters.

    Methods:
        __init__(config):
            Initializes the Configuration object with the given config dictionary.

        transcribe(filename: str, output_base_path: str = "./output"):
            Transcribes the given audio file using the configuration parameters.
            Args:
                filename (str): The path to the audio file to be transcribed.
                output_base_path (str, optional): The base path where the output will be saved.
                    Defaults to "./output".
                relative_audio_subdir (str, optional): The relative subdirectory structure
                    from the input path. Defaults to ".".

        __str__():
            Returns a JSON string representation of the configuration.
    """
    def __init__(self, config):
        self.config = config

    def _sanitize_foldername(self, name: str) -> str:
        """Sanitizes a string to be a valid folder name."""
        if not name:
            return "unnamed_configuration"  # Default for empty name

        temp_name = str(name)  # Ensure it's a string
        # Replace whitespace with underscores
        temp_name = re.sub(r"\\s+", "_", temp_name)
        # Remove characters that are not alphanumeric, underscore, or hyphen
        temp_name = re.sub(r"[^a-zA-Z0-9_-]", "", temp_name)
        # Avoid names that are too long (e.g., > 255 chars, common limit)
        temp_name = temp_name[:200]  # Truncate to a reasonable length

        if not temp_name:  # If sanitization results in an empty string (e.g., name was just "...")
            return "sanitized_empty_name"
        return temp_name

    def transcribe(self, filename: str, output_base_path: str = "./output", relative_audio_subdir: str = "."):
        """
        Transcribes the given audio file and saves the transcription to the specified output directory.
        Args:
            filename (str): The path to the audio file to be transcribed.
            output_base_path (str, optional): The root path where the output will be saved.
                                              Defaults to "./output".
            relative_audio_subdir (str, optional): The relative subdirectory structure from the input,
                                                   e.g., "subfolder1/hello" or ".".
        """
        full_path = os.path.abspath(filename)

        # Get date on YY-MM-DD format:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")

        # Work with a copy of the config for this specific transcription run
        # This copy will have items popped from it for the transcriber call
        active_config_params = self.config.copy()

        # Pop essential params needed for path construction or direct call
        description = active_config_params.pop("description", None)
        language_param = active_config_params.pop("language", None)
        model_param = active_config_params.pop("model", None)

        # Handle 'subfolder' from config: print info if it exists, then ensure it's popped
        # so it's not passed in **active_config_params later.
        config_original_subfolder = active_config_params.pop("subfolder", None)
        if config_original_subfolder is not None:
            print(f"INFO: The 'subfolder' key ('{config_original_subfolder}') in the configuration is being ignored. Folder name will be derived from 'description' or 'model'.")

        # Determine the folder name: use sanitized description, fallback to sanitized model
        folder_name_source = description if description else model_param
        sanitized_folder_name = self._sanitize_foldername(folder_name_source)

        # Construct the save_directory path
        # Desired structure: output_base_path / date_string / sanitized_folder_name / relative_audio_subdir

        path_parts = [output_base_path, date_string, sanitized_folder_name]

        # Add the relative audio subdirectory (if it's not "." which means current dir for os.path.join)
        if relative_audio_subdir and relative_audio_subdir != ".":
            path_parts.append(relative_audio_subdir)

        final_save_directory = os.path.join(*path_parts)

        # Instantiate the Transcriber - this fixes a likely bug in original code
        # where 'transcriber' (lowercase) was used without being an instance of Transcriber (uppercase class).
        whisper_service = Transcriber()

        whisper_service.transcribe(
            full_path,  # Assuming first arg is the file path as per original structure
            model=model_param,
            language=language_param,
            override_output_folder=final_save_directory,
            **active_config_params  # Pass remaining params from the copied config
        )

    # Add a custom __str__ method to print the configuration:
    def __str__(self):
        return json.dumps(self.config, indent=4)
