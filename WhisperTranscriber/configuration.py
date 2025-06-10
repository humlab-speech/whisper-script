"""
A module to handle the configuration for the WhisperTranscriber.

Classes:
    Configuration: A class to handle the configuration for the WhisperTranscriber.
"""

import datetime
import json
import logging
import os
import re  # Added for sanitizing folder names
from pathlib import Path

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

    def _write_config_log(self, 
                         final_save_directory: str, 
                         original_relative_path_to_raw: str,
                         processed_filename: str,
                         description: str,
                         model_param: str,
                         language_param: str,
                         config: dict,
                         sanitized_folder_name: str,
                         output_base_path: str = None) -> None:
        """
        Writes a JSON log file with configuration parameters and file information.
        
        Args:
            final_save_directory (str): The directory where the transcription and log will be saved
            original_relative_path_to_raw (str): The original path of the source file relative to raw_audio
            processed_filename (str): The name of the processed WAV file
            description (str): The configuration description
            model_param (str): The model used for transcription
            language_param (str): The language setting used
            config (dict): The full configuration dictionary
            sanitized_folder_name (str): The sanitized folder name derived from description or model
            output_base_path (str, optional): The base output path to calculate relative paths from
        """
        try:
            # Log file will be named based on the original audio file name, in the same directory as the transcription
            base_audio_filename = Path(original_relative_path_to_raw).stem
            params_log_filename = f"{base_audio_filename}__{sanitized_folder_name}.log.json"
            params_log_path = Path(final_save_directory) / params_log_filename

            # Extract the relative path from the final_save_directory
            # We want to construct a path like "2025-06-10/config_name/subdir"
            # without the base directory
            
            # Calculate the relative path
            if output_base_path:
                # Start by checking if the transcriptions_base_dir exists in final_save_directory
                output_base_abs = str(Path(output_base_path).resolve())
                final_save_directory_abs = str(Path(final_save_directory).resolve())
                
                # If final_save_directory starts with output_base_path, remove that prefix
                if final_save_directory_abs.startswith(output_base_abs):
                    # Calculate the relative path by removing the output_base_path prefix
                    relative_output_path = final_save_directory_abs[len(output_base_abs):].lstrip('/')
                else:
                    # Try to extract just the date and following directories
                    path_parts = Path(final_save_directory).parts
                    # Look for a part that could be a date (YYYY-MM-DD format)
                    date_index = None
                    for i, part in enumerate(path_parts):
                        if len(part) == 10 and part[4] == '-' and part[7] == '-':  # Simple date format check
                            date_index = i
                            break
                    
                    if date_index is not None:
                        # If we found a date part, use it and everything after
                        relative_output_path = '/'.join(path_parts[date_index:])
                    else:
                        # Fallback: just use the last 3 directory parts if available
                        relative_output_path = '/'.join(path_parts[-min(3, len(path_parts)):])
            else:
                # Fallback: just use the directory name and parent
                path_parts = Path(final_save_directory).parts
                relative_output_path = '/'.join(path_parts[-min(3, len(path_parts)):])

            log_data = {
                "original_file_relative_to_raw_audio": original_relative_path_to_raw,
                "processed_wav_filename": processed_filename,
                "transcription_output_directory": relative_output_path, 
                "transcription_run_datetime_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "configuration_description": description,  # Original description, not sanitized
                "configuration_model": model_param,
                "configuration_language": language_param,
                "all_parameters_used": config  # This includes all original config items for this run
            }

            # Ensure the directory for the log file exists (should be same as transcription output)
            Path(final_save_directory).mkdir(parents=True, exist_ok=True)
            with open(params_log_path, 'w', encoding='utf-8') as f_log:
                json.dump(log_data, f_log, indent=4, ensure_ascii=False)
            logging.info(f"Wrote transcription parameters log to: {params_log_path}")
        except Exception as e:
            logging.error(f"Failed to write transcription parameters log to {final_save_directory}: {e}")

    def transcribe(self, filename: str, output_base_path: str = "./output", relative_audio_subdir: str = ".", 
                  original_relative_path_to_raw: str = "Unknown_Original_Path", disable_config_logs: bool = False):
        """
        Transcribes the given audio file and saves the transcription to the specified output directory.
        Also writes a log file with transcription parameters.
        Args:
            filename (str): The path to the audio file to be transcribed.
            output_base_path (str, optional): The root path where the output will be saved.
                                              Defaults to "./output".
            relative_audio_subdir (str, optional): The relative subdirectory structure from the input,
                                                   e.g., "subfolder1/hello" or ".".
            original_relative_path_to_raw (str, optional): The original relative path of the source file
                                                           from the raw_audio directory.
            disable_config_logs (bool, optional): If True, skips creating the configuration log file.
                                                  Defaults to False.
        """
        full_path = os.path.abspath(filename)

        # Get date on YY-MM-DD format:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")

        # Work with a copy of the config for this specific transcription run
        # This copy will have items popped from it for the transcriber call
        active_config_params = self.config.copy()

        # Extract parameters needed for path construction and logging
        description = active_config_params.pop("description", None)
        language_param = active_config_params.pop("language", None)
        model_param = active_config_params.pop("model", None)

        # Handle 'subfolder' from config (just removing it, as it's ignored)
        config_original_subfolder = active_config_params.pop("subfolder", None)
        if config_original_subfolder is not None:
            print(f"INFO: The 'subfolder' key ('{config_original_subfolder}') in the configuration is being ignored. Folder name will be derived from 'description' or 'model'.")

        # Determine the folder name: use sanitized description, fallback to sanitized model
        folder_name_source = description if description else model_param
        sanitized_folder_name = self._sanitize_foldername(folder_name_source)

        # Construct the output directory path
        path_parts = [output_base_path, date_string, sanitized_folder_name]
        if relative_audio_subdir and relative_audio_subdir != ".":
            path_parts.append(relative_audio_subdir)
        final_save_directory = os.path.join(*path_parts)

        # Write the configuration log file if enabled
        if not disable_config_logs:
            self._write_config_log(
                final_save_directory=final_save_directory,
                original_relative_path_to_raw=original_relative_path_to_raw,
                processed_filename=Path(filename).name,
                description=description,
                model_param=model_param,
                language_param=language_param,
                config=self.config,
                sanitized_folder_name=sanitized_folder_name,
                output_base_path=output_base_path
            )
        else:
            logging.info("Configuration log file creation is disabled. Skipping.")

        # Initialize the transcriber service
        whisper_service = Transcriber()
        
        # Set up parameters for the transcriber
        active_config_params["original_filename"] = Path(filename).stem
        
        # The disable_args_file parameter is passed to control coordination with transcriber
        # It disables unnecessary file creation when config logs are disabled
        active_config_params["disable_args_file"] = disable_config_logs
        
        # Call the transcriber service
        whisper_service.transcribe(
            full_path,
            model=model_param,
            language=language_param,
            override_output_folder=final_save_directory,
            **active_config_params
        )

    # Add a custom __str__ method to print the configuration:
    def __str__(self):
        return json.dumps(self.config, indent=4)
