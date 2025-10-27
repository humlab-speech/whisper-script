"""
A reader for whisper configuration files, to be used with the WhisperTranscriber module.
"""
import importlib
import json
import os
import sys
import tempfile


def import_package(package_name):
    """In case of ImportError, try to import the package by stepping up through the parent directories.

    Args:
        package_name: The name of the package to import.

    Returns:
        The imported package if successful, or None if the import fails.
    """
    try:
        # Try to import the package
        package = importlib.import_module(package_name)
    except ImportError:
        # If the import fails, start stepping up through the parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        package = None

        while current_dir != os.path.dirname(current_dir):  # Stop when we reach the root directory
            parent_dir = os.path.dirname(current_dir)
            sys.path.insert(0, parent_dir)

            try:
                package = importlib.import_module(package_name)
                break  # If the import succeeds, break out of the loop
            except ImportError:
                pass  # If the import fails, continue to the next parent directory

            # Remove the parent directory from the system path
            sys.path.remove(parent_dir)
            # Set the current directory to the parent directory for the next iteration
            current_dir = parent_dir

        if package is None:
            print(f"Failed to import package {package_name}")

    return package


# Import WhisperTranscriber package
WhisperTranscriber = import_package("WhisperTranscriber")
# These imports must come after the package is imported dynamically
# noqa: E402 below disables the flake8 warning about module-level imports not at top
from WhisperTranscriber import demo_config  # noqa: E402
from WhisperTranscriber.configuration import Configuration  # noqa: E402


class ConfigurationReader:
    """A configuraion reader for whisper configuration files."""

    def __init__(self, configuration_path: str):
        self.config_file_path = configuration_path
        # Check if the configuration is a file path
        if os.path.isfile(self.config_file_path):
            # Verify that the file is readable
            if not os.access(self.config_file_path, os.R_OK):
                raise ValueError("The specified configuration file is not readable: " + self.config_file_path)
            # Try to load the JSON from the file
            try:
                with open(self.config_file_path, encoding="utf-8") as opened_file:
                    config = json.load(opened_file)
            except ValueError as exc:
                raise ValueError(
                    "The specified configuration file does not appear to be valid JSON: " + self.config_file_path
                ) from exc
        else:
            # If the configuration is not a file path, try to parse it as a JSON string
            try:
                config = json.loads(self.config_file_path)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "The specified configuration does not appear to be valid JSON: " + self.config_file_path
                ) from exc
        # Check that config has key 'run_configuration':
        if "run_configuration" not in config:
            raise ValueError(
                "The specified configuration file does not contain a 'run_configuration' key: " + self.config_file_path
            )
        self.configurations = config["run_configuration"]

    # def parse_config_file(self):
    #     config = {}
    #     # Read the specified json configiration file:
    #     with open(self.config_file_path) as config_file:
    #         config = json.load(config_file)

    #     self.configurations = config["run_configuration"]

    def get_configurations(self) -> list[Configuration]:
        """Get all loaded configurations as a list of Configuration objects."""
        print(self.configurations)
        return [Configuration(config) for config in self.configurations]


if __name__ == "__main__":
    # Write the configuration to temporary file:
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        temp.write(demo_config.encode())
        temp.flush()
        temp_path = temp.name
        config_reader = ConfigurationReader(temp_path)
        configurations = config_reader.get_configurations()

        for configuration in configurations:
            print("Now executing using configuration:\n------", configuration, "------")
            configuration.transcribe("ZOOM0020_LR.WAV")
