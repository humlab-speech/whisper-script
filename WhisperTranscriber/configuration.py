from WhisperTranscriber.transcriber import Transcriber
import datetime
import json
import os


class Configuration:
    def __init__(self, config):
        self.config = config

    def transcribe(self, filename: str, output_base_path: str = "./output"):
        transcriber = Transcriber()
        config = self.config.copy()
        # get full path of filename:
        full_path = os.path.abspath(filename)

        # Get date on YY-MM-DD format:
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")

        # Get all the required parameters from the configuration file:
        description = config.pop("description")
        language = config.pop("language")
        model = config.pop("model")
        subfolder = config.pop("subfolder", None)
        if subfolder is None:
            save_directory = os.path.join(output_base_path, date_string, description)
        else:
            save_directory = os.path.join(output_base_path, date_string, subfolder)

        transcriber.transcribe(
            full_path,
            model=model,
            language=language,
            override_output_folder=save_directory,
            **config
        )

        # # Write configuration used to the same folder:
        # config_file_path = os.path.join(save_directory, "configuration.json")
        # with open(config_file_path, "w") as config_file:
        #     json.dump(self.config, config_file, indent=4)

    # Add a custom __str__ method to print the configuration:
    def __str__(self):
        return json.dumps(self.config, indent=4)
