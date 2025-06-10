"""
Transcriber module for Whisper API.

Classes:
    Transcriber: Manages transcription, client setup, auth, and file handling.
"""

import tempfile
from enum import Enum
from pathlib import Path
import os
from typing import Union
import atexit
from datetime import datetime

import httpx
import pysrt
from dotenv import load_dotenv
from gradio_client import Client, handle_file

from .whispersettings import WhisperSettings


class Transcriber:
    """
    Manages transcription, client setup, auth, and file handling.
    """

    class WhisperModel(Enum):
        """Class to define the available Whisper models."""

        TINY_EN = "tiny.en"
        TINY = "tiny"
        BASE_EN = "base.en"
        BASE = "base"
        SMALL_EN = "small.en"
        SMALL = "small"
        MEDIUM_EN = "medium.en"
        MEDIUM = "medium"
        LARGE_V1 = "large-v1"
        LARGE_V2 = "large-v2"
        LARGE_V3 = "large-v3"
        LARGE = "large"
        WHISPER_LARGE_V3_TURBO = "whisper-large-v3-turbo"
        KB_WHISPER_LARGE = "kb-whisper-large-ct2"

        # class WhisperLanguages(Enum) =
        # ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese', 'azerbaijani', 'bashkir', 'basque', 'belarusian', 'bengali', 'bosnian', 'breton', 'bulgarian', 'cantonese', 'catalan', 'chinese', 'croatian', 'czech', 'danish', 'dutch', 'english', 'estonian', 'faroese', 'finnish', 'french', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hungarian', 'icelandic', 'indonesian', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'lao', 'latin', 'latvian', 'lingala', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar', 'nepali', 'norwegian', 'nynorsk', 'occitan', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'sanskrit', 'serbian', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tagalog', 'tajik', 'tamil', 'tatar', 'telugu', 'thai', 'tibetan', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yiddish', 'yoruba', 'Automatic Detection']

    client: Client = None  # Class attribute with type hint
    download_path: Path

    def __init__(self, download_path: Path = None):
        """
        Initializes the WhisperTranscriber instance.

        Args:
          download_path (Path): The path where files will be downloaded. If not specified, a temporary directory will be used.
        """
        # Load environment variables from .env file
        load_dotenv()
        if not os.getenv("GRADIO_WHISPERX_ENDPOINT"):
            raise ValueError(
                "GRADIO_WHISPERX_ENDPOINT environment variable not found. Did you forget to set it in the .env file?"
            )

        # If no download path is specified, create a temporary directory
        if download_path is None:
            self.temp_dir = tempfile.TemporaryDirectory()
            atexit.register(
                self.temp_dir.cleanup
            )  # Clean up the temporary directory on exit
            self.download_path = Path(self.temp_dir.name)
        else:
            self.download_path = download_path

        # Get username and password from environment variables

        try:
            username = os.getenv("BASIC_AUTH_USERNAME")
            password = os.getenv("BASIC_AUTH_PASSWORD")
            if username is None or password is None:
                raise ValueError(
                    "Username or password not found in environment variables."
                )
            self.auth = httpx.BasicAuth(username, password)
        except Exception as e:
            print(f"Error loading authentication credentials: {e}")
            self.auth = None

        self.initialize_client()

    def initialize_client(self):
        """Initializes the client."""
        httpx_kwargs = {"auth": self.auth} if self.auth else None
        if httpx_kwargs is None:
            self.client = Client(
                os.getenv("GRADIO_WHISPERX_ENDPOINT"),
                download_files="./temp/",
            )
        else:
            self.client = Client(
                os.getenv("GRADIO_WHISPERX_ENDPOINT"),
                httpx_kwargs=httpx_kwargs,
                download_files="./temp/",
            )

    def get_api_dict(self):
        return self.client.view_api(print_info=False, return_format="dict")

    @staticmethod
    def srt_to_txt(srt_file_path):
        # Load the .srt file
        subs = pysrt.open(srt_file_path)

        # Create the .txt file path
        base = os.path.splitext(srt_file_path)[0]
        txt_file_path = base + ".txt"

        # Open the .txt file (or create it if it doesn't exist)
        with open(txt_file_path, "w", encoding="utf-8") as txt_file:
            # Loop through each subtitle in the .srt file
            for sub in subs:
                # Write the subtitle text to the .txt file
                txt_file.write(sub.text + "\n")

    def transcribe(
        self,
        file_name: str,
        model: Union[str, WhisperModel] = WhisperModel.LARGE_V2,
        vad: bool = False,
        language: str = "Automatic Detection",
        output_path="./output",
        tag: str = None,
        override_output_folder: str = None,
        settings: WhisperSettings = None,
        original_filename: str = None,  # Optional parameter to specify the output filename
        disable_args_file: bool = False,  # Optional parameter to disable writing args file (functionality removed)
        ## kwargs are optional settings to modify!
        **kwargs,
    ):
        if settings is None:
            settings = WhisperSettings()

        # Verify that the file exists:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")

        # Note: The disable_args_file parameter is kept for backward compatibility,
        # but the functionality to create separate args.txt files has been removed.
        # This parameter is now only used to coordinate with the Configuration class.

        # If model is a string, convert it to WhisperModel enum
        if isinstance(model, str):
            try:
                model = self.WhisperModel(model.lower())
            except ValueError as exc:
                raise ValueError(f"Invalid model name: {model}") from exc

        # Ensure model is of type WhisperModel
        if not isinstance(model, self.WhisperModel):
            raise TypeError(
                f"Model must be of type Whisper model or str, got {type(model)}"
            )

        settings.upload_file = [handle_file(file_name)]
        settings["language"] = language
        settings["model"] = model.value
        settings["enable_silero_vad_filter"] = vad

        # Add kwargs to settings:
        for key, value in kwargs.items():
            if key in settings.setting_keys:
                print("Setting ", key, " to ", value)
                settings[key] = value
            else:
                print("Possible keys: ", settings.setting_keys)
                raise KeyError(f"Invalid setting: {key}")

        # result = self.client.predict(api_name="/lambda_1")
        result, filepath_list = self.client.predict(**settings.get_settings())

        # print(result)
        # print(filepath_list)

        print("Moving files: ", filepath_list)

        files = {}
        for file in filepath_list:
            with open(file, "r", encoding="utf-8") as f:
                files[os.path.basename(file)] = f.read()

        # Check that the output folder exists, and create it if not:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename, content in files.items():
            # Output path should be output_folder/YYYY_MM_DD/model_type/VAD_TRUE_OR_FALSE/filename:
            current_date = datetime.now().strftime("%Y_%m_%d")
            model_type = model.value
            vad_status = (
                "Using_Silence_Reduction" if vad else "Not_Using_Silence_Reduction"
            )

            # Create the directory structure
            output_path = os.path.join(
                output_path, current_date, tag if tag else "", model_type, vad_status
            )
            if override_output_folder is not None:  # Just use the specified regardless
                output_path = override_output_folder

            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)

            # Determine the output filename - use original_filename if provided
            output_filename = filename
            if original_filename is not None:
                # Get the extension from the original API output filename
                extension = os.path.splitext(filename)[1]
                # Create a new filename using the provided original_filename with the same extension
                output_filename = f"{original_filename}{extension}"

            # Save it to the appropriate folder
            with open(
                os.path.join(output_path, output_filename), "w", encoding="utf-8"
            ) as file:
                file.write(content)
                print(f"File saved: {os.path.join(output_path, output_filename)}")
            
            # Convert the just saved .srt file to .txt
            if output_filename.endswith(".srt"):
                print("Converting file to txt")
                Transcriber.srt_to_txt(os.path.join(output_path, output_filename))
                
