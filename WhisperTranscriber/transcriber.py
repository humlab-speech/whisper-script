from enum import Enum
from pathlib import Path
from gradio_client import Client, handle_file
import httpx
from dotenv import load_dotenv
import os
from typing import Any, Union
import tempfile
from .whisperxsettings import WhisperSettings
import pysrt


class Transcriber:
    """
    A transcriber

    Attributes:
        auth (Any): The authentication credentials.
        client (Client): The client used for making requests.
    """

    """
    Consider if we should use the API to getthese instead; however not very useful if we must connect first...
    """

    class WhisperXModel(Enum):
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

        # class WhisperLanguages(Enum) =
        # ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese', 'azerbaijani', 'bashkir', 'basque', 'belarusian', 'bengali', 'bosnian', 'breton', 'bulgarian', 'cantonese', 'catalan', 'chinese', 'croatian', 'czech', 'danish', 'dutch', 'english', 'estonian', 'faroese', 'finnish', 'french', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hungarian', 'icelandic', 'indonesian', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'lao', 'latin', 'latvian', 'lingala', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar', 'nepali', 'norwegian', 'nynorsk', 'occitan', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'sanskrit', 'serbian', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tagalog', 'tajik', 'tamil', 'tatar', 'telugu', 'thai', 'tibetan', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yiddish', 'yoruba', 'Automatic Detection']

    client: Client = None  # Class attribute with type hint
    download_path: Path

    def __init__(self, download_path: Path = None):
        """
        Initializes the WhisperXTranscriber instance.

        Args:
          download_path (Path): The path where files will be downloaded. If not specified, a temporary directory will be used.
        """
        # Load environment variables from .env file
        load_dotenv()

        # If no download path is specified, create a temporary directory
        if download_path is None:
            self.temp_dir = tempfile.TemporaryDirectory()
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

    def get_models(self):
        return self.client.get_models()

    def srt_to_txt(self, srt_file_path):
        # Load the .srt file
        subs = pysrt.open(srt_file_path)

        # Create the .txt file path
        base = os.path.splitext(srt_file_path)[0]
        txt_file_path = base + ".txt"

        # Open the .txt file (or create it if it doesn't exist)
        with open(txt_file_path, "w") as txt_file:
            # Loop through each subtitle in the .srt file
            for sub in subs:
                # Write the subtitle text to the .txt file
                txt_file.write(sub.text + "\n")

    def transcribe(
        self,
        file_name: str,
        model: Union[str, WhisperXModel] = WhisperXModel.LARGE_V2,
        vad: bool = False,
        language: str = "Automatic Detection",
        output_path="./output",
        tag: str = None,
        override_output_folder: str = None,
        write_results=False,
        settings: WhisperSettings = None,
        ## kwargs are optional settings to modify!
        **kwargs,
    ):
        if settings is None:
            settings = WhisperSettings()

        # Verify that the file exists:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")

        # If model is a string, convert it to WhisperXModel enum
        if isinstance(model, str):
            try:
                model = self.WhisperXModel(model.lower())
            except ValueError:
                raise ValueError(f"Invalid model name: {model}")

        # Ensure model is of type WhisperXModel
        if not isinstance(model, self.WhisperXModel):
            raise TypeError(
                f"Model must be of type WhisperXModel or str, got {type(model)}"
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

        files = {
            os.path.basename(file): open(file, "r").read() for file in filepath_list
        }

        # Check that the output folder exists, and create it if not:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        from datetime import datetime

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

            # Save it to the appropriate folder
            with open(os.path.join(output_path, filename), "w") as file:
                file.write(content)
                print(f"File saved: {os.path.join(output_path, filename)}")
            # Convert the just saved .srt file to .txt
            if filename.endswith(".srt"):
                print("Converting file to txt")
                self.srt_to_txt(os.path.join(output_path, filename))

            # Also save the arguments used to call this function:
            # Save both positional arguments and kwargs
            args_to_save = {
                "file_name": file_name,
                "model": model.value,
                "vad": vad,
                "language": language,
                "output_path": output_path,
                "tag": tag,
                "override_output_folder": override_output_folder,
                "write_results": write_results,
                **kwargs,
            }
            with open(os.path.join(output_path, filename + "_args.txt"), "w") as file:
                file.write(str(args_to_save))
                print(
                    f"Arguments saved: {os.path.join(output_path, filename + '_args.txt')}"
                )

            # ALso save the results for each, so that the user can access them, if wanted:
            if write_results:
                with open(
                    os.path.join(output_path, filename + "_results.txt"), "w"
                ) as file:
                    file.write(result)
                    print(
                        f"Result saved: {os.path.join(output_path, filename + '.txt')}"
                    )