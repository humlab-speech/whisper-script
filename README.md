# Introduction
This is a small script for calling a gradio instance running whisper.

The container we use for this is [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI), as this
container supports whisper, faster-whisper and insanely-fast-whisper backends, and can be called using the  
[gradio](https://www.gradio.app/) API.

Features:

* VAD - Voice Activity Detection
* Optional background music removal
* Many whisper model variants
* And much more!

# Usage

This script can either be used as-is using the supplied `requirements.txt` or by creating the Dockerfile image.

The docker image file can be created using `docker build`: 

`docker build -t 'test_whisper' ./Docker/`

The docker image can then be used for example as such:

`docker run --rm -v ./.env:/app/.env -v ./demo:/demo test_whisper run_whisper.py /demo/converted --configuration default_config --output /demo/transcription/v1`

This assumes we have a `.env`file in the folder we execute it, which should contain the following:

```
BASIC_AUTH_USERNAME=
BASIC_AUTH_PASSWORD=
GRADIO_WHISPERX_ENDPOINT=https://your.ip
```




