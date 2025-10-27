# Whisper Transcription Script

A robust audio transcription system using the Whisper API through a Gradio interface. This script automates the conversion of various audio formats to WAV and transcribes them using Whisper, maintaining organized directory structures throughout the process.

The container we use for this is [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI), as this
container supports whisper, faster-whisper and insanely-fast-whisper backends, and can be called using the
[gradio](https://www.gradio.app/) API.

## Features

* Automated audio format conversion using FFmpeg
* Structured project organization with separate directories for raw audio, converted WAVs, and transcriptions
* Directory structure preservation from input to output
* Voice Activity Detection (VAD)
* Optional speaker diarization
* Background music removal
* Multiple Whisper model variants
* Dry run mode for testing
* Selective configuration execution
* Customizable output filenames
* Detailed logging and summary reporting

## Project Directory Structure

The script expects a project directory with the following structure:

```
project_directory/
│
├── raw_audio/           # Original audio files in any supported format
│   ├── file1.mp3
│   ├── file2.m4a
│   └── subdir/          # Subdirectories are supported
│       └── file3.mp4
│
├── converted_wavs/      # Will be created automatically - stores converted WAV files
│
└── transcriptions/      # Will be created automatically - stores transcription outputs
```

## Usage

This script can either be used as-is using the supplied `requirements.txt` or by creating the Dockerfile image.

### Installation

#### Using Python directly:

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure FFmpeg is installed
sudo apt-get install ffmpeg  # For Debian/Ubuntu
```

#### Using Docker:

```bash
# Build the Docker image
docker build -t 'whisper_transcriber' ./Docker/

# Run with Docker
docker run --rm -v ./.env:/app/.env -v /path/to/project:/project whisper_transcriber run_whisper.py /project --configuration default_config
```

### Basic Usage

```bash
python run_whisper.py /path/to/project_directory --configuration configurations/large_default_swedish.json
```

This assumes you have a `.env` file in the folder where you execute the command, which should contain:

```
BASIC_AUTH_USERNAME=your_username
BASIC_AUTH_PASSWORD=your_password
GRADIO_WHISPERX_ENDPOINT=https://your.whisper.endpoint
# For speaker diarization (optional)
HF_TOKEN=your_huggingface_token
```

If your configuration does not use HTTPS basic login, omit those lines or leave them empty.

## Command-Line Options

* `--configuration`: Path to the Whisper configuration JSON file (in the `configurations/` directory)
* `--convert-extensions`: Comma-separated list of file extensions to convert (default: mp3,mp4,m4a,wav,etc.)
* `--force-convert`: Force conversion of audio files even if target WAV files already exist
* `--tag`: Create a subfolder within the transcriptions directory for this run
* `--no-recursive`: Disable recursive search in the 'raw_audio' directory
* `--dry-run`: Show what actions would be taken without actually converting or transcribing
* `--no-config-logs`: Disable creation of configuration log files in the output directory
* `--run-description`: Run only the configuration with this specific description from the JSON file
* `--enable-diarization`: Enable speaker diarization to identify different speakers (requires HF_TOKEN)

## Configuration Files

Configuration files are stored in the `configurations/` directory as JSON files. Example:

```json
{
  "description": "Large model with Swedish language",
  "language": "swedish",
  "model": "large-v3",
  "translate_to_english": false,
  "compute_type": "float16"
}
```

Multiple configurations can be included in a single file and will be run sequentially.

## Directory Structure Details

The script organizes files in a structured manner:

1. **raw_audio/**:
   - Place your original audio files here
   - Supports nested subdirectories
   - All common audio formats (mp3, m4a, wav, mp4, etc.)

2. **converted_wavs/**:
   - Created automatically
   - Contains converted WAV files at 16kHz mono
   - Preserves the same directory structure as raw_audio/

3. **transcriptions/**:
   - Created automatically
   - Organized by date, configuration, and directory structure
   - Example: `transcriptions/2025-06-10/large_swedish/subdir/filename.srt`
   - Each transcription includes corresponding JSON log files with configuration details

## Log Files

For each transcription, a log file is created with details about:
- Original file path
- Processing parameters
- Transcription settings
- Output location
- Timestamp

This makes it easy to track which configuration was used for each transcription.
