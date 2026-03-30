# Whisper Transcription Script

A batch transcription client for [WhisperVault](https://github.com/humlab-speech/WhisperVault).
Converts audio files to 16 kHz mono WAV, then sends them to a remote (or local)
WhisperVault server over HTTP or HTTPS for transcription.

The script handles directory-based workflows: point it at a project folder with
a `raw_audio/` subdirectory and it will convert, optionally chunk, and transcribe
every file — preserving the original directory structure in the output.

## Prerequisites

- **Python 3.12+** with a virtualenv
- **FFmpeg** installed and on `PATH` (used for audio conversion)
- **httpx** — the only runtime dependency for talking to WhisperVault
- A running **WhisperVault** instance reachable over the network (either locally via
  the nginx sidecar, or remotely behind a reverse proxy with HTTPS + basic auth)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file (see `.env.example`) with the WhisperVault endpoint:

```dotenv
# HTTPS behind a reverse proxy (typical for remote/production use):
WHISPERX_ENDPOINT=https://whisper.example.com
BASIC_AUTH_USERNAME=myuser
BASIC_AUTH_PASSWORD=mypassword

# Or plain HTTP on a local network:
# WHISPERX_ENDPOINT=http://192.168.1.50:8088
```

The endpoint should point to the WhisperVault nginx sidecar or any reverse
proxy in front of the WhisperVault Unix socket.  Both HTTP and HTTPS are
supported; basic auth credentials are sent automatically when configured.

## Project directory structure

The script expects a project directory with the following layout:

```
project_directory/
│
├── raw_audio/           # Original audio files in any supported format
│   ├── file1.mp3
│   ├── file2.m4a
│   └── subdir/          # Subdirectories are preserved in output
│       └── file3.mp4
│
├── converted_wavs/      # Created automatically — 16 kHz mono WAVs
│
└── transcriptions/      # Created automatically — transcription results
    └── 2026-03-28/
        └── config_name/
            ├── file1.srt
            ├── file1.txt
            └── subdir/
                └── file3.srt
```

## Usage

### Basic transcription

```bash
python run_whisper.py /path/to/project --configuration configurations/large_default_swedish.json
```

### Command-line options

| Flag | Description |
|---|---|
| `--configuration` | Path to a JSON configuration file (see below) |
| `--run-description` | Run only the config entry matching this description |
| `--convert-extensions` | Comma-separated extensions to convert (default: common audio/video) |
| `--force-convert` | Re-convert even if target WAV already exists |
| `--max-chunk-duration` | Split long files into chunks (e.g. `20m`, `1h30m`) |
| `--tag` | Create a named subfolder inside `transcriptions/` |
| `--no-recursive` | Don't recurse into subdirectories of `raw_audio/` |
| `--dry-run` | Show what would happen without doing anything |
| `--no-config-logs` | Skip writing per-file configuration log files |
| `--enable-diarization` | Enable speaker diarization for all files |

### Configuration files

Configuration files live in `configurations/` and define one or more
transcription passes.  Each entry specifies the model, language, and
tuning parameters.  Example (`configurations/large_default_swedish.json`):

```json
{
    "run_configuration": [
        {
            "description": "swedish_kb_whisper_large_ct2_with_vad_0.3",
            "package": "sv-standard",
            "file_format": "SRT",
            "condition_on_prev_text": false,
            "vad": true,
            "vad_speech_threshold": 0.3,
            "repetition_penalty": 1.3,
            "beam_size": 10,
            "diarize": true
        }
    ]
}
```

**Key fields:**

- `package` — a named model bundle defined on the server (resolves model,
  alignment model, diarization model, and language in one go)
- `model` — alternatively, a short model name like `kb-whisper-large-ct2`
  or `large-v3` (resolved via `MODEL_ALIASES` in the transcriber)
- `language` — full name (`swedish`, `english`) or ISO code (`sv`, `en`)
- `diarize` — enable speaker diarization for this config
- `vad` / `vad_speech_threshold` — voice activity detection toggle and threshold
- `file_format` — output format (`SRT`, `TXT`, `VTT`, `JSON`, etc.)

Multiple entries in `run_configuration` are processed sequentially — useful
for comparing different model or parameter combinations on the same audio.

## How it works

1. Scans `raw_audio/` for audio files matching the target extensions
2. Converts each to 16 kHz mono WAV in `converted_wavs/` (skips if already done)
3. Optionally splits long files into chunks (`--max-chunk-duration`)
4. For each configuration × file pair:
   - Calls `POST /reload` on the WhisperVault server (only if the server
     state differs from what this config requires — model, language, VAD, etc.)
   - Calls `POST /transcribe` with the audio file and per-request parameters
   - Saves the output (SRT, TXT, etc.) to `transcriptions/`
5. If chunking was used, merges chunk results back into whole-file transcriptions

## Architecture

```
┌─────────────────────┐       HTTP(S)         ┌──────────────────────┐
│  whisper-script      │ ───────────────────► │  WhisperVault         │
│  (this tool)         │                       │  reverse proxy /      │
│                      │                       │  nginx sidecar        │
│  Converts audio,     │                       │        │              │
│  sends to server,    │                       │        ▼ UDS          │
│  saves results       │                       │  whisperx container   │
│                      │                       │  (--network=none)     │
└─────────────────────┘                       └──────────────────────┘
```

The heavy lifting (model inference, alignment, diarization) happens entirely
on the WhisperVault server.  This script is a thin orchestration layer that
handles file discovery, conversion, and result storage.
