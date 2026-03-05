# Project instructions (Copilot + Claude)

This file is read automatically by GitHub Copilot in every session.
Keep it up to date when the architecture or conventions change.

---

## What this project is

**whisper-script** — a Python pipeline that converts audio files to WAV and
transcribes them using the **WhisperVault** HTTP API
([WhisperX](https://github.com/m-bain/whisperX) with word-level timestamps,
forced alignment, and speaker diarization) running on a remote server.

The client sends audio via `POST /transcribe` (multipart) and hot-swaps the
server's loaded model via `POST /reload` when the configuration changes.
There is no Gradio dependency — the backend is a plain `httpx` HTTP client.

---

## Repository layout

```
run_whisper.py                  Entry point: converts audio files and transcribes them
WhisperTranscriber/
  whispervault_transcriber.py   HTTP client for the WhisperVault API (the backend)
  configuration.py              Reads a config object and calls the transcriber
  configuration_reader.py       Parses JSON config files into Configuration objects
  __init__.py                   Package init / demo config string
configurations/                 JSON config files (one or more run_configurations per file)
scripts/
  audio_chunking.py             Split long audio into chunks and merge results
  convert_to_wav.py             Standalone ffmpeg wrapper
  srt_to_txt.py                 SRT → plain text conversion
  scan_folder_for_audio.py      Utility to list audio files
  normalize_volume.sh           Shell helper for volume normalisation
  convert_folder_srt.sh         Shell helper for batch SRT conversion
Docker/
  Dockerfile                    Legacy Docker image (kept for reference)
```

---

## How the pipeline works

```
run_whisper.py main()
  → find audio files in raw_audio/
  → convert to 16kHz mono WAV (ffmpeg)
  → optionally chunk long files (audio_chunking.py)
  → for each Configuration object:
      Configuration.transcribe()
        → WhisperVaultTranscriber.ensure_reload()   # diff & POST /reload if needed
        → WhisperVaultTranscriber.transcribe()       # POST /transcribe, save outputs
  → merge chunked results if applicable
```

### Configuration JSON format

```json
{
  "run_configuration": [
    {
      "description": "swedish_kb_vad_0.3",
      "language": "swedish",
      "model": "kb-whisper-large-ct2",
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

Key config fields:

| Field | Notes |
|---|---|
| `model` | Short alias (`kb-whisper-large-ct2`, `large-v3`, …) **or** full server path (`/models/extra/…`) |
| `language` | Full name (`"swedish"`, `"english"`) or ISO-639-1 (`"sv"`, `"en"`), or `"Automatic Detection"` |
| `vad` | `true`/`false` — maps to `vad_onset` on the server (false → 0.01) |
| `vad_speech_threshold` | Float, used as `vad_onset` when `vad=true` |
| `diarize` | `true` to enable speaker diarization |
| `condition_on_prev_text` | Maps to `condition_on_previous_text` reload param |
| `beam_size`, `repetition_penalty`, `initial_prompt`, `hotwords` | All reload params |
| `file_format` | `"SRT"` → server produces both `srt` and `txt`; output saved to disk |

### Reload vs per-request params

**Reload params** are baked into the loaded model — `WhisperVaultTranscriber.ensure_reload()`
diffs the desired state against a cached copy and only calls `POST /reload` when something
actually changed. The reload is triggered once per configuration block, not per file.

**Per-request params** (`language`, `output_format`, `diarize`, `min_speakers`, `max_speakers`)
are sent fresh with every `POST /transcribe` and never require a reload.

---

## WhisperVault server

- **Endpoint:** `https://whispervault.berra.humlab.umu.se`
- **Auth:** HTTP Basic, same credentials as `BASIC_AUTH_USERNAME` / `BASIC_AUTH_PASSWORD` in `.env`
- **API surface:** `GET /health`, `GET /models`, `GET /params`, `POST /transcribe`, `POST /reload`
- The server runs WhisperX inside a network-isolated podman container; it is managed separately.

### Available models (as of 2026-03-05)

| Config short name | Server path |
|---|---|
| `kb-whisper-large-ct2` | `/models/extra/kb-whisper-large-ct2` |
| `large-v3` | `/models/extra/faster-whisper-large-v3-ct2` |
| `large-v2` | `/models/extra/faster-whisper-large-v2-ct2` (check if present) |
| `whisper-large-v3-turbo` | `/models/extra/faster-whisper-large-v3-turbo-ct2` (check if present) |

You can also use the full `/models/extra/…` path directly in config files — the alias lookup is
skipped and the path is passed to the server as-is.

---

## Environment variables (`.env`)

```dotenv
BASIC_AUTH_USERNAME=          # HTTP Basic auth username for WhisperVault
BASIC_AUTH_PASSWORD=          # HTTP Basic auth password
WHISPERX_ENDPOINT=https://whispervault.berra.humlab.umu.se
GRADIO_WHISPERX_ENDPOINT=     # Legacy — used as fallback if WHISPERX_ENDPOINT is unset
HF_TOKEN=                     # HuggingFace token (used by server, not client)
```

Copy `.env.example` to `.env` and fill in the values. Never commit `.env`.

---

## Key conventions

### Language / style
- Python 3.10+, type hints in all new code
- `httpx` (not `requests`) for all HTTP
- No Gradio, no `gradio_client` — those dependencies have been removed

### Pre-commit

The venv lives at `./venv/` — activate it before any Python work:

```bash
source venv/bin/activate
```

Hooks: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-json`,
`check-toml`, `debug-statements`, `black` (line length 120), `isort`, `flake8`.

**Always run pre-commit before committing, and after making code changes:**

```bash
pre-commit run --all-files
```

If pre-commit is not yet installed in the venv:

```bash
pip install pre-commit
pre-commit install
```

Agents must run `pre-commit run --all-files` after any code edit and fix all
failures before considering a task complete.

### Git
- Always use plain `git` CLI commands — never use MCP git tools.
- Stage and commit via `git add` / `git commit`; check status with `git status` / `git diff`.

### Commit messages
- One concise subject line — what changed and why.
- No line counts, file counts, or before/after comparisons.
- Good: `fix: resolve model alias for large-v3 to correct /models/extra path`
- Good: `feat: add WhisperVaultTranscriber to replace Gradio backend`
- Bad: *(updated 3 files, added 42 lines)*

---

## Common commands

### Run transcription

```bash
source venv/bin/activate
python run_whisper.py /path/to/project --configuration configurations/large_default_swedish.json
```

The project directory must contain a `raw_audio/` subdirectory with the audio files.
Outputs land in `<project>/transcriptions/`.

### Key flags

```
--configuration   Path to JSON config file (or name under configurations/)
--run-description Run only the config entry with this description
--tag             Extra subdirectory label under transcriptions/
--max-chunk-duration  Split long files, e.g. "20m"
--force-convert   Re-convert even if WAV already exists
--dry-run         Show what would happen without doing it
--no-config-logs  Skip writing .log.json files alongside transcriptions
```

### Check server status

```bash
python -c "
import os, httpx
from dotenv import load_dotenv
load_dotenv()
auth=(os.getenv('BASIC_AUTH_USERNAME'), os.getenv('BASIC_AUTH_PASSWORD'))
r = httpx.get(os.getenv('WHISPERX_ENDPOINT')+'/health', auth=auth)
print(r.json())
"
```

---

## What NOT to do
- Do not add `gradio` or `gradio_client` back as dependencies
- Do not commit `.env` or audio files
- Do not skip `pre-commit run --all-files` before committing
- Do not use `requests` — use `httpx`
- Do not hardcode the WhisperVault endpoint URL in source files — read from env
