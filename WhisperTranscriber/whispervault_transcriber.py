"""
WhisperVaultTranscriber — HTTP client for the WhisperVault FastAPI backend.

Drop-in replacement for the Gradio-based Transcriber class.
Talks to POST /transcribe for transcription and POST /reload to ensure the
server is in the correct state before each configuration block is processed.

Configuration:
    WHISPERX_ENDPOINT   URL of the WhisperVault nginx sidecar, e.g.
                        http://130.239.57.54:8088
                        Falls back to GRADIO_WHISPERX_ENDPOINT for backwards
                        compatibility with existing .env files.

Key design decision — class-level shared state:
    The httpx client, base URL and cached server state are stored as class
    attributes rather than instance attributes.  This means creating a new
    WhisperVaultTranscriber() per file (as Configuration.transcribe() does)
    reuses the same TCP connection and never triggers a spurious reload.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv


class WhisperVaultTranscriber:
    """
    HTTP client for the WhisperVault FastAPI backend.

    Matches the public interface of the Gradio-based Transcriber so that
    Configuration.transcribe() works without modification.
    """

    # ------------------------------------------------------------------ #
    # Class-level shared state (shared across all instances)              #
    # ------------------------------------------------------------------ #

    _http_client: Optional[httpx.Client] = None
    _base_url: Optional[str] = None
    # Mirrors what the server currently has loaded (reload_param names → values).
    # Seeded from GET /health on first instantiation; updated after every /reload.
    _cached_reload_state: dict = {}

    # ------------------------------------------------------------------ #
    # Translation tables                                                   #
    # ------------------------------------------------------------------ #

    # Maps config-file short model names → server-side model identifiers.
    # Full /models/extra/ paths take priority; HF IDs are used as fallback.
    MODEL_ALIASES: dict = {
        "kb-whisper-large-ct2": "/models/extra/kb-whisper-large-ct2",
        "large-v3": "/models/extra/faster-whisper-large-v3-ct2",
        "large-v2": "/models/extra/faster-whisper-large-v2-ct2",
        "whisper-large-v3-turbo": "/models/extra/faster-whisper-large-v3-turbo-ct2",
        # Add new models here as they are installed on the server.
    }

    # Maps full language names (used in config files) → ISO-639-1 codes.
    # None means "auto-detect", which WhisperVault accepts as null.
    LANGUAGE_MAP: dict = {
        "automatic detection": None,
        "swedish": "sv",
        "english": "en",
        "finnish": "fi",
        "norwegian": "no",
        "danish": "da",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "dutch": "nl",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "japanese": "ja",
        "chinese": "zh",
        "arabic": "ar",
        "korean": "ko",
    }

    # Config key → reload_param name for params that map 1-to-1.
    # model, language, vad/vad_speech_threshold are handled separately.
    _RELOAD_PARAM_MAP: dict = {
        "beam_size": "beam_size",
        "condition_on_prev_text": "condition_on_previous_text",
        "initial_prompt": "initial_prompt",
        "hotwords": "hotwords",
        "best_of": "best_of",
        "patience": "patience",
        "length_penalty": "length_penalty",
        "temperature": "temperature",
        "no_speech_threshold": "no_speech_threshold",
        "log_prob_threshold": "logprob_threshold",
        "compression_ratio_threshold": "compression_ratio_threshold",
        "suppress_tokens": "suppress_tokens",
        "repetition_penalty": "repetition_penalty",
    }

    # Params that exist in config files or WhisperSettings but have no
    # equivalent in WhisperVault.  Silently dropped (with a warning for
    # repetition_penalty since it materially affects quality).
    _IGNORED_PARAMS: set = {
        "no_repeat_ngram_size",  # not exposed by server
        # repetition_penalty is now in _RELOAD_PARAM_MAP
        # no_repeat_ngram_size is not yet exposed
        "hallucination_silence_threshold_sec",
        "enable_silero_vad_filter",
        "vad_min_speech_duration_ms",
        "vad_max_speech_duration_s",
        "vad_min_silence_duration_ms",
        "vad_speech_padding_ms",
        "enable_diarization",  # use diarize= kwarg instead
        "offload_whisper_model",
        "offload_diarization_model",
        "enable_bg_music_remover_filter",
        "bg_music_remover_model",
        "bg_remover_device",
        "segment_size",
        "save_separated_files_to_output",
        "offload_sub_model_after_removing_bg_music",
        "whisper_device",
        "huggingface_token",
        "prompt_reset_on_temp",
        "word_timestamps",
        "prepend_punctuations",
        "append_punctuations",
        "max_initial_timestamp",
        "suppress_blank",
        "prefix",
        "max_new_tokens",
        "chunk_length_s",
        "language_detection_threshold",
        "language_detection_segments",
        "offload_whisper_model",
        # Internal coordination flags from Configuration.transcribe():
        "disable_args_file",
        "file_format",  # translated to output_format explicitly
        "subfolder",
        "description",
    }

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self):
        load_dotenv()
        self._ensure_client_initialized()

    # ------------------------------------------------------------------ #
    # Class-level initialisation                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def _ensure_client_initialized(cls) -> None:
        """
        Create the shared httpx client and seed server state from /health.
        No-op on subsequent calls.
        """
        if cls._http_client is not None:
            return

        endpoint = os.getenv("WHISPERX_ENDPOINT") or os.getenv("GRADIO_WHISPERX_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "WHISPERX_ENDPOINT environment variable not found. "
                "Set it to your WhisperVault HTTP endpoint, "
                "e.g.  WHISPERX_ENDPOINT=http://130.239.57.54:8088"
            )

        cls._base_url = endpoint.rstrip("/")

        # Basic auth is optional — only applied when both vars are present.
        username = os.getenv("BASIC_AUTH_USERNAME")
        password = os.getenv("BASIC_AUTH_PASSWORD")
        auth = httpx.BasicAuth(username, password) if (username and password) else None
        if auth:
            logging.info("WhisperVaultTranscriber: basic auth enabled")

        cls._http_client = httpx.Client(
            auth=auth,
            timeout=httpx.Timeout(600.0, connect=10.0),
        )
        logging.info(f"WhisperVaultTranscriber: using endpoint {cls._base_url}")
        cls._refresh_server_state()

    @classmethod
    def _refresh_server_state(cls) -> None:
        """
        Fetch /health and update the cached reload state.
        Called once at startup and again after any transcription error so the
        cache stays accurate if the server is reloaded externally.
        """
        try:
            r = cls._http_client.get(f"{cls._base_url}/health", timeout=10)
            r.raise_for_status()
            health = r.json()
        except Exception as exc:
            logging.warning(
                f"WhisperVaultTranscriber: could not reach /health: {exc}. "
                "Cache will be empty — a reload will be triggered on first use."
            )
            return

        # Only cache the fields that correspond to reload_param names.
        HEALTH_TO_RELOAD = {
            "model": "model",
            "device": "device",
            "compute_type": "compute_type",
            "language": "language",
            "vad_method": "vad_method",
        }
        for health_key, reload_key in HEALTH_TO_RELOAD.items():
            if health_key in health:
                cls._cached_reload_state[reload_key] = health[health_key]

        logging.info(f"WhisperVaultTranscriber: server state refreshed → {cls._cached_reload_state}")

    # ------------------------------------------------------------------ #
    # Translation helpers                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def _resolve_model(cls, short_name: str) -> str:
        """Map a config-file model short name to the server-side model path."""
        resolved = cls.MODEL_ALIASES.get(short_name, short_name)
        if resolved == short_name and not short_name.startswith(("/", "http")):
            logging.warning(
                f"WhisperVaultTranscriber: model '{short_name}' has no alias entry. "
                "Passing as-is — add it to MODEL_ALIASES if the server rejects it."
            )
        return resolved

    @classmethod
    def _resolve_language(cls, lang: Optional[str]) -> Optional[str]:
        """Convert a full language name or ISO code to the ISO-639-1 code the server expects."""
        if lang is None:
            return None
        lower = lang.lower()
        if lower in cls.LANGUAGE_MAP:
            return cls.LANGUAGE_MAP[lower]
        # Already an ISO code (2 chars) or unknown — pass through
        return lang if lang else None

    @classmethod
    def _resolve_vad(cls, vad: bool, vad_speech_threshold: float) -> dict:
        """
        Convert the old vad bool + threshold pair to the WhisperVault vad_onset
        reload param.

        vad=True  → vad_onset = vad_speech_threshold  (e.g. 0.3 or 0.5)
        vad=False → vad_onset = 0.01  (accept nearly all frames, effectively
                                        bypassing VAD filtering)
        """
        return {"vad_onset": vad_speech_threshold if vad else 0.01}

    @classmethod
    def _file_format_to_output_format(cls, file_format: Optional[str]) -> list:
        """
        Convert config file_format to a WhisperVault output_format list.
        Always includes 'txt' alongside 'srt' so callers that expect a .txt
        file continue to work without a separate conversion step.
        """
        if file_format is None:
            return ["srt", "txt"]
        fmt = file_format.strip().lower()
        if fmt == "srt":
            return ["srt", "txt"]
        if fmt in ("vtt", "tsv", "json", "aud"):
            return [fmt, "txt"]
        if fmt == "txt":
            return ["txt"]
        logging.warning(f"Unknown file_format '{file_format}', defaulting to ['srt','txt']")
        return ["srt", "txt"]

    # ------------------------------------------------------------------ #
    # Reload logic                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def ensure_reload(cls, desired: dict) -> None:
        """
        POST /reload with only the params that differ from the cached server
        state.  Does nothing if the server is already in the desired state.
        """
        diff = {k: v for k, v in desired.items() if cls._cached_reload_state.get(k) != v}
        if not diff:
            logging.info("WhisperVaultTranscriber: server already in desired state — skipping reload.")
            return

        logging.info(f"WhisperVaultTranscriber: reloading server with diff: {diff}")
        try:
            r = cls._http_client.post(
                f"{cls._base_url}/reload",
                json=diff,
                timeout=httpx.Timeout(180.0, connect=10.0),
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"WhisperVault /reload timed out: {exc}") from exc

        if r.status_code == 503:
            raise RuntimeError(
                "WhisperVault /reload returned 503 — another reload is already in progress. " "Wait a moment and retry."
            )
        r.raise_for_status()

        response_data = r.json()
        cls._cached_reload_state.update(response_data)
        logging.info(f"WhisperVaultTranscriber: reload complete → {response_data}")

    def _build_reload_dict(
        self,
        model: str,
        language: str,
        vad: bool,
        vad_speech_threshold: float,
        extra_kwargs: dict,
    ) -> dict:
        """
        Build the complete desired-server-state dict for ensure_reload().

        Includes model, language (as ISO code), VAD onset and every kwargs
        key that maps to a reload_param.
        """
        reload = {}
        reload["model"] = self._resolve_model(model)
        reload["language"] = self._resolve_language(language)
        reload.update(self._resolve_vad(vad, vad_speech_threshold))

        for config_key, reload_key in self._RELOAD_PARAM_MAP.items():
            if config_key in extra_kwargs:
                reload[reload_key] = extra_kwargs[config_key]

        return reload

    def _build_transcribe_dict(
        self,
        language: str,
        file_format: Optional[str],
        diarize: bool,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict:
        """
        Build the per-request params JSON for POST /transcribe.
        These are sent fresh with every file; they do not require a reload.
        """
        params: dict = {
            "language": self._resolve_language(language),
            "output_format": self._file_format_to_output_format(file_format),
        }
        if diarize:
            params["diarize"] = True
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers

        return params

    # ------------------------------------------------------------------ #
    # Core transcribe method                                               #
    # ------------------------------------------------------------------ #

    def transcribe(
        self,
        file_name: str,
        model: str = "large-v2",
        vad: bool = False,
        language: str = "Automatic Detection",
        output_path: str = "./output",
        tag: str = None,
        override_output_folder: str = None,
        settings=None,  # accepted but ignored (Gradio artefact)
        original_filename: str = None,
        disable_args_file: bool = False,
        **kwargs,
    ) -> None:
        """
        Transcribe a single audio file using the WhisperVault HTTP API.

        Signature is compatible with the original Gradio-based
        Transcriber.transcribe() so Configuration.transcribe() works unchanged.

        Before sending the audio, ensure_reload() is called to bring the server
        into the state required by this configuration.  Only the parameters that
        differ from the cached state trigger an actual /reload call.

        Args:
            file_name:              Path to the audio file to transcribe.
            model:                  Short model name, e.g. 'kb-whisper-large-ct2'.
            vad:                    Whether to apply VAD filtering.
            language:               Full language name or ISO code.
            output_path:            Base output directory (used if override_output_folder is None).
            tag:                    Unused; kept for interface compatibility.
            override_output_folder: If set, saves output here instead of output_path.
            settings:               Ignored (Gradio-era WhisperSettings object).
            original_filename:      Stem to use for output filenames.
            disable_args_file:      Internal flag, ignored.
            **kwargs:               Remaining config keys:
                                    vad_speech_threshold, file_format,
                                    condition_on_prev_text, beam_size,
                                    initial_prompt, hotwords, repetition_penalty,
                                    diarize, min_speakers, max_speakers, …
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")

        # --- warn about unknown params ----------------------------------------
        for key in kwargs:
            if (
                key not in self._IGNORED_PARAMS
                and key not in self._RELOAD_PARAM_MAP
                and key
                not in (
                    "vad_speech_threshold",
                    "file_format",
                    "diarize",
                    "min_speakers",
                    "max_speakers",
                )
            ):
                logging.debug(f"WhisperVaultTranscriber: unknown kwarg '{key}' will be ignored.")

        # --- extract per-invocation values from kwargs --------------------
        vad_speech_threshold = float(kwargs.get("vad_speech_threshold", 0.5))
        file_format = kwargs.get("file_format", "SRT")
        # Support both 'diarize' (new) and 'enable_diarization' (old Gradio name)
        diarize = bool(kwargs.get("diarize", kwargs.get("enable_diarization", False)))
        min_speakers = kwargs.get("min_speakers")
        max_speakers = kwargs.get("max_speakers")

        # --- ensure server state matches this config ----------------------
        reload_dict = self._build_reload_dict(
            model=model,
            language=language,
            vad=vad,
            vad_speech_threshold=vad_speech_threshold,
            extra_kwargs=kwargs,
        )
        self.ensure_reload(reload_dict)

        # --- per-request params -------------------------------------------
        transcribe_params = self._build_transcribe_dict(
            language=language,
            file_format=file_format,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # --- POST /transcribe ---------------------------------------------
        audio_path = Path(file_name)
        logging.info(f"WhisperVaultTranscriber: transcribing '{audio_path.name}' " f"params={transcribe_params}")

        try:
            with open(file_name, "rb") as audio_fh:
                response = self._http_client.post(
                    f"{self._base_url}/transcribe",
                    files={"audio": (audio_path.name, audio_fh, "audio/wav")},
                    data={"params": json.dumps(transcribe_params)},
                    timeout=httpx.Timeout(3600.0, connect=10.0),
                )
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Transcription request timed out for {file_name}: {exc}") from exc

        if not response.is_success:
            # Resync state in case the server was reloaded externally
            logging.warning(
                f"WhisperVaultTranscriber: /transcribe returned {response.status_code}. "
                "Refreshing server state cache."
            )
            self._refresh_server_state()
            response.raise_for_status()

        result = response.json()
        outputs: dict = result.get("outputs", {})

        if not outputs:
            raise RuntimeError(f"No outputs in /transcribe response for {file_name}. " f"Response: {result}")

        logging.info(
            f"WhisperVaultTranscriber: done. "
            f"language={result.get('language')}, "
            f"duration={result.get('duration_seconds', '?')}s, "
            f"formats={list(outputs.keys())}"
        )

        # --- save output files --------------------------------------------
        save_dir = override_output_folder or output_path
        os.makedirs(save_dir, exist_ok=True)

        stem = original_filename or audio_path.stem
        for fmt, content in outputs.items():
            out_path = os.path.join(save_dir, f"{stem}.{fmt}")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(content)
            logging.info(f"WhisperVaultTranscriber: saved {out_path}")

    # ------------------------------------------------------------------ #
    # Utility / diagnostic methods                                         #
    # ------------------------------------------------------------------ #

    def get_health(self) -> dict:
        """Return current server state from GET /health."""
        r = self._http_client.get(f"{self._base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def get_api_dict(self) -> dict:
        """Return the full parameter schema from GET /params."""
        r = self._http_client.get(f"{self._base_url}/params", timeout=10)
        r.raise_for_status()
        return r.json()

    def list_models(self) -> dict:
        """Return the model inventory from GET /models."""
        r = self._http_client.get(f"{self._base_url}/models", timeout=10)
        r.raise_for_status()
        return r.json()

    def reload_model(self, model: str, **reload_kwargs) -> dict:
        """
        Convenience method to hot-swap the server's ASR model.

        Args:
            model:          Short name (resolved via MODEL_ALIASES) or full path.
            **reload_kwargs: Any additional /reload params (beam_size, etc.).
        """
        payload = {"model": self._resolve_model(model), **reload_kwargs}
        self.ensure_reload(payload)
        return self._cached_reload_state.copy()
