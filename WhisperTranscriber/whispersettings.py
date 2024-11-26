class WhisperSettings:
    """
    This is a settings class that tries to convert the often obscure names of the settings from
    the Whisper API to more human-readable names. This class can be used to use these more readable
    names and change the default settings, for further transcribation using Whisper.

    """

    @property
    def setting_keys(self):
        if not hasattr(self, "_cached_setting_keys"):
            self._cached_setting_keys = self.settings.keys()
        return self._cached_setting_keys

    def __init__(self):
        self.settings = {
            "upload_file": ["files", None],  # must be provided manually
            "input_folder": ["input_folder_path", ""],
            "file_format": ["file_format", "SRT"],
            "add_timestamp": ["add_timestamp", True],
            "model": ["progress", "large-v2"],
            "language": ["param_5", "Automatic Detection"],
            "translate_to_english": ["param_6", False],
            "beam_size": ["param_7", 5],
            "log_prob_threshold": ["param_8", -1],
            "no_speech_threshold": ["param_9", 0.6],
            "compute_type": ["param_10", "float16"],
            "best_of": ["param_11", 5],
            "patience": ["param_12", 1],
            "condition_on_prev_text": ["param_13", True],
            "prompt_reset_on_temp": ["param_14", 0.5],
            "initial_prompt": ["param_15", ""],
            "temperature": ["param_16", 0],
            "compression_ratio_threshold": ["param_17", 2.4],
            "length_penalty": ["param_18", 1],
            "repetition_penalty": ["param_19", 1],
            "no_repeat_ngram_size": ["param_20", 0],
            "prefix": ["param_21", ""],
            "suppress_blank": ["param_22", True],
            "suppress_tokens": ["param_23", "[-1]"],
            "max_initial_timestamp": ["param_24", 1],
            "word_timestamps": ["param_25", False],
            "prepend_punctuations": ["param_26", "\"'“¿([{-"],
            "append_punctuations": ["param_27", "\"'.。,，!！?？:：”)]}、"],
            "max_new_tokens": ["param_28", 0],
            "chunk_length_s": ["param_29", 30],
            "hallucination_silence_threshold_sec": ["param_30", 0],
            "hotwords": ["param_31", ""],
            "language_detection_threshold": ["param_32", 0],
            "language_detection_segments": ["param_33", 1],
            "batch_size": ["param_34", 24],
            "enable_silero_vad_filter": ["param_35", False],
            "vad_speech_threshold": ["param_36", 0.5],
            "vad_min_speech_duration_ms": ["param_37", 250],
            "vad_max_speech_duration_s": ["param_38", 9999],
            "vad_min_silence_duration_ms": ["param_39", 1000],
            "vad_speech_padding_ms": ["param_40", 2000],
            "enable_diarization": ["param_41", False],
            "device_1": ["param_42", "cuda"],
            "huggingface_token": ["param_43", ""],
            "enable_bg_music_remover_filter": ["param_44", False],
            "bg_music_remover_model": ["param_45", "UVR-MDX-NET-Inst_HQ_4"],
            "device_2": ["param_46", "cuda"],
            "segment_size": ["param_47", 256],
            "save_separated_files_to_output": ["param_48", False],
            "offload_sub_model_after_removing_bg_music": ["param_49", True],
            "api_name": ["api_name", "/transcribe_file"],
        }

    def __getattribute__(self, name):
        # First check if 'settings' has been set yet
        if "settings" in object.__getattribute__(self, "__dict__"):
            # If it has, check if the attribute is in 'settings'
            settings = object.__getattribute__(self, "settings")
            if name in settings:
                return settings[name][1]

        # If 'settings' hasn't been set yet or the attribute is not in 'settings',
        # use the default attribute access method
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        # First check if 'settings' has been set yet
        if "settings" in object.__getattribute__(self, "__dict__"):
            # If it has, check if the attribute is in 'settings'
            settings = object.__getattribute__(self, "settings")
            if name in settings:
                # If the attribute is in 'settings', set the value there
                settings[name][1] = value
                return
        object.__setattr__(self, name, value)

    def get_settings(self):
        return {self.settings[key][0]: self.settings[key][1] for key in self.settings}

    def get_possible_settings(self):
        return [x[1] for x in self.settings.values()]

    def __getitem__(self, key):
        return self.settings[key][1]

    def __setitem__(self, key, value):
        if key in self.settings:
            self.settings[key][1] = value
        else:
            raise KeyError(f"Invalid setting key: {key}")
