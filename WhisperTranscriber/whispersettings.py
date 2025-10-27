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
            # Direct API mappings (no "param_X" name)
            "upload_file": ["files", None],  # must be provided manually
            "input_folder": ["input_folder_path", ""],
            "include_subdirectory": ["include_subdirectory", False],  # New
            "save_same_dir": ["save_same_dir", True],  # New
            "file_format": ["file_format", "SRT"],
            "add_timestamp": ["add_timestamp", False],  # Default changed from True
            "model": ["progress", "large-v2"],  # API calls this "progress"
            # Parameterized API settings
            "language": ["param_7", "Automatic Detection"],  # Was param_5
            "translate_to_english": ["param_8", False],  # Was param_6
            "beam_size": ["param_9", 5],  # Was param_7
            "log_prob_threshold": ["param_10", -1],  # Was param_8
            "no_speech_threshold": ["param_11", 0.6],  # Was param_9
            "compute_type": ["param_12", "float16"],  # Was param_10
            "best_of": ["param_13", 5],  # Was param_11
            "patience": ["param_14", 1],  # Was param_12
            "condition_on_prev_text": ["param_15", True],  # Was param_13
            "prompt_reset_on_temp": ["param_16", 0.5],  # Was param_14
            "initial_prompt": ["param_17", ""],  # Was param_15, API default "Hello!!", kept ""
            "temperature": ["param_18", 0],  # Was param_16
            "compression_ratio_threshold": ["param_19", 2.4],  # Was param_17
            "length_penalty": ["param_20", 1],  # Was param_18
            "repetition_penalty": ["param_21", 1],  # Was param_19
            "no_repeat_ngram_size": ["param_22", 0],  # Was param_20
            "prefix": ["param_23", ""],  # Was param_21, API default "Hello!!", kept ""
            "suppress_blank": ["param_24", True],  # Was param_22
            "suppress_tokens": ["param_25", "[-1]"],  # Was param_23
            "max_initial_timestamp": ["param_26", 1],  # Was param_24
            "word_timestamps": ["param_27", False],  # Was param_25
            "prepend_punctuations": ["param_28", "\"'“¿([{-"],  # Was param_26
            "append_punctuations": ["param_29", "\"'.。,，!！?？:：”)]}、"],  # Was param_27
            "max_new_tokens": ["param_30", 0],  # Was param_28, API default 3, kept 0
            "chunk_length_s": ["param_31", 30],  # Was param_29
            "hallucination_silence_threshold_sec": ["param_32", 0],  # Was param_30, API default 3, kept 0
            "hotwords": ["param_33", ""],  # Was param_31, API default "Hello!!", kept ""
            "language_detection_threshold": ["param_34", 0.5],  # Was param_32, default changed from 0
            "language_detection_segments": ["param_35", 1],  # Was param_33
            "batch_size": ["param_36", 24],  # Was param_34
            "offload_whisper_model": ["param_37", True],  # New - "Offload sub model when finished"
            "enable_silero_vad_filter": ["param_38", False],  # Was param_35
            "vad_speech_threshold": ["param_39", 0.5],  # Was param_36
            "vad_min_speech_duration_ms": ["param_40", 250],  # Was param_37
            "vad_max_speech_duration_s": ["param_41", 9999],  # Was param_38
            "vad_min_silence_duration_ms": ["param_42", 1000],  # Was param_39
            "vad_speech_padding_ms": ["param_43", 2000],  # Was param_40
            "enable_diarization": ["param_44", False],  # Was param_41
            "whisper_device": ["param_45", "cuda"],  # Was device_1 (param_42)
            "huggingface_token": ["param_46", ""],  # Was param_43
            "offload_diarization_model": ["param_47", True],  # New - "Offload sub model when finished"
            "enable_bg_music_remover_filter": ["param_48", False],  # Was param_44
            "bg_music_remover_model": ["param_49", "UVR-MDX-NET-Inst_HQ_4"],  # Was param_45
            "bg_remover_device": ["param_50", "cuda"],  # Was device_2 (param_46)
            "segment_size": ["param_51", 256],  # Was param_47 (BG remover segment size)
            "save_separated_files_to_output": ["param_52", False],  # Was param_48
            "offload_sub_model_after_removing_bg_music": ["param_53", True],  # Was param_49
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
