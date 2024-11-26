demo_config = """
{
    "run_configuration": [    
        {
            "description": "swedish_large_v2_with_vad_0.5",
            "language": "swedish",
            "model": "large-v2",
            "file_format": "SRT",
            "condition_on_prev_text": false,
            "vad": true,
            "vad_speech_threshold": 0.5,
            "subfolder": "safe_swedish/large_v2/vad/0.5"
        }
    ]
}
"""
