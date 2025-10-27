#!/bin/bash

# Check if a directory has been provided
if [ $# -eq 0 ]; then
    echo "No directory supplied. Usage: ./script.sh /path/to/directory"
    exit 1
fi

# Get the directory from the command-line argument
dir="$1"

# Create a subdirectory for the normalized files
mkdir -p "$dir/normalized"

# Loop through all WAV files in the directory
for file in "$dir"/*.wav; do
  # Extract the base filename without the extension
  base=$(basename "$file" .wav)

  # Normalize the file and save it in the subdirectory
  ffmpeg -i "$file" -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary "$dir/normalized/${base}_normalized.wav"
done
