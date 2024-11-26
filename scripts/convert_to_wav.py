#!/usr/bin/python3

import os
import subprocess
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_file(file, output_dir):
    output = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".wav")
    if os.path.exists(output):
        logging.info(f"File {output} already exists. Skipping conversion.")
        return
    try:
        subprocess.run(["ffmpeg", "-i", file, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", output], check=True)
        logging.info(f"Converted {file} to {output}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {file}: {e}")

def process_directory(input_dir, extensions, output_dir):
    for root, dirs, files in os.walk(input_dir):
        if output_dir in dirs:
            dirs.remove(output_dir)  # don't visit the output directory
        for file in files:
            if any(file.lower().endswith(extension) for extension in extensions):
                convert_file(os.path.join(root, file), output_dir)

def convert_audio(input_path, extensions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        extensions = extensions.lower().split(',')
        process_directory(input_path, extensions, output_dir)
    elif os.path.isfile(input_path):
        convert_file(input_path, output_dir)
    else:
        logging.error("Input is not a valid file or directory.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Convert audio files to 16 kHz mono .wav')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--extensions', default='wma,wmv,mp3,mp4,mkv,aac,flac,ogg,m4a,wav,avi,mov,flv,mpeg,mpg,webm', help='File extensions to convert, separated by comma')
    parser.add_argument('--output', default='converted_wavs', help='Output directory')

    args = parser.parse_args()
    convert_audio(args.input, args.extensions, args.output)

if __name__ == "__main__":
    main()