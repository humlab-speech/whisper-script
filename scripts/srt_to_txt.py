import pysrt
import os
import sys


def srt_to_txt(srt_file_path):
    # Load the .srt file
    subs = pysrt.open(srt_file_path)

    # Create the .txt file path
    base = os.path.splitext(srt_file_path)[0]
    txt_file_path = base + ".txt"

    # Open the .txt file (or create it if it doesn't exist)
    with open(txt_file_path, "w") as txt_file:
        # Loop through each subtitle in the .srt file
        for sub in subs:
            # Write the subtitle text to the .txt file
            txt_file.write(sub.text + "\n")


def main():
    srt_to_txt(sys.argv[1])


if __name__ == "__main__":
    main()
