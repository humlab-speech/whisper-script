#!/bin/bash
echo "Hello, World!"

python3 -m venv venv/
source venv/bin/activate
python3 -m pip install pysrt

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if a directory has been provided
if [ -z "\$1" ]; then
  echo "Please provide a directory"
  exit 1
fi

echo "Let's go"

# Run the Python script on each .srt file in the provided directory
find $1 -name "*.srt" -exec python "$DIR/srt_to_txt.py" {} \;
