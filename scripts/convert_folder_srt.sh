#!/bin/bash
# Check if a directory has been provided
if [ -z "$1" ]; then
  echo "Please provide a directory"
  exit 1
fi

echo "Hello, World!"

python3 -m venv venv/
source venv/bin/activate
python3 -m pip install pysrt

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Let's go"

# Run the Python script on each .srt file in the provided directory.
# -mindepth 1 + basename match: prune dot-paths without pruning the argument
# itself (a dot-named root is user data, same rule as junk_filter.is_junk_path).
# Unlike the Python scanners (which keep unknown hidden dirs) this prunes every
# dot-dir, because find cannot read the junk_filter deny-list.
find "$1" -mindepth 1 -name '.*' -prune -o -name "*.srt" ! -name '~$*' -exec python "$DIR/srt_to_txt.py" {} \;
