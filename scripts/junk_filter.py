"""Shared filter for system/junk files found on SD cards and network shares.

macOS writes AppleDouble resource forks (._Track.mp3), .DS_Store, .Trashes/ etc.
next to real audio; Windows writes ~$ lock files and $RECYCLE.BIN/; Linux writes
.Trash-<uid>/. pathlib glob/rglob match dot-files (unlike shell globbing) and
find's -name matches them too, so every scanner in this repo must filter them
(one exception: convert_folder_srt.sh prunes *every* dot-path, because find
cannot read the deny-list below).
"""

from pathlib import Path

# Directories that are pure OS metadata and can never contain user audio.
# Deliberately an explicit list (plus the .Trash* prefix for the uid-suffixed
# variants): hidden dirs with unknown names (e.g. a user's .recordings/) must
# NOT be silently dropped from a scan.
JUNK_DIR_NAMES = frozenset(
    {
        ".Spotlight-V100",
        ".fseventsd",
        ".AppleDouble",
        ".AppleDB",
        ".AppleDesktop",
        ".TemporaryItems",
        ".DocumentRevisions-V100",
        ".vol",
        "$RECYCLE.BIN",  # Windows / SMB recycle bin on the same share
    }
)

# Wording of the per-scanner skip tally, kept here so the scanners cannot drift.
# "item(s)": the os.walk scanners count a pruned junk directory, the rglob-based
# ones (run_whisper, strip_speakers) count the junk files inside it.
JUNK_SKIPPED_MSG = "Skipped {count} system/junk item(s) (AppleDouble, dot-files, lock files, OS trash)."


def is_junk_file(name: str) -> bool:
    """Return True for macOS AppleDouble (._*, .__*), any other dot-file
    (incl. .DS_Store) and Windows Office temp/lock files (~$*)."""
    return name.startswith(".") or name.startswith("~$")


def is_junk_dir(name: str) -> bool:
    """Return True for directories that are known OS metadata (never user data)."""
    return name.startswith("._") or name.startswith("~$") or name.startswith(".Trash") or name in JUNK_DIR_NAMES


def is_junk_path(path: Path, root: Path) -> bool:
    """Return True if *path* is a junk file, or sits under a junk directory.

    *root* itself is never judged, so scanning a dot-named folder directly
    (e.g. a project inside .private/) still finds the audio inside it.  *path*
    must be below *root* as produced by glob/rglob/os.walk of that same root.
    """
    parts = Path(path).relative_to(root).parts
    return is_junk_file(parts[-1]) or any(is_junk_dir(part) for part in parts[:-1])
