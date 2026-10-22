"""Shared filter for system/junk files found on SD cards and network shares.

macOS writes AppleDouble resource forks (._Track.mp3), .DS_Store, .Trashes/ etc.
next to real audio; Windows Office writes ~$ lock files. pathlib glob/rglob match
dot-files (unlike shell globbing), so every scanner in this repo must filter them.
"""

# Directories that are pure OS metadata and can never contain user audio.
# Deliberately an explicit list: hidden dirs with unknown names (e.g. a user's
# .recordings/) must NOT be silently dropped from a scan.
JUNK_DIR_NAMES = frozenset(
    {
        ".DS_Store",
        ".Trashes",
        ".Trash",
        ".Spotlight-V100",
        ".fseventsd",
        ".AppleDouble",
        ".AppleDB",
        ".AppleDesktop",
        ".TemporaryItems",
        ".DocumentRevisions-V100",
        ".vol",
    }
)


def is_junk_file(name: str) -> bool:
    """Return True for macOS AppleDouble (._*, .__*), any other dot-file
    (incl. .DS_Store) and Windows Office temp/lock files (~$*)."""
    return name.startswith(".") or name.startswith("~$")


def is_junk_dir(name: str) -> bool:
    """Return True for directories that are known OS metadata (never user data)."""
    return name.startswith("._") or name.startswith("~$") or name in JUNK_DIR_NAMES
