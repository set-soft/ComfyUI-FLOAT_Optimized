#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# --- Configuration ---
# The path to your pyproject.toml file, relative to the project root
PYPROJECT_PATH = Path("pyproject.toml")
# The path to the Python file containing the __version__ string
SOURCE_VERSION_PATH = Path("src/nodes/__init__.py")
# --- End Configuration ---


def get_version_from_pyproject(file_path: Path) -> str | None:
    """Extracts the version string from a pyproject.toml file."""
    try:
        content = file_path.read_text()
        # A simple regex to find `version = "..."` under the `[project]` table
        match = re.search(r'^version\s*=\s*"(.*?)"', content, re.MULTILINE)
        if match:
            return match.group(1)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.", file=sys.stderr)
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}", file=sys.stderr)
    return None


def get_version_from_source(file_path: Path) -> str | None:
    """Extracts the __version__ string from a Python source file."""
    try:
        content = file_path.read_text()
        # A simple regex to find `__version__ = "..."`
        match = re.search(r'^__version__\s*=\s*"(.*?)"', content, re.MULTILINE)
        if match:
            return match.group(1)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.", file=sys.stderr)
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}", file=sys.stderr)
    return None


def main() -> int:
    """
    Compares version strings from pyproject.toml and the source code.
    Exits with a non-zero status code if they do not match.
    """
    print("--- Checking version consistency ---")

    # Get versions
    pyproject_version = get_version_from_pyproject(PYPROJECT_PATH)
    source_version = get_version_from_source(SOURCE_VERSION_PATH)

    # Validate that we found both
    if not pyproject_version:
        print(f"Error: Could not find version in {PYPROJECT_PATH}", file=sys.stderr)
        return 1

    if not source_version:
        print(f"Error: Could not find `__version__` in {SOURCE_VERSION_PATH}", file=sys.stderr)
        return 1

    print(f"Version in {PYPROJECT_PATH}: {pyproject_version}")
    print(f"Version in {SOURCE_VERSION_PATH}: {source_version}")

    # Compare and exit
    if pyproject_version == source_version:
        print("✅ Versions are consistent.")
        return 0
    else:
        print("\n❌ Error: Version mismatch!", file=sys.stderr)
        print(f"  pyproject.toml has version '{pyproject_version}'", file=sys.stderr)
        print(f"  {SOURCE_VERSION_PATH} has version '{source_version}'", file=sys.stderr)
        print("  Please ensure both versions are identical.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
