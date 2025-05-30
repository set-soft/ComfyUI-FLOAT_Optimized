#!/bin/bash

# Get the directory to search in. Defaults to the current directory if no argument is provided.
search_dir="${1:-.}"

# Normalize the search directory path (e.g., remove trailing slashes if any)
# and ensure it's a valid directory.
search_dir=$(realpath "$search_dir")
if [ ! -d "$search_dir" ]; then
    echo "Error: '$search_dir' is not a valid directory."
    exit 1
fi

# Use find to locate all .py files recursively
# -print0 and read -d $'\0' are used to handle filenames with spaces or special characters.
find "$search_dir" -type f -name "*.py" -print0 | while IFS= read -r -d $'\0' filepath; do
    # Get the relative path from the current execution directory
    # realpath --relative-to=. "$filepath" gives the path relative to current dir
    # If you want it relative to $search_dir, use --relative-to="$search_dir"
    relative_filepath=$(realpath --relative-to="." "$filepath")

    # Output the header
    echo "Content of $relative_filepath"

    # Output the opening backticks
    echo "\`\`\`" # Escape the backticks so they are printed literally

    # Output the file content
    cat "$filepath"

    # Output the closing backticks
    echo "\`\`\`"

    # Add a blank line for separation (optional)
    echo ""
done
