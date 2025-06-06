#!/usr/bin/python3
import argparse
import os
import shutil  # For file backup
import logging

# Setup logging
logger = logging.getLogger("StyleFixer")
logger.setLevel(logging.INFO)  # Default level
# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # Let handler decide what to show based on logger's effective level
# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add formatter to ch
ch.setFormatter(formatter)
# Add ch to logger
logger.addHandler(ch)


def expand_tabs(line, tab_size=4):
    """Expands tabs to spaces, respecting tab stops."""
    out_line = []
    current_col = 0
    for char in line:
        if char == '\t':
            spaces_to_add = tab_size - (current_col % tab_size)
            out_line.append(' ' * spaces_to_add)
            current_col += spaces_to_add
        else:
            out_line.append(char)
            current_col += 1
    return "".join(out_line)


def fix_indentation(line, tab_size=4):
    """Rounds leading whitespace indentation to the nearest multiple of tab_size."""
    stripped_line = line.lstrip(' ')
    if not stripped_line or stripped_line == line:  # No leading spaces or empty line
        return line

    leading_spaces_count = len(line) - len(stripped_line)

    remainder = leading_spaces_count % tab_size
    if remainder == 0:
        new_indent_count = leading_spaces_count
    elif remainder <= tab_size / 2:  # Prioritize rounding down or to current if exact multiple
        new_indent_count = leading_spaces_count - remainder
    else:  # Round up
        new_indent_count = leading_spaces_count + (tab_size - remainder)

    new_indent_count = max(0, new_indent_count)  # Ensure non-negative

    if new_indent_count != leading_spaces_count:
        logger.debug(f"Adjusting indent: from {leading_spaces_count} to {new_indent_count} for line: {line.rstrip()!r}")

    return ' ' * new_indent_count + stripped_line


def process_line(line_num, line_content_with_eol, expand_tabs_enabled=True):
    original_line_for_debug = line_content_with_eol  # Keep for debugging if needed
    line = line_content_with_eol

    # 1. Expand tabs (if enabled)
    if expand_tabs_enabled:
        line = expand_tabs(line)
        if line != original_line_for_debug and not line.isspace():  # Log only if actual content changed
            logger.debug(f"L{line_num}: Tabs expanded.")

    # 2. W291: Remove trailing whitespace (done before other checks that might rely on EOL)
    # Also handles W293 (blank line contains whitespace) implicitly if the line becomes empty.
    processed_line = line.rstrip()
    if len(processed_line) < len(line.rstrip('\r\n')):  # Compare lengths without EOL
        logger.debug(f"L{line_num}: Trailing whitespace removed.")

    # If the line became empty after rstrip, it's a truly blank line
    if not processed_line and line.strip() == "":
        # Return an empty string for a blank line, newline char will be added later if original had it
        if line.endswith(('\n', '\r\n', '\r')):
            return '\n'
        return ""

    # 3. Fix indentation (E111) - only if line is not blank
    if processed_line:
        indented_line = fix_indentation(processed_line)
        if indented_line != processed_line:  # Log if indentation changed
            # Already logged inside fix_indentation with more detail
            pass
        processed_line = indented_line

    # 4. E261: At least two spaces before inline comment
    in_string_char = None
    temp_part = ""
    comment_starts_at = -1

    for idx, char_val in enumerate(processed_line):
        if in_string_char:
            temp_part += char_val
            if char_val == in_string_char:
                if len(temp_part) >= 2 and temp_part[-2] == '\\':
                    pass
                else:
                    in_string_char = None
        elif char_val in ("'", '"'):
            temp_part += char_val
            in_string_char = char_val
        elif char_val == '#':
            comment_starts_at = idx
            break  # Found the first non-string comment char
        else:
            temp_part += char_val

    if comment_starts_at != -1:
        code_part = processed_line[:comment_starts_at]
        comment_part = processed_line[comment_starts_at:]  # Includes '#'

        rstripped_code_part = code_part.rstrip()
        # Only add spaces if there's actual code before the comment
        if rstripped_code_part:
            if not code_part.endswith('  '):  # Needs fixing (less than 2 spaces)
                new_line_with_comment_spacing = rstripped_code_part + '  ' + comment_part
                if new_line_with_comment_spacing != processed_line:
                    logger.debug(f"L{line_num}: Adjusted inline comment spacing.")
                processed_line = new_line_with_comment_spacing
        # If rstripped_code_part is empty, it's a comment at the start of the line (after indent), leave it.

    # Add back the newline character that rstrip might have removed, if the original line had one
    # or if it's not an empty line that was purely whitespace
    if line_content_with_eol.endswith('\n'):
        return processed_line + '\n'
    elif line_content_with_eol.endswith('\r\n'):
        return processed_line + '\r\n'
    elif line_content_with_eol.endswith('\r'):
        return processed_line + '\r'
    else:  # Original line did not end with EOL
        return processed_line


def process_file(filepath, expand_tabs_enabled=True, dry_run=False, no_backup=False):
    logger.info(f"Processing {filepath}...")
    try:
        # Read with universal newlines mode, then splitlines to preserve EOLs correctly for rejoining
        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            # original_content = f.read() # Reading all at once
            # original_lines = original_content.splitlines(keepends=True) # This is better
            original_lines = f.readlines()  # Simpler, usually works fine
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return False

    fixed_lines = []
    changes_made = False
    for i, line_content_with_eol in enumerate(original_lines):
        fixed_line_content = process_line(i + 1, line_content_with_eol, expand_tabs_enabled)
        fixed_lines.append(fixed_line_content)
        if fixed_line_content != line_content_with_eol:
            changes_made = True
            logger.debug(f"L{i+1}: Original: {line_content_with_eol.rstrip()!r}")
            logger.debug(f"L{i+1}: Fixed   : {fixed_line_content.rstrip()!r}")

    if changes_made:
        if dry_run:
            logger.info(f"Would fix style issues in {filepath} (dry run)")
        else:
            if not no_backup:
                backup_dir = os.path.dirname(filepath)
                backup_filename = "." + os.path.basename(filepath) + "~"
                backup_path = os.path.join(backup_dir, backup_filename)
                try:
                    shutil.copy2(filepath, backup_path)  # copy2 preserves metadata
                    logger.info(f"Backup of original file created at {backup_path}")
                except Exception as e:
                    logger.error(f"Failed to create backup for {filepath}: {e}. File not modified.")
                    return False

            try:
                # Write back lines using the original EOLs if possible, or common EOL
                with open(filepath, 'w', encoding='utf-8', newline='') as f:
                    f.writelines(fixed_lines)
                logger.info(f"Fixed style issues in {filepath}")
            except Exception as e:
                logger.error(f"Error writing to file {filepath}: {e}")
                return False
    else:
        logger.info(f"No style issues to fix in {filepath}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Fix common Python code style issues (Flake8).")
    parser.add_argument("files", metavar="FILE", type=str, nargs='+',
                        help="Python files to process")
    parser.add_argument("--no-expand-tabs", action="store_false", dest="expand_tabs",
                        help="Disable expansion of tabs to spaces.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without modifying files.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not create a backup of the original file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose (debug) logging.")
    parser.set_defaults(expand_tabs=True)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    for filepath in args.files:
        if not os.path.isfile(filepath):
            logger.warning(f"File not found: {filepath}. Skipping.")
            continue
        process_file(filepath, args.expand_tabs, args.dry_run, args.no_backup)


if __name__ == "__main__":
    main()
