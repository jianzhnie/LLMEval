import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple


def load_and_validate_jsonl(file_path: str) -> List[str]:
    """Load JSONL file and return valid JSON lines.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of valid JSON strings

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    valid_lines = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)  # Validate JSON
                    valid_lines.append(line)
                except json.JSONDecodeError as e:
                    print(
                        f'⚠️  Line {line_num} JSON error in {Path(file_path).name}: {e}'
                    )
    except FileNotFoundError:
        print(f'❌ File not found: {file_path}')
        raise

    return valid_lines


def merge_jsonl_files(folder1: str, folder2: str, output_dir: str) -> None:
    """Merge JSONL files from two folders.

    Args:
        folder1: First folder path
        folder2: Second folder path
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get JSONL files from both folders
    files1 = {
        Path(f).name: f
        for f in glob.glob(os.path.join(folder1, '*.jsonl'))
    }
    files2 = {
        Path(f).name: f
        for f in glob.glob(os.path.join(folder2, '*.jsonl'))
    }
    all_files = set(files1.keys()) | set(files2.keys())

    if not all_files:
        print('⚠️  No JSONL files found')
        return

    total_lines = 0

    for filename in all_files:
        output_path = os.path.join(output_dir, filename)
        file_lines = 0

        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                # Process files from both folders
                for folder_files in [files1, files2]:
                    if filename in folder_files:
                        valid_lines = load_and_validate_jsonl(
                            folder_files[filename])
                        for line in valid_lines:
                            out_f.write(line + '\n')
                        file_lines += len(valid_lines)

            print(f'✅ Merged: {filename} ({file_lines} lines)')
            total_lines += file_lines

        except Exception as e:
            print(f'❌ Error processing {filename}: {e}')
            continue

    print(
        f"✅ Merge complete! {len(all_files)} files, {total_lines} lines total, saved to '{output_dir}'"
    )


def validate_jsonl_file(file_path: str) -> Tuple[int, int]:
    """Validate JSONL file format.

    Args:
        file_path: Path to validate

    Returns:
        Tuple of (valid_lines, invalid_lines)
    """
    valid_lines = 0
    invalid_lines = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError:
                    invalid_lines += 1
                    print(f'❌ Line {line_num} JSON error: {line[:100]}...')
    except Exception as e:
        print(f'❌ Error reading {file_path}: {e}')
        return 0, 0

    return valid_lines, invalid_lines


def main():
    parser = argparse.ArgumentParser(
        description='Merge JSONL files from two folders with JSON validation')
    parser.add_argument('--folder1', required=True, help='First folder path')
    parser.add_argument('--folder2', required=True, help='Second folder path')
    parser.add_argument('--output', help='Output folder path (optional)')
    parser.add_argument('--validate',
                        action='store_true',
                        help='Validate output files after merge')

    args = parser.parse_args()

    # Generate output directory name if not provided
    if not args.output:
        output_dir = f'{Path(args.folder1).name}_{Path(args.folder2).name}_merged'
    else:
        output_dir = args.output

    merge_jsonl_files(args.folder1, args.folder2, output_dir)

    # Validate output files if requested
    if args.validate:
        print('\n🔍 Validating output files...')
        for file_path in glob.glob(os.path.join(output_dir, '*.jsonl')):
            valid, invalid = validate_jsonl_file(file_path)
            print(
                f'📊 {Path(file_path).name}: Valid {valid}, Invalid {invalid}')


if __name__ == '__main__':
    main()
