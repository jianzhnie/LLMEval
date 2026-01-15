import argparse
import os


def rename_files(directory: str,
                 old_pattern: str,
                 new_pattern: str,
                 dry_run: bool = False) -> int:
    """
    批量重命名目录中的文件

    Args:
        directory: 目标目录路径
        old_pattern: 要替换的旧字符串
        new_pattern: 新字符串
        dry_run: 是否只预览不实际执行

    Returns:
        成功重命名的文件数量
    """
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return 0

    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是目录")
        return 0

    renamed_count = 0

    try:
        for filename in os.listdir(directory):
            if old_pattern in filename:
                new_name = filename.replace(old_pattern, new_pattern)

                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)

                # 检查目标文件是否已存在
                if os.path.exists(new_path):
                    print(f"警告: 目标文件 '{new_name}' 已存在，跳过")
                    continue

                if dry_run:
                    print(f'预览: {filename} -> {new_name}')
                    renamed_count += 1
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f'已重命名: {filename} -> {new_name}')
                        renamed_count += 1
                    except OSError as e:
                        print(f"错误: 重命名 '{filename}' 失败: {e}")

    except OSError as e:
        print(f'错误: 读取目录失败: {e}')
        return 0

    return renamed_count


def main():
    parser = argparse.ArgumentParser(description='批量重命名文件工具')
    parser.add_argument('--directory', help='目标目录路径')
    parser.add_argument('--old_pattern', help='要替换的旧字符串')
    parser.add_argument('--new_pattern', help='新字符串')
    parser.add_argument('--dry-run', action='store_true', help='只预览不实际执行重命名操作')

    args = parser.parse_args()

    print(f'目录: {args.directory}')
    print(f"替换 '{args.old_pattern}' -> '{args.new_pattern}'")
    if args.dry_run:
        print('模式: 预览模式 (不实际执行重命名)')

    count = rename_files(args.directory, args.old_pattern, args.new_pattern,
                         args.dry_run)

    if args.dry_run:
        print(f'\n预览完成，共 {count} 个文件将被重命名')
    else:
        print(f'\n重命名完成，共 {count} 个文件被重命名')


if __name__ == '__main__':
    main()
