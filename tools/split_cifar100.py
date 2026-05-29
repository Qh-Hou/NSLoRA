import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create CIFAR-100 train/val folders from split files.")
    parser.add_argument("--source-root", required=True, type=Path, help="Root directory containing raw CIFAR-100 files.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory where train/val folders are created.")
    parser.add_argument("--split-dir", default=Path(__file__).resolve().parent, type=Path, help="Directory with cifar100_train.txt and cifar100_val.txt.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy_file: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_file:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def split_func(mode: str, source_root: Path, output_root: Path, split_dir: Path, copy_file: bool):
    list_path = split_dir / f"cifar100_{mode}.txt"
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    for line in list_path.read_text().splitlines():
        rel_path = line.strip()
        if not rel_path:
            continue
        src_path = (source_root / rel_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        dst_path = mode_dir / Path(rel_path).parent / src_path.name
        link_or_copy(src_path, dst_path, copy_file)


def main():
    args = parse_args()
    for mode in ("train", "val"):
        split_func(mode, args.source_root, args.output_root, args.split_dir, args.copy)


if __name__ == "__main__":
    main()
