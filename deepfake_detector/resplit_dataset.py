#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


FAKE = "Fake"
REAL = "Real"
CLASS_TO_LABEL = {FAKE: 1, REAL: 0}


@dataclass(frozen=True)
class VideoEntry:
    source_split: str
    class_name: str
    label: int
    source_dir: Path
    video_id: str
    unique_video_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resplit deepfake dataset by video_id with 70/15/15 stratified splits."
    )
    parser.add_argument("--src-train", required=True, type=Path, help="Path to source train split")
    parser.add_argument("--src-val", required=True, type=Path, help="Path to source val split")
    parser.add_argument(
        "--src-test",
        type=Path,
        default=None,
        help="Optional path to source test split (if provided, include it in the resplit pool)",
    )
    parser.add_argument("--dst", required=True, type=Path, help="Path to destination split root")
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for reproducible shuffling (default: 42)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan and report only; do not move files"
    )
    parser.add_argument(
        "--swap-to-processed-faces",
        action="store_true",
        help=(
            "After successful move, rename source dataset root to backup and rename "
            "destination root to source dataset root."
        ),
    )
    parser.add_argument(
        "--backup-suffix",
        default="_old_backup",
        help="Suffix used when backing up original source root during swap",
    )
    return parser.parse_args()


def list_video_dirs(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted([p for p in class_dir.iterdir() if p.is_dir()])


def collect_entries(source_splits: Dict[str, Path]) -> List[VideoEntry]:
    raw: List[Tuple[str, str, Path, str]] = []
    for split_name, split_root in source_splits.items():
        for class_name in (FAKE, REAL):
            class_dir = split_root / class_name
            videos = list_video_dirs(class_dir)
            for video_dir in videos:
                raw.append((split_name, class_name, video_dir, video_dir.name))

    # If same class/video_id appears in both train and val, add source prefix.
    by_key: Dict[Tuple[str, str], List[Tuple[str, str, Path, str]]] = {}
    for row in raw:
        split_name, class_name, _, video_id = row
        by_key.setdefault((class_name, video_id), []).append(row)

    entries: List[VideoEntry] = []
    used_dest_names: Dict[str, set] = {FAKE: set(), REAL: set()}
    for split_name, class_name, video_dir, video_id in raw:
        key_rows = by_key[(class_name, video_id)]
        if len(key_rows) > 1:
            base = f"{split_name}_{video_id}"
        else:
            base = video_id

        unique_name = base
        idx = 1
        while unique_name in used_dest_names[class_name]:
            idx += 1
            unique_name = f"{base}_{idx}"
        used_dest_names[class_name].add(unique_name)

        entries.append(
            VideoEntry(
                source_split=split_name,
                class_name=class_name,
                label=CLASS_TO_LABEL[class_name],
                source_dir=video_dir,
                video_id=video_id,
                unique_video_id=unique_name,
            )
        )
    return entries


def stratified_split(entries: Sequence[VideoEntry], seed: int) -> Dict[str, List[VideoEntry]]:
    rng = random.Random(seed)
    fake_entries = [e for e in entries if e.class_name == FAKE]
    real_entries = [e for e in entries if e.class_name == REAL]
    rng.shuffle(fake_entries)
    rng.shuffle(real_entries)

    def split_group(group: Sequence[VideoEntry]) -> Dict[str, List[VideoEntry]]:
        n = len(group)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val
        train = list(group[:n_train])
        val = list(group[n_train : n_train + n_val])
        test = list(group[n_train + n_val : n_train + n_val + n_test])
        return {"train": train, "val": val, "test": test}

    fake_split = split_group(fake_entries)
    real_split = split_group(real_entries)

    result: Dict[str, List[VideoEntry]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        merged = fake_split[split] + real_split[split]
        rng.shuffle(merged)
        result[split] = merged
    return result


def ensure_dst_dirs(dst: Path, dry_run: bool) -> None:
    required = [
        dst / "train" / FAKE,
        dst / "train" / REAL,
        dst / "val" / FAKE,
        dst / "val" / REAL,
        dst / "test" / FAKE,
        dst / "test" / REAL,
    ]
    if dry_run:
        return
    if dst.exists() and any(dst.iterdir()):
        raise RuntimeError(f"Destination already exists and is not empty: {dst}")
    for p in required:
        p.mkdir(parents=True, exist_ok=True)


def move_entries(split_map: Dict[str, List[VideoEntry]], dst: Path, dry_run: bool) -> None:
    total = sum(len(v) for v in split_map.values())
    iterator: Iterable[Tuple[str, VideoEntry]] = (
        (split_name, entry)
        for split_name in ("train", "val", "test")
        for entry in split_map[split_name]
    )
    for split_name, entry in tqdm(iterator, total=total, desc="Moving video folders", unit="video"):
        target_dir = dst / split_name / entry.class_name / entry.unique_video_id
        if dry_run:
            continue
        if target_dir.exists():
            raise RuntimeError(f"Target already exists: {target_dir}")
        shutil.move(str(entry.source_dir), str(target_dir))


def remove_if_empty(path: Path) -> None:
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def cleanup_old_split_dirs(source_splits: Dict[str, Path], dry_run: bool) -> None:
    if dry_run:
        return
    for split_root in source_splits.values():
        for class_name in (FAKE, REAL):
            remove_if_empty(split_root / class_name)
        remove_if_empty(split_root)


def swap_roots(source_splits: Dict[str, Path], dst: Path, suffix: str, dry_run: bool) -> None:
    parent_roots = {p.parent for p in source_splits.values()}
    if len(parent_roots) != 1:
        raise RuntimeError("Cannot swap roots: source splits are not under the same root")
    src_root = next(iter(parent_roots))
    backup_root = src_root.with_name(src_root.name + suffix)
    if dry_run:
        print(f"[DRY-RUN] Swap plan: {src_root} -> {backup_root}, then {dst} -> {src_root}")
        return
    if backup_root.exists():
        raise RuntimeError(f"Backup destination already exists: {backup_root}")
    shutil.move(str(src_root), str(backup_root))
    shutil.move(str(dst), str(src_root))


def summarize_split(split_name: str, entries: Sequence[VideoEntry]) -> str:
    fake = sum(1 for e in entries if e.class_name == FAKE)
    real = sum(1 for e in entries if e.class_name == REAL)
    total = fake + real
    fake_ratio = (fake / total * 100.0) if total else 0.0
    real_ratio = (real / total * 100.0) if total else 0.0
    return (
        f"{split_name:<5}: {fake:>6} Fake + {real:>6} Real = {total:>6} videos "
        f"({fake_ratio:>5.2f}% Fake / {real_ratio:>5.2f}% Real)"
    )


def count_overlap(a: Sequence[VideoEntry], b: Sequence[VideoEntry]) -> int:
    sa = {e.unique_video_id for e in a}
    sb = {e.unique_video_id for e in b}
    return len(sa.intersection(sb))


def print_report(split_map: Dict[str, List[VideoEntry]]) -> None:
    train = split_map["train"]
    val = split_map["val"]
    test = split_map["test"]
    total = len(train) + len(val) + len(test)
    print("\n=== KET QUA RESPLIT ===")
    print(summarize_split("Train", train))
    print(summarize_split("Val", val))
    print(summarize_split("Test", test))
    print(f"\nTong: {total} videos")

    overlaps = {
        "train_vs_val": count_overlap(train, val),
        "train_vs_test": count_overlap(train, test),
        "val_vs_test": count_overlap(val, test),
    }
    print(
        "Overlap check (unique video_id theo split nguon): "
        f"train-val={overlaps['train_vs_val']}, "
        f"train-test={overlaps['train_vs_test']}, "
        f"val-test={overlaps['val_vs_test']}"
    )

    train_ratio = sum(1 for e in train if e.class_name == FAKE) / len(train) if train else 0.0
    val_ratio = sum(1 for e in val if e.class_name == FAKE) / len(val) if val else 0.0
    test_ratio = sum(1 for e in test if e.class_name == FAKE) / len(test) if test else 0.0
    print(
        "Fake ratio theo split: "
        f"train={train_ratio*100:.2f}%, val={val_ratio*100:.2f}%, test={test_ratio*100:.2f}%"
    )
    print("Ratio nhat quan o 3 split: " + ("YES" if max(train_ratio, val_ratio, test_ratio) - min(train_ratio, val_ratio, test_ratio) < 0.01 else "CHECK"))


def validate_sources(source_splits: Dict[str, Path]) -> None:
    for _, root in source_splits.items():
        for class_name in (FAKE, REAL):
            class_dir = root / class_name
            if not class_dir.exists() or not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")


def main() -> None:
    args = parse_args()
    src_train = args.src_train.resolve()
    src_val = args.src_val.resolve()
    src_test = args.src_test.resolve() if args.src_test else None
    dst = args.dst.resolve()

    source_splits: Dict[str, Path] = {"train": src_train, "val": src_val}
    if src_test is not None:
        source_splits["test"] = src_test

    validate_sources(source_splits)
    entries = collect_entries(source_splits)
    if not entries:
        raise RuntimeError("No video_id directories found in source splits.")

    split_map = stratified_split(entries, args.seed)
    print_report(split_map)
    ensure_dst_dirs(dst, args.dry_run)
    move_entries(split_map, dst, args.dry_run)
    cleanup_old_split_dirs(source_splits, args.dry_run)

    if args.swap_to_processed_faces:
        swap_roots(source_splits, dst, args.backup_suffix, args.dry_run)
    elif not args.dry_run:
        print("\nChua swap root. Du lieu moi dang o:", dst)


if __name__ == "__main__":
    main()
