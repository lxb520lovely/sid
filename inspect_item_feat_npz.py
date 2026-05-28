#!/usr/bin/env python3
"""Inspect item_feat.npz without loading the whole file by default.

Examples:
  .venv/bin/python inspect_item_feat_npz.py
  .venv/bin/python inspect_item_feat_npz.py --head 20
  .venv/bin/python inspect_item_feat_npz.py --stats --force-stats
"""

from __future__ import annotations

import argparse
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib import format as npy_format


BYTE_UNITS = ("B", "KB", "MB", "GB", "TB")


@dataclass(frozen=True)
class NpyMember:
    member_name: str
    key: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    file_size: int
    compress_size: int
    compress_type: int

    @property
    def nbytes(self) -> int:
        if not self.shape:
            return self.dtype.itemsize
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def order(self) -> str:
        return "F" if self.fortran_order else "C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print keys, shapes, dtypes, and sample rows from item_feat.npz."
    )
    parser.add_argument("--path", type=Path, default=Path("item_feat.npz"))
    parser.add_argument("--head", type=int, default=5, help="Number of leading rows to print.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute min/max/mean/std for arrays. Large arrays are skipped unless --force-stats is set.",
    )
    parser.add_argument(
        "--stats-max-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Max array size to load for --stats without --force-stats.",
    )
    parser.add_argument(
        "--force-stats",
        action="store_true",
        help="Allow --stats to load large arrays into memory.",
    )
    return parser.parse_args()


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in BYTE_UNITS:
        if value < 1024.0 or unit == BYTE_UNITS[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{n} B"


def read_npy_header(handle) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = npy_format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
    else:
        shape, fortran_order, dtype = npy_format._read_array_header(handle, version)
    return tuple(int(v) for v in shape), bool(fortran_order), np.dtype(dtype)


def npz_key(member_name: str) -> str:
    return member_name[:-4] if member_name.endswith(".npy") else member_name


def read_members(path: Path) -> list[NpyMember]:
    members: list[NpyMember] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info) as handle:
                shape, fortran_order, dtype = read_npy_header(handle)
            members.append(
                NpyMember(
                    member_name=info.filename,
                    key=npz_key(info.filename),
                    shape=shape,
                    dtype=dtype,
                    fortran_order=fortran_order,
                    file_size=info.file_size,
                    compress_size=info.compress_size,
                    compress_type=info.compress_type,
                )
            )
    return members


def reopen_member_at_data(path: Path, meta: NpyMember):
    zf = zipfile.ZipFile(path)
    handle = zf.open(meta.member_name)
    read_npy_header(handle)
    return zf, handle, handle.tell()


def read_head(path: Path, meta: NpyMember, head: int) -> np.ndarray:
    if head <= 0:
        return np.empty((0,), dtype=meta.dtype)

    zf, handle, data_offset = reopen_member_at_data(path, meta)
    try:
        if not meta.shape:
            raw = handle.read(meta.dtype.itemsize)
            return np.frombuffer(raw, dtype=meta.dtype).reshape(())

        rows = min(head, meta.shape[0])
        if len(meta.shape) == 1:
            raw = handle.read(rows * meta.dtype.itemsize)
            return np.frombuffer(raw, dtype=meta.dtype).copy()

        if not meta.fortran_order:
            row_width = math.prod(meta.shape[1:])
            raw = handle.read(rows * row_width * meta.dtype.itemsize)
            return np.frombuffer(raw, dtype=meta.dtype).copy().reshape((rows, *meta.shape[1:]))

        if len(meta.shape) == 2:
            total_rows, total_cols = meta.shape
            out = np.empty((rows, total_cols), dtype=meta.dtype)
            column_bytes = total_rows * meta.dtype.itemsize
            for col in range(total_cols):
                handle.seek(data_offset + col * column_bytes)
                raw = handle.read(rows * meta.dtype.itemsize)
                out[:, col] = np.frombuffer(raw, dtype=meta.dtype)
            return out

        raise ValueError(
            f"{meta.key} is Fortran-order with shape {meta.shape}; "
            "this script only samples Fortran-order arrays up to 2-D."
        )
    finally:
        handle.close()
        zf.close()


def format_scalar_or_list(value: np.ndarray) -> str:
    if value.shape == ():
        return repr(value.item())
    return repr(value.tolist())


def print_head(path: Path, members: Iterable[NpyMember], head: int) -> None:
    print(f"\nHead rows: {head}")
    for meta in members:
        try:
            sample = read_head(path, meta, head)
        except Exception as exc:
            print(f"\n[{meta.key}] unable to read sample: {exc}")
            continue

        print(f"\n[{meta.key}]")
        if sample.shape == ():
            print(f"  value: {format_scalar_or_list(sample)}")
        elif sample.ndim == 1:
            print(f"  first {sample.shape[0]}: {sample.tolist()}")
        else:
            for row_idx, row in enumerate(sample):
                print(f"  row {row_idx}: {row.tolist()}")


def print_stats(path: Path, members: Iterable[NpyMember], max_bytes: int, force: bool) -> None:
    print("\nStats")
    with np.load(path, allow_pickle=False) as data:
        for meta in members:
            if meta.nbytes > max_bytes and not force:
                print(
                    f"  {meta.key}: skipped ({human_bytes(meta.nbytes)} > "
                    f"{human_bytes(max_bytes)}; use --force-stats to load)"
                )
                continue
            arr = np.asarray(data[meta.key])
            if arr.size == 0:
                print(f"  {meta.key}: empty")
                continue
            numeric = arr.astype(np.float64, copy=False)
            unique = np.unique(arr).size if arr.size <= 5_000_000 else "skipped"
            print(
                f"  {meta.key}: min={numeric.min():.6g}, max={numeric.max():.6g}, "
                f"mean={numeric.mean():.6g}, std={numeric.std():.6g}, unique={unique}"
            )


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)

    members = read_members(args.path)
    print(f"File: {args.path.resolve()}")
    print(f"Size: {human_bytes(args.path.stat().st_size)}")
    print("\nMembers")
    for meta in members:
        compression = "stored" if meta.compress_type == zipfile.ZIP_STORED else "compressed"
        print(
            f"  {meta.key:<16} shape={meta.shape!s:<18} dtype={str(meta.dtype):<8} "
            f"order={meta.order} nbytes={human_bytes(meta.nbytes):>10} "
            f"zip={human_bytes(meta.compress_size):>10} {compression}"
        )

    print_head(args.path, members, args.head)

    if args.stats:
        print_stats(args.path, members, args.stats_max_bytes, args.force_stats)


if __name__ == "__main__":
    main()
