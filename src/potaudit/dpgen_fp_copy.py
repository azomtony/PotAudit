from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_FILES = {
    "vasp": ["OUTCAR", "vasprun.xml", "fp.log"],
    "pwscf": ["output"],
    "abacus": ["output", "OUT.ABACUS"],
    "siesta": ["output"],
    "gaussian": ["output"],
    "cp2k": ["output"],
    "pwmat": ["REPORT", "OUT.MLMD", "output"],
}


def find_tasks(root: str | Path, recursive: bool = False) -> dict[str, Path]:
    root = Path(root)
    if recursive:
        out = {}
        for dirpath, dirnames, _ in os.walk(root):
            for name in dirnames:
                if name.startswith("task."):
                    out.setdefault(name, Path(dirpath) / name)
        return out
    return {p.name: p for p in root.glob("task.*") if p.is_dir()}


def copy_one(src: Path, dst: Path, apply: bool = False, overwrite: bool = False) -> str:
    if not src.exists():
        return "missing"
    if dst.exists():
        if not overwrite:
            return "exists"
        if apply:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    if apply:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return "copied"


def run(args: argparse.Namespace) -> int:
    local_tasks = find_tasks(args.local_fp)
    remote_tasks = find_tasks(args.remote_fp, recursive=args.search)
    files = args.files or DEFAULT_FILES[args.style]

    print("MODE:", "APPLY" if args.apply else "DRY RUN")
    print("local tasks:", len(local_tasks))
    print("remote tasks:", len(remote_tasks))
    print("files:", files)

    copied = existing = missing = unmatched = 0

    for task_name, local_task in sorted(local_tasks.items()):
        remote_task = remote_tasks.get(task_name)
        if remote_task is None:
            unmatched += 1
            print("[no remote task]", task_name)
            continue

        for fname in files:
            src = remote_task / fname
            dst = local_task / fname
            status = copy_one(src, dst, apply=args.apply, overwrite=args.overwrite)

            if status == "copied":
                copied += 1
                print("[copy]", task_name, fname)
            elif status == "exists":
                existing += 1
            else:
                missing += 1
                print("[missing]", task_name, fname)

    print()
    print("summary")
    print("copied:", copied)
    print("existing:", existing)
    print("missing files:", missing)
    print("unmatched tasks:", unmatched)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to copy.")

    return 0


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--local-fp", required=True, help="e.g. iter.000001/02.fp")
    ap.add_argument("--remote-fp", required=True, help="old scratch FP dir or parent")
    ap.add_argument("--style", default="vasp", choices=DEFAULT_FILES)
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--search", action="store_true", help="search remote recursively")
    ap.add_argument("--apply", action="store_true", help="actually copy")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(prog="dpgen-fp-copy"))


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
