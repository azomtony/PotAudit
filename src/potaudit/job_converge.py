from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict
from pathlib import Path

from .dpgen_fp_converge import ConvergenceResult, check_vasp_task


def find_job_dirs(
    jobs_dir: str | Path,
    job_glob: str,
    *,
    recursive: bool = False,
    single: bool = False,
) -> list[Path]:
    root = Path(jobs_dir)
    if single:
        return [root] if root.is_dir() else []

    if recursive:
        return sorted(
            p for p in root.rglob(job_glob)
            if p.is_dir() and fnmatch.fnmatch(p.name, job_glob)
        )
    return sorted(
        p for p in root.glob(job_glob)
        if p.is_dir() and fnmatch.fnmatch(p.name, job_glob)
    )


def _format_result(result: ConvergenceResult, job_dir: Path, root: Path) -> str:
    try:
        label_name = str(job_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        label_name = job_dir.name

    label = "not converged" if result.status == "not_converged" else result.status
    steps_done = result.max_scf_steps if result.max_scf_steps is not None else "NA"
    nelm = result.nelm if result.nelm is not None else "NA"
    detail = f"steps_completed={steps_done}/{nelm} ionic_steps={result.ionic_steps}"
    if result.status != "converged":
        detail += f" reason={result.reason}"
    return f"{label_name} {label} {detail}"


def run(args: argparse.Namespace) -> int:
    root = Path(args.jobs_dir)
    jobs = find_job_dirs(root, args.job_glob, recursive=args.recursive, single=args.single)
    checked: list[tuple[Path, ConvergenceResult]] = [
        (job, check_vasp_task(job, nelm_override=args.nelm, check_ionic=args.check_ionic))
        for job in jobs
    ]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "job": job.name,
                        "path": str(job),
                        **asdict(result),
                    }
                    for job, result in checked
                ],
                indent=2,
            )
        )
    else:
        for job, result in checked:
            if args.only_problems and result.status == "converged":
                continue
            print(_format_result(result, job, root))

        counts: dict[str, int] = {}
        for _, result in checked:
            counts[result.status] = counts.get(result.status, 0) + 1

        print()
        print("summary")
        print("checked:", len(checked))
        for key in ("converged", "not_converged", "incomplete", "unknown"):
            print(f"{key}:", counts.get(key, 0))

    if args.fail_on_problems and any(result.status != "converged" for _, result in checked):
        return 1
    return 0


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--jobs-dir", default=".", help="Directory containing job folders")
    p.add_argument("--job-glob", default="*", help='Job folder glob, e.g. "job_*"')
    p.add_argument("--recursive", action="store_true", help="Search for job folders recursively")
    p.add_argument("--single", action="store_true", help="Treat --jobs-dir itself as one job folder")
    p.add_argument("--nelm", type=int, default=None, help="Override NELM if OUTCAR/INCAR cannot be trusted")
    p.add_argument("--check-ionic", action="store_true", help="Also flag VASP relaxations that did not satisfy EDIFFG")
    p.add_argument("--only-problems", action="store_true", help="Only print incomplete, unknown, or unconverged jobs")
    p.add_argument("--json", action="store_true", help="Write machine-readable JSON")
    p.add_argument("--fail-on-problems", action="store_true", help="Exit nonzero when any job is not converged")
    return p


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(prog="job-converge"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
