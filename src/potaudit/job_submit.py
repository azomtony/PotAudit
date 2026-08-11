from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import time
from pathlib import Path


DEFAULT_EXCLUDE = os.environ.get("POTAUDIT_JOB_EXCLUDE", "chpc129,chpc098")
DEFAULT_TIME = "12:00:00"
DEFAULT_COMMAND = "mpirun -np ${SLURM_NTASKS} vasp_std 1>>vasp.log 2>>vasp.log"
DEFAULT_VASP_REQUIRED = ("INCAR", "POSCAR", "POTCAR", "KPOINTS")

DEFAULT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir={job_dir}
{exclude_directive}

set -euo pipefail

module purge
module load vasp/gnu14.6.5.0
export OMP_NUM_THREADS=1

cd "{job_dir}"
{command}
"""

ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
}

DONE_STATES = {
    "COMPLETED",
}


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def sanitize_job_name(job_path: Path, root: Path) -> str:
    try:
        label = str(job_path.resolve().relative_to(root.resolve()))
    except ValueError:
        label = job_path.name
    label = label.strip(".") or job_path.name
    label = re.sub(r"[^A-Za-z0-9_]+", "_", label)
    return ("pa_" + label).strip("_")[:128]


def job_complete(job_dir: str | Path, style: str) -> bool:
    job_dir = Path(job_dir)
    if style == "vasp":
        outcar = job_dir / "OUTCAR"
        if not outcar.is_file():
            return False
        try:
            text = outcar.read_text(errors="ignore")
        except OSError:
            return False
        return "General timing and accounting" in text
    if style == "abacus":
        log = job_dir / "OUT.ABACUS" / "running_scf.log"
        return log.is_file() and "!FINAL_ETOT_IS" in log.read_text(errors="ignore")
    if style in {"pwscf", "qe"}:
        output = job_dir / "output"
        return output.is_file() and "JOB DONE" in output.read_text(errors="ignore")
    if style == "cp2k":
        output = job_dir / "output"
        return output.is_file() and "SCF run converged" in output.read_text(errors="ignore")
    return False


def squeue_state(job_name: str) -> str | None:
    out = run_cmd(["squeue", "-h", "-n", job_name, "-o", "%T"])
    states = [x.strip() for x in out.splitlines() if x.strip()]
    if not states:
        return None
    if any(s in ACTIVE_STATES for s in states):
        return states[0]
    return states[0]


def sacct_state(job_name: str) -> str | None:
    out = run_cmd(["sacct", "-n", "-X", "--name", job_name, "--format=State"])
    states = [x.strip().split()[0] for x in out.splitlines() if x.strip()]
    if not states:
        return None
    if any(s in DONE_STATES for s in states):
        return "COMPLETED"
    return states[-1]


def scheduler_state(job_name: str) -> str | None:
    sq = squeue_state(job_name)
    if sq:
        return sq
    return sacct_state(job_name)


def sbatch_job_name(script: Path) -> str | None:
    if not script.is_file():
        return None
    try:
        text = script.read_text(errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        match = re.match(r"\s*#SBATCH\s+--job-name(?:=|\s+)(\S+)", line)
        if match:
            return match.group(1)
    return None


def scheduler_state_for_names(job_names: list[str]) -> tuple[str | None, str | None]:
    for job_name in job_names:
        state = scheduler_state(job_name)
        if state in ACTIVE_STATES:
            return state, job_name
    for job_name in job_names:
        state = scheduler_state(job_name)
        if state:
            return state, job_name
    return None, None


def parse_name_map(items: list[str] | None) -> dict[str, str]:
    result = {}
    for item in items or []:
        key, value = item.split("=", 1)
        result[key] = value
    return result


def value_for_job(default: str, mapping: dict[str, str], job_dir: Path, root: Path) -> str:
    try:
        rel = str(job_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = job_dir.name

    candidates = (rel, job_dir.name)
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    for pattern, value in mapping.items():
        if any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates):
            return value
    return default


def load_template(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEMPLATE
    return Path(path).read_text()


def required_files_for_args(args: argparse.Namespace) -> list[str]:
    required = list(args.require or [])
    if args.style == "vasp" and not args.no_default_require:
        required = list(DEFAULT_VASP_REQUIRED) + required
    return required


def missing_required_files(job_dir: Path, required: list[str]) -> list[str]:
    return [name for name in required if not (job_dir / name).exists()]


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


def write_submit_script(
    job_dir: Path,
    root: Path,
    template: str,
    args: argparse.Namespace,
    maps: dict[str, dict[str, str]],
) -> tuple[Path, str]:
    job_name = sanitize_job_name(job_dir, root)

    partition = value_for_job(args.partition, maps["partition"], job_dir, root)
    slurm_time = value_for_job(args.time, maps["time"], job_dir, root)
    exclude = value_for_job(args.exclude, maps["exclude"], job_dir, root)
    nodes = value_for_job(str(args.nodes), maps["nodes"], job_dir, root)
    ntasks_per_node = value_for_job(str(args.ntasks_per_node), maps["ntasks_per_node"], job_dir, root)
    exclude_directive = f"#SBATCH --exclude={exclude}" if exclude else ""

    text = template.format(
        job_name=job_name,
        partition=partition,
        time=slurm_time,
        exclude=exclude,
        exclude_directive=exclude_directive,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        job_dir=str(job_dir.resolve()),
        command=args.command,
        job_name_raw=job_dir.name,
    )

    script = job_dir / args.submit_file
    script.write_text(text)
    script.chmod(0o755)
    return script, job_name


def submit_job(script: Path, dry_run: bool) -> str:
    if dry_run:
        return "DRYRUN"
    out = subprocess.check_output(["sbatch", str(script)], text=True)
    return out.strip()


def one_pass(args: argparse.Namespace, template: str, maps: dict[str, dict[str, str]]) -> None:
    root = Path(args.jobs_dir)
    jobs = find_job_dirs(root, args.job_glob, recursive=args.recursive, single=args.single)
    required = required_files_for_args(args)

    submitted = 0
    skipped_complete = 0
    skipped_active = 0
    skipped_done_scheduler = 0
    skipped_missing_files = 0
    checked = 0

    for job_dir in jobs:
        checked += 1
        missing = missing_required_files(job_dir, required)
        if missing:
            skipped_missing_files += 1
            print(f"[missing-files] {job_dir.name}: {','.join(missing)}")
            continue

        if job_complete(job_dir, args.style):
            skipped_complete += 1
            print(f"[complete-local] {job_dir.name}")
            continue

        generated_job_name = sanitize_job_name(job_dir, root)
        existing_job_name = sbatch_job_name(job_dir / args.submit_file)
        scheduler_names = [generated_job_name]
        if existing_job_name and existing_job_name not in scheduler_names:
            scheduler_names.append(existing_job_name)
        state, matched_job_name = scheduler_state_for_names(scheduler_names)

        if state in ACTIVE_STATES:
            skipped_active += 1
            print(f"[{state.lower()}] {job_dir.name} {matched_job_name}")
            continue

        if state in DONE_STATES:
            skipped_done_scheduler += 1
            print(f"[completed-scheduler] {job_dir.name} {matched_job_name}")
            continue

        if args.max_submit is not None and submitted >= args.max_submit:
            print(f"[submit-limit] {job_dir.name}")
            continue

        script, job_name = write_submit_script(job_dir, root, template, args, maps)
        result = submit_job(script, args.dry_run)
        submitted += 1
        print(f"[submit] {job_dir.name} {job_name}: {result}")

    print(
        json.dumps(
            {
                "checked": checked,
                "submitted": submitted,
                "complete_local": skipped_complete,
                "active_or_pending": skipped_active,
                "completed_scheduler": skipped_done_scheduler,
                "missing_required_files": skipped_missing_files,
            },
            indent=2,
        )
    )


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--jobs-dir", default=".", help="Directory containing job folders")
    p.add_argument("--job-glob", default="*", help='Job folder glob, e.g. "job_*"')
    p.add_argument("--recursive", action="store_true", help="Search for job folders recursively")
    p.add_argument("--single", action="store_true", help="Treat --jobs-dir itself as one job folder")
    p.add_argument("--style", default="vasp", choices=["vasp", "abacus", "pwscf", "qe", "cp2k"])
    p.add_argument("--template", default=None)
    p.add_argument("--submit-file", default="submit.slurm")

    p.add_argument("--partition", required=True)
    p.add_argument("--time", default=DEFAULT_TIME, help=f"Slurm walltime (default: {DEFAULT_TIME})")
    p.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help="Comma-separated Slurm nodes to exclude. Default can be set with POTAUDIT_JOB_EXCLUDE.",
    )
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--ntasks-per-node", type=int, default=1)
    p.add_argument("--command", default=DEFAULT_COMMAND, help=f'Default: "{DEFAULT_COMMAND}"')

    p.add_argument("--require", action="append", default=[], help="Required file relative to each job folder")
    p.add_argument("--no-default-require", action="store_true", help="Do not require VASP input files by default")
    p.add_argument("--max-submit", type=int, default=None)

    p.add_argument("--partition-map", action="append", default=[], help="Example: job_*=debug or relax/a=normal")
    p.add_argument("--time-map", action="append", default=[])
    p.add_argument("--exclude-map", action="append", default=[])
    p.add_argument("--nodes-map", action="append", default=[])
    p.add_argument("--ntasks-per-node-map", action="append", default=[])

    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args: argparse.Namespace) -> int:
    maps = {
        "partition": parse_name_map(args.partition_map),
        "time": parse_name_map(args.time_map),
        "exclude": parse_name_map(args.exclude_map),
        "nodes": parse_name_map(args.nodes_map),
        "ntasks_per_node": parse_name_map(args.ntasks_per_node_map),
    }

    template = load_template(args.template)

    while True:
        one_pass(args, template, maps)
        if not args.watch:
            break
        print(f"sleeping {args.interval}s...")
        time.sleep(args.interval)

    return 0


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(prog="job-submit"))


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
