from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_EXCLUDE = os.environ.get("POTAUDIT_DPGEN_FP_EXCLUDE", "chpc129,chpc098")
DEFAULT_TIME = "12:00:00"
DEFAULT_COMMAND = "mpirun -np ${SLURM_NTASKS} vasp_std 1>>fp.log 2>>fp.log"

DEFAULT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir={task_dir}
{exclude_directive}
#SBATCH --exclusive
#SBATCH --mem=0

set -euo pipefail

module purge
module load vasp/gnu14.6.5.0
export OMP_NUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Expanded nodes:"
scontrol show hostnames "${SLURM_JOB_NODELIST}"
echo "Running on host: $(hostname)"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "Start time: $(date -Is)"

cd "{task_dir}"
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


def sanitize_job_name(task_name: str) -> str:
    return "fp_" + task_name.replace(".", "_")


def task_complete(task_dir: str | Path, style: str) -> bool:
    task_dir = Path(task_dir)
    if style == "vasp":
        outcar = task_dir / "OUTCAR"
        if not outcar.is_file():
            return False
        try:
            text = outcar.read_text(errors="ignore")
        except OSError:
            return False
        return text.count("Elapse") == 1
    if style == "abacus":
        log = task_dir / "OUT.ABACUS" / "running_scf.log"
        return log.is_file() and "!FINAL_ETOT_IS" in log.read_text(errors="ignore")
    if style in {"pwscf", "qe"}:
        output = task_dir / "output"
        return output.is_file() and "JOB DONE" in output.read_text(errors="ignore")
    if style == "cp2k":
        output = task_dir / "output"
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
    out = run_cmd(
        [
            "sacct",
            "-n",
            "-X",
            "--name",
            job_name,
            "--format=State",
        ]
    )
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


def parse_system_map(items: list[str] | None) -> dict[str, str]:
    result = {}
    for item in items or []:
        key, value = item.split("=", 1)
        result[key] = value
    return result


def system_id(task_name: str) -> str:
    # task.012.000024 -> 012
    return task_name.split(".")[1]


def value_for_task(default: str, mapping: dict[str, str], task_name: str) -> str:
    sid = system_id(task_name)
    return mapping.get(task_name, mapping.get(sid, default))


def load_template(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEMPLATE
    return Path(path).read_text()


def write_submit_script(
    task_dir: Path,
    template: str,
    args: argparse.Namespace,
    task_name: str,
    maps: dict[str, dict[str, str]],
) -> tuple[Path, str]:
    job_name = sanitize_job_name(task_name)

    partition = value_for_task(args.partition, maps["partition"], task_name)
    slurm_time = value_for_task(args.time, maps["time"], task_name)
    exclude = value_for_task(args.exclude, maps["exclude"], task_name)
    nodes = value_for_task(str(args.nodes), maps["nodes"], task_name)
    ntasks_per_node = value_for_task(str(args.ntasks_per_node), maps["ntasks_per_node"], task_name)
    exclude_directive = f"#SBATCH --exclude={exclude}" if exclude else ""

    text = template.format(
        job_name=job_name,
        partition=partition,
        time=slurm_time,
        exclude=exclude,
        exclude_directive=exclude_directive,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        task_dir=str(task_dir.resolve()),
        command=args.command,
        task_name=task_name,
    )

    script = task_dir / args.submit_file
    script.write_text(text)
    script.chmod(0o755)
    return script, job_name


def submit_job(script: Path, dry_run: bool) -> str:
    if dry_run:
        return "DRYRUN"
    out = subprocess.check_output(["sbatch", str(script)], text=True)
    return out.strip()


def eligible_tasks(fp_dir: str, task_glob: str) -> list[Path]:
    tasks = []
    for path in sorted(Path(fp_dir).glob("task.*")):
        if path.is_dir() and fnmatch.fnmatch(path.name, task_glob):
            tasks.append(path)
    return tasks


def one_pass(args: argparse.Namespace, template: str, maps: dict[str, dict[str, str]]) -> None:
    tasks = eligible_tasks(args.fp_dir, args.task_glob)

    submitted = 0
    skipped_complete = 0
    skipped_active = 0
    skipped_done_scheduler = 0
    checked = 0

    for task_dir in tasks:
        task_name = task_dir.name
        checked += 1

        script, job_name = write_submit_script(task_dir, template, args, task_name, maps)

        if task_complete(task_dir, args.style):
            skipped_complete += 1
            print(f"[complete-local] {task_name}")
            continue

        state = scheduler_state(job_name)

        if state in ACTIVE_STATES:
            skipped_active += 1
            print(f"[{state.lower()}] {task_name} {job_name}")
            continue

        if state in DONE_STATES:
            skipped_done_scheduler += 1
            print(f"[completed-scheduler] {task_name} {job_name}")
            continue

        if args.max_submit is not None and submitted >= args.max_submit:
            print(f"[submit-limit] {task_name}")
            continue

        result = submit_job(script, args.dry_run)
        submitted += 1
        print(f"[submit] {task_name} {job_name}: {result}")

    print(
        json.dumps(
            {
                "checked": checked,
                "submitted": submitted,
                "complete_local": skipped_complete,
                "active_or_pending": skipped_active,
                "completed_scheduler": skipped_done_scheduler,
            },
            indent=2,
        )
    )


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--fp-dir", default="iter.000001/02.fp")
    p.add_argument("--style", default="vasp", choices=["vasp", "abacus", "pwscf", "qe", "cp2k"])
    p.add_argument("--template", default=None)
    p.add_argument("--submit-file", default="submit.slurm")

    p.add_argument("--partition", required=True)
    p.add_argument("--time", default=DEFAULT_TIME, help=f"Slurm walltime (default: {DEFAULT_TIME})")
    p.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help="Comma-separated Slurm nodes to exclude. Default can be set with POTAUDIT_DPGEN_FP_EXCLUDE.",
    )
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--ntasks-per-node", type=int, default=1)
    p.add_argument("--command", default=DEFAULT_COMMAND, help=f'Default: "{DEFAULT_COMMAND}"')

    p.add_argument("--task-glob", default="task.*", help='Example: "task.000.*"')
    p.add_argument("--max-submit", type=int, default=None)

    p.add_argument("--partition-map", action="append", default=[], help="Example: 000=debug or task.000.000001=normal")
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
        "partition": parse_system_map(args.partition_map),
        "time": parse_system_map(args.time_map),
        "exclude": parse_system_map(args.exclude_map),
        "nodes": parse_system_map(args.nodes_map),
        "ntasks_per_node": parse_system_map(args.ntasks_per_node_map),
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
    return add_arguments(argparse.ArgumentParser(prog="dpgen-fp"))


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
