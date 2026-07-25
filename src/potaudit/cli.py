from __future__ import annotations
#------------Load Required Modules----------------
import argparse
from pathlib import Path
#------------End Required Modules----------------

#------------Load Internal Modules----------------
from .select import select_frames
from .prep_vasp import prep_vasp_from_indices_file

#------------End Internal Modules----------------

#------------Add Subcommand----------------

# select coomand: deterministic frame selection from extxyz
def _run_select(args: argparse.Namespace) -> int:
    """Run: potaudit select ..."""
    sel=select_frames(
        args.input,
        max_frames=args.max_frames,
        stride=args.stride,
        start=args.start,
        stop=args.stop,
    )
    out=Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(i) for i in sel.indices))
    print(f"[PotAudit] Selected {len(sel.indices)} frames --> {out}")
    return 0

def _add_select_subcommand(subparser: argparse._SubParsersAction) -> None:
    p=subparser.add_parser(
        "select",
        help="Select frames from a multi-frame extxyz (deterministic sampling).",        
    )
    p.add_argument("--input", required=True, help="Input extxyz with multiple frames")
    p.add_argument("--max-frames", type=int, default=50, help="Max frames to select (auto mode cap)")
    p.add_argument("--stride", type=int, default=None, help="Select every Nth frame (overrides auto)")
    p.add_argument("--start", type=int, default=0, help="Start frame (inclusive). Can be negative.")
    p.add_argument("--stop", type=int, default=None, help="Stop frame (exclusive). Can be negative.")
    p.add_argument("--out", default="selected_frames.txt", help="Write selected indices to this file")
    p.set_defaults(func=_run_select)

#PREPPING VASP INPUTS FROM SELECTED FRAMES

def _run_prep_vasp(args: argparse.Namespace) -> int:
    """Run: potaudit prep-vasp ..."""
    report = prep_vasp_from_indices_file(
        extxyz_path=args.input,
        indices_file=args.indices,
        out_root=args.out_root,
        templates_dir=args.templates,
        potcar_root=args.potcar_root,
        potcar_suffix=args.potcar_suffix,
        force=args.force,
    )
    print(f"[PotAudit] prep-vasp prepared={len(report.prepared)} skipped={len(report.skipped)}")
    return 0
def _add_prep_vasp_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prep-vasp",
        help="Prepare per-frame VASP single-point folders (POSCAR/INCAR/KPOINTS/POTCAR) from extxyz + indices.",
    )
    p.add_argument("--input", required=True, help="Input extxyz with multiple frames")
    p.add_argument("--indices", required=True, help="Text file with frame indices (one per line)")
    p.add_argument("--out-root", required=True, help="Output root dir, e.g. runs/case001/vasp")
    p.add_argument(
        "--templates",
        default=None,
        help="Templates directory containing INCAR and KPOINTS (default: <repo>/templates/vasp).",
    )

    p.add_argument(
        "--potcar-root",
        default=None,
        help="Pseudopotential root (e.g. .../potpaw_PBE.64). If omitted, POTCAR is not created.",
    )
    p.add_argument(
        "--potcar-suffix",
        default="",
        help="Suffix appended to element folder under potcar-root (e.g. _pv, _sv, _d, _GW, _sv_GW). Default: '' (standard).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing prepared folders")

    p.set_defaults(func=_run_prep_vasp)


def _run_prep_vasp_sp_dir(args: argparse.Namespace) -> int:
    from .prep_vasp import prep_vasp_singlepoints_from_extxyz_dir

    report = prep_vasp_singlepoints_from_extxyz_dir(
        input_dir=args.input_dir,
        out_root=args.out_root,
        templates_dir=args.templates_dir,
        potcar_root=args.potcar_root,
        potcar_suffix=args.potcar_suffix,
        pattern=args.pattern,
        recursive=args.recursive,
        index=args.index,
        job_prefix=args.job_prefix,
        incar_set=args.incar_set,
        force=args.force,
    )
    print(
        f"[PotAudit] prep-vasp-sp-dir prepared={len(report.prepared)} "
        f"skipped={len(report.skipped)}"
    )
    return 0


def _add_prep_vasp_sp_dir_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prep-vasp-sp-dir",
        help="Prepare VASP single-point folders from *_opt.extxyz files in a directory.",
    )
    p.add_argument("--input-dir", required=True, help="Directory containing extxyz files")
    p.add_argument("--out-root", required=True, help="Output root directory for VASP single-point folders")
    p.add_argument("--pattern", default="*_opt.extxyz", help="File glob to match inside input-dir (default: *_opt.extxyz)")
    p.add_argument("--recursive", action="store_true", help="Search input-dir recursively")
    p.add_argument(
        "--index",
        default=":",
        help="ASE frame index to read from each extxyz (default ':' = all frames; examples: 0, -1, 0:10:2)",
    )
    p.add_argument(
        "--templates-dir",
        "--templates",
        dest="templates_dir",
        default=None,
        help="Templates directory containing INCAR, KPOINTS, and sub_vasp.sh (default: repo templates/vasp)",
    )
    p.add_argument("--potcar-root", default=None, help="Pseudopotential root (e.g. potpaw_PBE.64)")
    p.add_argument("--potcar-suffix", default="", help="Suffix like _pv, _GW, _sv_GW (applied to all elements)")
    p.add_argument("--job-prefix", default="sp", help="Short job folder prefix (default: sp)")
    p.add_argument(
        "--incar-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra INCAR override. Can be used multiple times.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite even if state.json exists")
    p.set_defaults(func=_run_prep_vasp_sp_dir)


def _run_prep_vasp_opt_dir(args: argparse.Namespace) -> int:
    from .prep_vasp import prep_vasp_optimizations_from_extxyz_dir

    report = prep_vasp_optimizations_from_extxyz_dir(
        input_dir=args.input_dir,
        out_root=args.out_root,
        templates_dir=args.templates_dir,
        potcar_root=args.potcar_root,
        potcar_suffix=args.potcar_suffix,
        pattern=args.pattern,
        recursive=args.recursive,
        index=args.index,
        nsw=args.nsw,
        ibrion=args.ibrion,
        isif=args.isif,
        ediffg=args.ediffg,
        incar_set=args.incar_set,
        fix_bottom_layers=0 if args.no_fix_bottom else args.fix_bottom_layers,
        bottom_layer_tol=args.bottom_layer_tol,
        bottom_layer_axis=args.bottom_layer_axis,
        force=args.force,
    )
    print(
        f"[PotAudit] prep-vasp-opt-dir prepared={len(report.prepared)} "
        f"skipped={len(report.skipped)}"
    )
    return 0


def _add_prep_vasp_opt_dir_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prep-vasp-opt-dir",
        help="Prepare VASP geometry-optimization folders from extxyz files in a directory.",
    )
    p.add_argument("--input-dir", required=True, help="Directory containing extxyz files")
    p.add_argument("--out-root", required=True, help="Output root directory for VASP optimization folders")
    p.add_argument("--pattern", default="*.extxyz", help="File glob to match inside input-dir (default: *.extxyz)")
    p.add_argument("--recursive", action="store_true", help="Search input-dir recursively")
    p.add_argument(
        "--index",
        default=":",
        help="ASE frame index to read from each extxyz (default ':' = all frames; examples: 0, -1, 0:10:2)",
    )
    p.add_argument(
        "--templates-dir",
        "--templates",
        dest="templates_dir",
        default=None,
        help="Templates directory containing INCAR, KPOINTS, and sub_vasp.sh (default: repo templates/vasp_relax)",
    )
    p.add_argument("--potcar-root", default=None, help="Pseudopotential root (e.g. potpaw_PBE.64)")
    p.add_argument("--potcar-suffix", default="", help="Suffix like _pv, _GW, _sv_GW (applied to all elements)")
    p.add_argument("--nsw", type=int, default=None, help="Override template NSW value")
    p.add_argument("--ibrion", type=int, default=None, help="Override template IBRION value")
    p.add_argument("--isif", type=int, default=None, help="Override template ISIF value")
    p.add_argument("--ediffg", type=float, default=None, help="Override template EDIFFG value")
    p.add_argument(
        "--fix-bottom-layers",
        type=int,
        default=1,
        help="Automatically fix this many bottom slab layers in POSCAR selective dynamics (default: 1)",
    )
    p.add_argument(
        "--bottom-layer-tol",
        type=float,
        default=0.5,
        help="Layer clustering tolerance in Angstrom for bottom-layer detection (default: 0.5)",
    )
    p.add_argument(
        "--bottom-layer-axis",
        choices=("frac-c", "cart-z"),
        default="frac-c",
        help="Coordinate used to detect bottom layers. frac-c handles non-orthogonal slab cells (default: frac-c)",
    )
    p.add_argument("--no-fix-bottom", action="store_true", help="Do not fix bottom slab atoms")
    p.add_argument(
        "--incar-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra INCAR override. Can be used multiple times.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite even if state.json exists")
    p.set_defaults(func=_run_prep_vasp_opt_dir)


# Seleect and Prep VASP can be chained in a shell script like:

def _run_select_prep_vasp(args: argparse.Namespace) -> int:
    from .prep_vasp import select_and_prep_vasp

    report = select_and_prep_vasp(
        extxyz_path=args.input,
        out_root=args.out_root,
        templates_dir=args.templates_dir,
        potcar_root=args.potcar_root,
        potcar_suffix=args.potcar_suffix,
        max_frames=args.max_frames,
        stride=args.stride,
        start=args.start,
        stop=args.stop,
        force=args.force,
    )
    print(f"[PotAudit] prepared={len(report.prepared)} skipped={len(report.skipped)}")
    return 0


def _add_select_prep_vasp_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "select-prep-vasp",
        help="Select frames and prepare VASP job folders in one command.",
    )
    p.add_argument("--input", required=True, help="Input multi-frame extxyz")
    p.add_argument("--out-root", required=True, help="Output root directory for VASP folders")

    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)

    p.add_argument("--templates-dir", default=None, help="Override templates dir (default: repo templates/vasp)")
    p.add_argument("--potcar-root", default=None, help="Pseudopotential root (e.g. potpaw_PBE.64)")
    p.add_argument("--potcar-suffix", default="", help="Suffix like _pv, _GW, _sv_GW (applied to all elements)")
    p.add_argument("--force", action="store_true", help="Overwrite even if state.json exists")

    p.set_defaults(func=_run_select_prep_vasp)

#Submit jobs and check status subcommands are in submit.py and status.py respectively, to avoid circular imports in this cli module.

def _run_submit(args: argparse.Namespace) -> int:
    from .submit import submit_jobs

    rep = submit_jobs(
        out_root=args.out_root,
        max_inflight=args.max_inflight,
        watch=args.watch,
        poll_sec=args.poll_sec,
        limit=args.limit,
        verbose=(not args.quiet),
        crosscheck_squeue=(not args.no_crosscheck_squeue),
        refresh_status_in_watch=(not args.no_status_refresh),
    )

    # Print a useful summary
    if not args.quiet:
        if rep.submitted:
            # show all submitted names (or cap it, your choice)
            print(f"[PotAudit] submitted {len(rep.submitted)}: " + ", ".join(rep.submitted))
        else:
            print("[PotAudit] submitted 0")

        print(
            f"[PotAudit] inflight={rep.inflight} "
            f"remaining_ready={getattr(rep, 'remaining_ready', 'NA')} "
            f"skipped={len(rep.skipped)}"
        )
    else:
        print(f"[PotAudit] inflight={rep.inflight} submitted={len(rep.submitted)} skipped={len(rep.skipped)}")

    return 0


def _add_submit_subcommand(subparsers) -> None:
    p = subparsers.add_parser("submit", help="Submit VASP jobs (idempotent).")
    p.add_argument("--out-root", required=True, help="Root directory containing 000000/ style job folders")
    p.add_argument("--max-inflight", type=int, default=100, help="Max RUNNING+PENDING jobs at once (default: 100)")
    p.add_argument("--watch", action="store_true", help="Keep feeding jobs until nothing left to submit")
    p.add_argument("--poll-sec", type=int, default=60, help="Polling interval in watch mode (default: 60)")
    p.add_argument("--limit", type=int, default=None, help="Submit at most N new jobs this run")

    # NEW flags
    p.add_argument("--quiet", action="store_true", help="Less output")
    p.add_argument(
        "--no-crosscheck-squeue",
        action="store_true",
        help="Do NOT cross-check inflight with squeue (not recommended; can get stuck if state.json is stale)",
    )
    p.add_argument(
        "--no-status-refresh",
        action="store_true",
        help="In --watch mode, do NOT call status_update() to free capacity (not recommended)",
    )

    p.set_defaults(func=_run_submit)


def _run_resubmit(args: argparse.Namespace) -> int:
    from .submit import resubmit_jobs

    rep = resubmit_jobs(
        out_root=args.out_root,
        max_inflight=args.max_inflight,
        limit=args.limit,
        partition=args.partition,
        nodes=args.nodes,
        ntasks=args.ntasks,
        exclude=args.exclude,
        dry_run=args.dry_run,
        verbose=(not args.quiet),
        crosscheck_squeue=(not args.no_crosscheck_squeue),
        refresh_status=(not args.no_status_refresh),
        strict_slurm_logs=args.strict_slurm_log,
    )

    if args.dry_run:
        action = "would resubmit"
    else:
        action = "resubmitted"

    if not args.quiet:
        if rep.resubmitted:
            print(f"[PotAudit] {action} {len(rep.resubmitted)}: " + ", ".join(rep.resubmitted))
        else:
            print(f"[PotAudit] {action} 0")

        skipped_by_reason = {}
        for _, reason in rep.skipped:
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
        if skipped_by_reason:
            summary = ", ".join(f"{reason}={count}" for reason, count in sorted(skipped_by_reason.items()))
            print(f"[PotAudit] skipped: {summary}")

        print(
            f"[PotAudit] eligible={rep.eligible} "
            f"inflight={rep.inflight} "
            f"remaining_capacity={rep.capacity}"
        )
    else:
        print(f"[PotAudit] {action}={len(rep.resubmitted)} eligible={rep.eligible} skipped={len(rep.skipped)}")

    return 0


def _add_resubmit_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "resubmit",
        help="Resubmit failed VASP jobs only when the failure came from Slurm.",
    )
    p.add_argument("--out-root", required=True, help="Root directory containing VASP job folders")
    p.add_argument("--max-inflight", type=int, default=100, help="Max RUNNING+PENDING jobs at once (default: 100)")
    p.add_argument("--limit", type=int, default=None, help="Resubmit at most N jobs this run")
    p.add_argument("--partition", default=None, help="Rewrite #SBATCH --partition for resubmitted jobs")
    p.add_argument("--nodes", type=int, default=None, help="Rewrite #SBATCH --nodes for resubmitted jobs")
    p.add_argument(
        "--ntasks",
        "--cores",
        dest="ntasks",
        type=int,
        default=None,
        help="Rewrite #SBATCH --ntasks for resubmitted jobs",
    )
    p.add_argument("--exclude", default=None, help="Rewrite/add #SBATCH --exclude, e.g. node001,node002")
    p.add_argument("--dry-run", action="store_true", help="Show jobs that would be resubmitted without calling sbatch")
    p.add_argument(
        "--strict-slurm-log",
        action="store_true",
        help="Only resubmit incomplete VASP outputs when job logs contain Slurm/srun failure markers",
    )
    p.add_argument("--quiet", action="store_true", help="Less output")
    p.add_argument(
        "--no-crosscheck-squeue",
        action="store_true",
        help="Do NOT cross-check inflight with squeue",
    )
    p.add_argument(
        "--no-status-refresh",
        action="store_true",
        help="Do NOT run status_update() before selecting failed jobs",
    )
    p.set_defaults(func=_run_resubmit)


def _run_status(args: argparse.Namespace) -> int:
    from .status import status_update
    rep = status_update(out_root=args.out_root)

    for ln in rep.lines:
        print(ln)

    print(f"[PotAudit] updated={rep.updated} ok={rep.ok} bad={rep.bad} running={rep.running} pending={rep.pending}")
    return 0


def _add_status_subcommand(subparsers) -> None:
    p = subparsers.add_parser("status", help="Update status + validate VASP outputs.")
    p.add_argument("--out-root", required=True, help="Root directory containing 000000/ style job folders")
    p.set_defaults(func=_run_status)

#Collect vasp output as extxyz

def _run_collect_vasp(args: argparse.Namespace) -> int:
    from .collect_vasp import collect_vasp_extxyz
    rep = collect_vasp_extxyz(
        out_root=args.out_root,
        out_extxyz=args.out,
        only_ok=(not args.include_bad),
        require_forces=(not args.allow_missing_forces),
    )
    print(
        f"[PotAudit] wrote={rep.written_frames} "
        f"skipped_not_completed={rep.skipped_not_completed} "
        f"skipped_failed={rep.skipped_failed} "
        f"skipped_missing_output={rep.skipped_missing_output} "
        f"out={rep.out_path}"
    )
    return 0


def _add_collect_vasp_subcommand(subparsers) -> None:
    p = subparsers.add_parser("collect-vasp", help="Merge completed VASP jobs into a multi-frame extxyz (positions/forces/energy).")
    p.add_argument("--out-root", required=True, help="Root directory containing 000000/ style job folders")
    p.add_argument("--out", default="vasp_merged.extxyz", help="Output merged extxyz path")
    p.add_argument("--include-bad", action="store_true", help="Include jobs with vasp_ok=false (not recommended)")
    p.add_argument("--allow-missing-forces", action="store_true", help="Do not require forces to be present")
    p.set_defaults(func=_run_collect_vasp)

# Single point from UMA

def _run_uma_annotate(args: argparse.Namespace) -> int:
    from .uma_annotate import annotate_extxyz_with_uma

    rep = annotate_extxyz_with_uma(
        in_extxyz=args.input,
        out_extxyz=args.out,
        model_name=args.model_name,
        task_name=args.task_name,
        device=args.device,
        overwrite=args.overwrite,
        add_deltas=(not args.no_deltas),
        ft=args.ft,
    )

    print(f"[PotAudit] wrote={rep.written_frames} out={rep.out_path}")
    return 0


def _add_uma_annotate_subcommand(subparsers) -> None:
    p = subparsers.add_parser(
        "uma-annotate",
        help="Compute UMA single-point energy/forces and append to extxyz.",
    )
    p.add_argument("--input", required=True, help="Input extxyz (VASP merged)")
    p.add_argument("--out", default="vasp_plus_uma.extxyz", help="Output extxyz")
    p.add_argument("--model-name", required=True, help="e.g. uma-s-1p1")
    p.add_argument("--task-name", default="omol", help="Default: omol")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-deltas", action="store_true")
    p.add_argument("--ft", action="store_true", help="Load fine-tuned model from model-name path instead of pretrained")
    p.set_defaults(func=_run_uma_annotate)

# Merge extxyz
def _run_merge_extxyz(args):
    from .merge_extxyz import merge_vasp_and_uma_extxyz

    merge_vasp_and_uma_extxyz(
        vasp_extxyz=args.vasp,
        uma_extxyz=args.uma,
        out_extxyz=args.out,
        overwrite=args.overwrite,
    )

    print(f"[PotAudit] merged -> {args.out}")
    return 0

def _add_merge_extxyz_subcommand(subparsers):
    p = subparsers.add_parser(
        "merge-extxyz",
        help="Merge VASP extxyz and UMA extxyz into one file.",
    )
    p.add_argument("--vasp", required=True, help="VASP extxyz file")
    p.add_argument("--uma", required=True, help="UMA extxyz file")
    p.add_argument("--out", default="vasp_plus_uma.extxyz", help="Output extxyz file")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_run_merge_extxyz)


def _run_dpgen_fp(args: argparse.Namespace) -> int:
    from .dpgen_fp import run

    return run(args)


def _add_dpgen_fp_subcommand(subparsers):
    p = subparsers.add_parser(
        "dpgen-fp",
        help="Submit and monitor DP-GEN first-principles task folders.",
    )
    from .dpgen_fp import add_arguments

    add_arguments(p)
    p.set_defaults(func=_run_dpgen_fp)


def _run_dpgen_fp_copy(args: argparse.Namespace) -> int:
    from .dpgen_fp_copy import run

    return run(args)


def _add_dpgen_fp_copy_subcommand(subparsers):
    p = subparsers.add_parser(
        "dpgen-fp-copy",
        aliases=["dpgen-fp-sync"],
        help="Copy completed FP output files from an old DP-GEN FP directory into local tasks.",
    )
    from .dpgen_fp_copy import add_arguments

    add_arguments(p)
    p.set_defaults(func=_run_dpgen_fp_copy)
#------------End Subcommand----------------

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog="potaudit", description="PotAudit: A tool to validate MLIP with DFT (VASP)")
    subparser=parser.add_subparsers(dest="command", required=True)

#------------Register subcommands here----------------
    _add_select_subcommand(subparser)
    _add_prep_vasp_subcommand(subparser)
    _add_prep_vasp_sp_dir_subcommand(subparser)
    _add_prep_vasp_opt_dir_subcommand(subparser)
    _add_select_prep_vasp_subcommand(subparser)
    _add_submit_subcommand(subparser)
    _add_resubmit_subcommand(subparser)
    _add_status_subcommand(subparser)
    _add_collect_vasp_subcommand(subparser)
    _add_uma_annotate_subcommand(subparser)
    _add_merge_extxyz_subcommand(subparser)
    _add_dpgen_fp_subcommand(subparser)
    _add_dpgen_fp_copy_subcommand(subparser)
#------------End subcommands registration----------------

    args=parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
