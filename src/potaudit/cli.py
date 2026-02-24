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

#------------End Subcommand----------------

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog="potaudit", description="PotAudit: A tool to validate MLIP with DFT (VASP)")
    subparser=parser.add_subparsers(dest="command", required=True)

#------------Register subcommands here----------------
    _add_select_subcommand(subparser)
    _add_prep_vasp_subcommand(subparser)
#------------End subcommands registration----------------

    args=parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())