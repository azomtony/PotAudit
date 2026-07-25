from __future__ import annotations

import argparse
import fnmatch
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConvergenceResult:
    task: str
    system: str
    status: str
    reason: str
    normal_exit: bool
    electronic_converged: bool | None
    ediff_reached_count: int
    nelm: int | None
    max_scf_steps: int | None
    hit_nelm_steps: list[int]
    ionic_steps: int
    nsw: int | None
    ionic_converged: bool | None


def _task_system(task_name: str) -> str:
    parts = task_name.split(".")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def find_tasks(fp_dir: str | Path, task_glob: str, recursive: bool = False) -> list[Path]:
    root = Path(fp_dir)
    if recursive:
        return sorted(p for p in root.rglob("task.*") if p.is_dir() and fnmatch.fnmatch(p.name, task_glob))
    return sorted(p for p in root.glob(task_glob) if p.is_dir() and fnmatch.fnmatch(p.name, task_glob))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def _parse_int_setting(text: str, key: str) -> int | None:
    matches = list(re.finditer(rf"\b{re.escape(key)}\s*=\s*(-?\d+)", text))
    if not matches:
        return None
    return int(matches[-1].group(1))


def _setting_from_outcar_or_incar(task_dir: Path, outcar_text: str, key: str) -> int | None:
    value = _parse_int_setting(outcar_text, key)
    if value is not None:
        return value
    incar = task_dir / "INCAR"
    if incar.is_file():
        return _parse_int_setting(_read_text(incar), key)
    return None


def _outcar_scf_steps_by_ionic_step(outcar_text: str) -> dict[int, int]:
    steps: dict[int, int] = {}
    for match in re.finditer(r"\bIteration\s+(\d+)\(\s*(\d+)\)", outcar_text):
        ionic_step = int(match.group(1))
        electronic_step = int(match.group(2))
        steps[ionic_step] = max(steps.get(ionic_step, 0), electronic_step)
    return steps


def _oszicar_scf_steps_by_ionic_step(oszicar_text: str) -> dict[int, int]:
    steps: dict[int, int] = {}
    current = 0
    ionic_step = 0

    for line in oszicar_text.splitlines():
        if re.match(r"\s*(DAV|RMM|CG|N)\s*:", line):
            current += 1
            continue
        if re.match(r"\s*\d+\s+(F=|E=|T=)", line):
            ionic_step += 1
            if current:
                steps[ionic_step] = current
            current = 0

    if current:
        steps[ionic_step + 1] = current

    return steps


def _scf_steps_by_ionic_step(task_dir: Path, outcar_text: str) -> dict[int, int]:
    steps = _outcar_scf_steps_by_ionic_step(outcar_text)
    if steps:
        return steps

    oszicar = task_dir / "OSZICAR"
    if oszicar.is_file():
        return _oszicar_scf_steps_by_ionic_step(_read_text(oszicar))
    return {}


def _ediff_reached_count(outcar_text: str) -> int:
    return len(re.findall(r"aborting\s+loop\s+because\s+EDIFF\s+is\s+reached", outcar_text, re.I))


def check_vasp_task(task_dir: Path, *, nelm_override: int | None = None, check_ionic: bool = False) -> ConvergenceResult:
    outcar = task_dir / "OUTCAR"
    if not outcar.is_file():
        return ConvergenceResult(
            task=task_dir.name,
            system=_task_system(task_dir.name),
            status="incomplete",
            reason="missing_OUTCAR",
            normal_exit=False,
            electronic_converged=None,
            ediff_reached_count=0,
            nelm=nelm_override,
            max_scf_steps=None,
            hit_nelm_steps=[],
            ionic_steps=0,
            nsw=None,
            ionic_converged=None,
        )

    outcar_text = _read_text(outcar)
    normal_exit = "General timing and accounting" in outcar_text
    nelm = nelm_override or _setting_from_outcar_or_incar(task_dir, outcar_text, "NELM")
    nsw = _setting_from_outcar_or_incar(task_dir, outcar_text, "NSW")
    scf_steps = _scf_steps_by_ionic_step(task_dir, outcar_text)
    max_scf_steps = max(scf_steps.values()) if scf_steps else None
    ionic_steps = max(scf_steps.keys()) if scf_steps else 0
    ediff_reached_count = _ediff_reached_count(outcar_text)

    hit_nelm_steps: list[int] = []
    electronic_converged: bool | None
    if ionic_steps > 0:
        electronic_converged = ediff_reached_count >= ionic_steps
    elif ediff_reached_count > 0:
        electronic_converged = True
    else:
        electronic_converged = False

    if nelm is not None and scf_steps:
        hit_nelm_steps = [step for step, count in sorted(scf_steps.items()) if count >= nelm]

    ionic_converged: bool | None = None
    if check_ionic and nsw is not None and nsw > 0:
        ionic_converged = "reached required accuracy - stopping structural energy minimisation" in outcar_text

    reasons: list[str] = []
    if not normal_exit:
        reasons.append("no_timing_footer")
    if electronic_converged is False:
        if hit_nelm_steps:
            reasons.append("scf_hit_NELM_steps=" + ",".join(str(x) for x in hit_nelm_steps))
        if ediff_reached_count == 0:
            reasons.append("no_EDIFF_reached")
        elif ediff_reached_count < ionic_steps:
            reasons.append(f"EDIFF_reached={ediff_reached_count}/{ionic_steps}")
    elif electronic_converged is None:
        reasons.append("scf_convergence_unknown")
    if check_ionic and ionic_converged is False:
        reasons.append("ionic_not_converged")

    if not normal_exit:
        status = "incomplete"
    elif electronic_converged is False or (check_ionic and ionic_converged is False):
        status = "not_converged"
    elif electronic_converged is None:
        status = "unknown"
    else:
        status = "converged"

    return ConvergenceResult(
        task=task_dir.name,
        system=_task_system(task_dir.name),
        status=status,
        reason="ok" if not reasons else ";".join(reasons),
        normal_exit=normal_exit,
        electronic_converged=electronic_converged,
        ediff_reached_count=ediff_reached_count,
        nelm=nelm,
        max_scf_steps=max_scf_steps,
        hit_nelm_steps=hit_nelm_steps,
        ionic_steps=ionic_steps,
        nsw=nsw,
        ionic_converged=ionic_converged,
    )


def _format_result(result: ConvergenceResult) -> str:
    label = "not converged" if result.status == "not_converged" else result.status
    steps_done = result.max_scf_steps if result.max_scf_steps is not None else "NA"
    nelm = result.nelm if result.nelm is not None else "NA"
    detail = f"steps_completed={steps_done}/{nelm} ionic_steps={result.ionic_steps}"
    if result.status != "converged":
        detail += f" reason={result.reason}"
    return f"{result.task} {label} {detail}"


def _print_grouped_results(results: list[ConvergenceResult], *, only_problems: bool) -> None:
    by_system: dict[str, list[ConvergenceResult]] = {}
    for result in results:
        if only_problems and result.status == "converged":
            continue
        by_system.setdefault(result.system, []).append(result)

    for system in sorted(by_system):
        print(f"------task.{system}-----")
        for result in sorted(by_system[system], key=lambda item: item.task):
            print(_format_result(result))


def run(args: argparse.Namespace) -> int:
    tasks = find_tasks(args.fp_dir, args.task_glob, recursive=args.recursive)
    results = [
        check_vasp_task(task, nelm_override=args.nelm, check_ionic=args.check_ionic)
        for task in tasks
    ]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        _print_grouped_results(results, only_problems=args.only_problems)

        counts: dict[str, int] = {}
        systems: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
            if result.status != "converged":
                systems[result.system] = systems.get(result.system, 0) + 1

        print()
        print("summary")
        print("checked:", len(results))
        for key in ("converged", "not_converged", "incomplete", "unknown"):
            print(f"{key}:", counts.get(key, 0))
        if systems:
            print("problem systems:", ", ".join(f"{system}:{count}" for system, count in sorted(systems.items())))

    if args.fail_on_problems and any(result.status != "converged" for result in results):
        return 1
    return 0


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--fp-dir", required=True, help="DP-GEN FP directory, e.g. iter.000001/02.fp")
    p.add_argument("--task-glob", default="task.*", help='Task glob, e.g. "task.000.*"')
    p.add_argument("--recursive", action="store_true", help="Search for task folders recursively")
    p.add_argument("--nelm", type=int, default=None, help="Override NELM if OUTCAR/INCAR cannot be trusted")
    p.add_argument("--check-ionic", action="store_true", help="Also flag VASP relaxations that did not satisfy EDIFFG")
    p.add_argument("--only-problems", action="store_true", help="Only print incomplete, unknown, or unconverged tasks")
    p.add_argument("--json", action="store_true", help="Write machine-readable JSON")
    p.add_argument("--fail-on-problems", action="store_true", help="Exit nonzero when any task is not converged")
    return p


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(argparse.ArgumentParser(prog="dpgen-fp-converge"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
