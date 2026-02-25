from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


@dataclass(frozen=True)
class CollectReport:
    written_frames: int
    skipped_not_completed: int
    skipped_failed: int
    skipped_missing_output: int
    out_path: str
    bad_jobs: List[Tuple[str, str]]  # (jobdir, reason)


@dataclass(frozen=True)
class CollectedFrame:
    job: str
    atoms: Atoms
    energy_ev: float
    ok: bool
    reason: str
    completed: bool


_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)\s+eV")
_FORCE_HEADER_RE = re.compile(
    r"^\s*POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)\s*$", re.MULTILINE
)
_DASH_RE = re.compile(r"^\s*-{5,}\s*$")


def _parse_last_toten(outcar_text: str) -> Optional[float]:
    vals = [float(m.group(1)) for m in _TOTEN_RE.finditer(outcar_text)]
    return vals[-1] if vals else None


def _parse_last_force_block(outcar_text: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (positions[N,3], forces[N,3]) from the LAST
    'POSITION ... TOTAL-FORCE (eV/Angst)' block.
    """
    headers = list(_FORCE_HEADER_RE.finditer(outcar_text))
    if not headers:
        return None

    start = headers[-1].end()
    tail = outcar_text[start:]
    lines = tail.splitlines()

    collecting = False
    pos_rows: List[List[float]] = []
    force_rows: List[List[float]] = []

    for ln in lines:
        s = ln.strip()

        if not collecting and (_DASH_RE.match(ln) or s == ""):
            continue

        if collecting:
            if s.lower().startswith("total drift:"):
                break
            if _DASH_RE.match(ln):
                break
            if s == "":
                break

        parts = s.split()
        if len(parts) >= 6:
            try:
                x, y, z, fx, fy, fz = map(float, parts[:6])
            except ValueError:
                if collecting:
                    break
                continue
            collecting = True
            pos_rows.append([x, y, z])
            force_rows.append([fx, fy, fz])
        else:
            if collecting:
                break

    if not pos_rows:
        return None

    pos = np.asarray(pos_rows, dtype=float)
    frc = np.asarray(force_rows, dtype=float)
    return pos, frc


def _outcar_completed(outcar_text: str) -> bool:
    # Normal VASP footer marker
    return "General timing and accounting" in outcar_text


def collect_one_job_dir(
    jobdir: Path,
    *,
    require_forces: bool,
) -> CollectedFrame:
    poscar = jobdir / "POSCAR"
    outcar = jobdir / "OUTCAR"

    if not poscar.exists():
        return CollectedFrame(jobdir.name, Atoms(), 0.0, False, "missing_POSCAR", False)
    if not outcar.exists() or outcar.stat().st_size < 1000:
        return CollectedFrame(jobdir.name, Atoms(), 0.0, False, "missing_or_small_OUTCAR", False)

    base = read(str(poscar), format="vasp")
    symbols = base.get_chemical_symbols()
    cell = base.get_cell()
    pbc = base.get_pbc()

    txt = outcar.read_text(errors="ignore")
    completed = _outcar_completed(txt)

    toten = _parse_last_toten(txt)
    if toten is None:
        return CollectedFrame(jobdir.name, Atoms(), 0.0, False, "missing_TOTEN", completed)

    fb = _parse_last_force_block(txt)
    if fb is None:
        if require_forces:
            return CollectedFrame(jobdir.name, Atoms(), float(toten), False, "missing_force_block", completed)

        # If forces are not required, still output positions (from POSCAR)
        atoms = Atoms(symbols=symbols, positions=base.get_positions(), cell=cell, pbc=pbc)
        atoms.info["energy"] = float(toten)
        atoms.info["potaudit_job"] = jobdir.name
        return CollectedFrame(jobdir.name, atoms, float(toten), True, "ok_no_forces", completed)

    pos, frc = fb
    if pos.shape[0] != len(symbols):
        return CollectedFrame(
            jobdir.name,
            Atoms(),
            float(toten),
            False,
            f"nions_mismatch_poscar={len(symbols)}_outcar={pos.shape[0]}",
            completed,
        )

    atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=pbc)
    atoms.info["energy"] = float(toten)
    atoms.arrays["forces"] = frc
    atoms.info["potaudit_job"] = jobdir.name

    return CollectedFrame(jobdir.name, atoms, float(toten), True, "ok", completed)


def collect_vasp_extxyz(
    *,
    out_root: str,
    out_extxyz: str,
    only_ok: bool = True,
    require_forces: bool = True,
) -> CollectReport:
    """
    Collect a merged extxyz from VASP job folders using only POSCAR + OUTCAR.
    Does NOT read state.json.

    only_ok=True -> require OUTCAR to look completed (footer present).
    require_forces=True -> require force block in OUTCAR.
    """
    out_root_p = Path(out_root).resolve()
    if not out_root_p.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_p}")

    jobs = [d for d in sorted(out_root_p.iterdir()) if d.is_dir()]

    frames: List[Atoms] = []
    bad_jobs: List[Tuple[str, str]] = []

    skipped_not_completed = 0
    skipped_failed = 0
    skipped_missing_output = 0

    for d in jobs:
        rep = collect_one_job_dir(d, require_forces=require_forces)

        if not rep.ok:
            bad_jobs.append((rep.job, rep.reason))
            if rep.reason.startswith("missing_") or rep.reason == "missing_or_small_OUTCAR":
                skipped_missing_output += 1
            else:
                skipped_failed += 1
            continue

        if only_ok and not rep.completed:
            bad_jobs.append((rep.job, "not_completed_OUTCAR_footer_missing"))
            skipped_not_completed += 1
            continue

        frames.append(rep.atoms)

    if not frames:
        msg = "No frames collected.\n"
        msg += f"Checked {len(jobs)} job folders.\n"
        msg += f"only_ok={only_ok} require_forces={require_forces}\n"
        if bad_jobs:
            msg += "Top failures:\n"
            for j, r in bad_jobs[:30]:
                msg += f"  {j}: {r}\n"
        raise RuntimeError(msg)

    out_path = Path(out_extxyz).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), frames, format="extxyz")

    return CollectReport(
        written_frames=len(frames),
        skipped_not_completed=skipped_not_completed,
        skipped_failed=skipped_failed,
        skipped_missing_output=skipped_missing_output,
        out_path=str(out_path),
        bad_jobs=bad_jobs,
    )