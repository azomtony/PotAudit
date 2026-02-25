from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


@dataclass(frozen=True)
class CollectedFrame:
    job: str
    atoms: Atoms
    energy_ev: float
    ok: bool
    reason: str


_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)\s+eV")
_FORCE_HEADER_RE = re.compile(r"^\s*POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)\s*$", re.MULTILINE)
_DASH_RE = re.compile(r"^\s*-{5,}\s*$")


def _parse_last_toten(outcar_text: str) -> Optional[float]:
    vals = [float(m.group(1)) for m in _TOTEN_RE.finditer(outcar_text)]
    return vals[-1] if vals else None


def _parse_last_force_block(outcar_text: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (positions[N,3], forces[N,3]) from the LAST
    'POSITION ... TOTAL-FORCE (eV/Angst)' block.
    """
    # find all headers
    headers = list(_FORCE_HEADER_RE.finditer(outcar_text))
    if not headers:
        return None

    # take the last header and parse after it
    start = headers[-1].end()
    tail = outcar_text[start:]

    # The block typically starts after dashed line(s)
    # We'll scan line-by-line, collecting rows until we hit:
    # - "total drift:" line, or
    # - a dashed separator after rows, or
    # - blank line after we started collecting
    lines = tail.splitlines()

    collecting = False
    pos_rows: List[List[float]] = []
    force_rows: List[List[float]] = []

    for ln in lines:
        s = ln.strip()

        # skip initial separators
        if not collecting and (_DASH_RE.match(ln) or s == ""):
            continue

        # stop conditions
        if collecting:
            if s.lower().startswith("total drift:"):
                break
            if _DASH_RE.match(ln):
                # many OUTCARs have dashed line after the table
                break
            if s == "":
                break

        # parse a data row: x y z fx fy fz
        parts = s.split()
        if len(parts) >= 6:
            try:
                x, y, z, fx, fy, fz = map(float, parts[:6])
            except ValueError:
                # not a numeric line
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


def collect_one_job_dir(jobdir: Path) -> CollectedFrame:
    poscar = jobdir / "POSCAR"
    outcar = jobdir / "OUTCAR"
    if not poscar.exists():
        return CollectedFrame(jobdir.name, Atoms(), 0.0, False, "missing_POSCAR")
    if not outcar.exists() or outcar.stat().st_size < 1000:
        return CollectedFrame(jobdir.name, Atoms(), 0.0, False, "missing_or_small_OUTCAR")

    # read POSCAR for symbols/cell/pbc
    base = read(str(poscar), format="vasp")
    symbols = base.get_chemical_symbols()
    cell = base.get_cell()
    pbc = base.get_pbc()

    txt = outcar.read_text(errors="ignore")

    toten = _parse_last_toten(txt)
    if toten is None:
        return CollectedFrame(jobdir.name, Atoms(), 0.0, False, "missing_TOTEN")

    fb = _parse_last_force_block(txt)
    if fb is None:
        return CollectedFrame(jobdir.name, Atoms(), toten, False, "missing_force_block")
    pos, frc = fb

    if pos.shape[0] != len(symbols):
        return CollectedFrame(
            jobdir.name, Atoms(), toten, False,
            f"nions_mismatch_poscar={len(symbols)}_outcar={pos.shape[0]}"
        )

    atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=pbc)
    atoms.info["energy"] = float(toten)
    atoms.arrays["forces"] = frc

    # keep provenance
    atoms.info["potaudit_job"] = jobdir.name

    return CollectedFrame(jobdir.name, atoms, float(toten), True, "ok")


def collect_vasp_extxyz(
    *,
    out_root: str,
    out_path: str,
    include_bad: bool = False,
) -> None:
    out_root_p = Path(out_root).resolve()
    jobs = [d for d in sorted(out_root_p.iterdir()) if d.is_dir()]

    frames: List[Atoms] = []
    bad: List[Tuple[str, str]] = []

    for d in jobs:
        rep = collect_one_job_dir(d)
        if rep.ok:
            frames.append(rep.atoms)
        else:
            bad.append((rep.job, rep.reason))
            if include_bad:
                # optional: you could still write something, but usually skip
                pass

    if not frames:
        # print a helpful summary
        msg = "No frames collected.\n"
        msg += f"Checked {len(jobs)} job folders.\n"
        if bad:
            msg += "Top failures:\n"
            for j, r in bad[:20]:
                msg += f"  {j}: {r}\n"
        raise RuntimeError(msg)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write(out_path, frames, format="extxyz")
    print(f"[PotAudit] collected {len(frames)} frames -> {out_path}")
    if bad:
        print(f"[PotAudit] skipped {len(bad)} bad jobs (showing up to 20):")
        for j, r in bad[:20]:
            print(f"  {j}: {r}")