from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ase import Atoms
from ase.io import read, write


@dataclass(frozen=True)
class CollectReport:
    written_frames: int
    skipped_not_completed: int
    skipped_failed: int
    skipped_missing_output: int
    jobdirs_total: int
    out_path: str


def _read_state(p: Path) -> Dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _state_path(jobdir: Path) -> Path:
    return jobdir / "state.json"


def _pick_vasp_result_file(jobdir: Path) -> Optional[Path]:
    """
    Prefer vasprun.xml; fallback to OUTCAR if needed.
    """
    vxml = jobdir / "vasprun.xml"
    if vxml.exists() and vxml.stat().st_size > 1000:
        return vxml
    outcar = jobdir / "OUTCAR"
    if outcar.exists() and outcar.stat().st_size > 1000:
        return outcar
    return None


def _load_vasp_atoms(result_path: Path) -> Atoms:
    """
    Reads VASP outputs via ASE and returns an Atoms containing:
      - positions + cell + pbc
      - energies/forces (if ASE can parse them)
    """
    # ASE can read vasprun.xml directly. OUTCAR read can be heavier but works.
    a = read(str(result_path))

    # Ensure pbc is True True True for periodic systems if missing
    if a.get_pbc() is None or (hasattr(a.get_pbc(), "__len__") and len(a.get_pbc()) == 0):
        a.set_pbc([True, True, True])

    return a


def _get_energy_and_forces(atoms: Atoms) -> Tuple[Optional[float], Optional[List[List[float]]]]:
    """
    Extract energy/forces robustly.
    Returns energy (eV) and forces (eV/Å) if available.
    """
    e = None
    f = None

    # Energy
    try:
        e = float(atoms.get_potential_energy())
    except Exception:
        e = None

    # Forces
    try:
        f_arr = atoms.get_forces()
        f = f_arr.tolist()
    except Exception:
        f = None

    return e, f


def collect_vasp_extxyz(
    *,
    out_root: str,
    out_extxyz: str,
    only_ok: bool = True,
    require_forces: bool = True,
    sort_by_folder: bool = True,
) -> CollectReport:
    out_root_p = Path(out_root).resolve()
    if not out_root_p.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_p}")

    jobdirs = [p for p in out_root_p.iterdir() if p.is_dir()]
    if sort_by_folder:
        jobdirs = sorted(jobdirs, key=lambda p: p.name)

    frames: List[Atoms] = []

    skipped_not_completed = 0
    skipped_failed = 0
    skipped_missing_output = 0

    for jobdir in jobdirs:
        st = _read_state(_state_path(jobdir))

        # We only merge prepared jobs; ignore random folders
        if not st:
            continue

        if not st.get("completed", False):
            skipped_not_completed += 1
            continue

        if only_ok and not st.get("vasp_ok", False):
            skipped_failed += 1
            continue

        result_path = _pick_vasp_result_file(jobdir)
        if result_path is None:
            skipped_missing_output += 1
            continue

        atoms = _load_vasp_atoms(result_path)
        energy, forces = _get_energy_and_forces(atoms)

        # If you demand forces, skip frames without forces
        if require_forces and forces is None:
            skipped_failed += 1
            continue

        # Attach metadata in atoms.info (ASE writes these into extxyz header)
        atoms.info = dict(atoms.info) if atoms.info else {}
        atoms.info["potaudit_jobdir"] = jobdir.name
        atoms.info["potaudit_frame_index"] = st.get("frame_index")
        atoms.info["vasp_energy_ev"] = energy
        atoms.info["vasp_toten_ev"] = st.get("vasp_energy_toten_ev", energy)
        atoms.info["slurm_job_id"] = st.get("slurm_job_id")
        atoms.info["slurm_state"] = st.get("slurm_state")

        # Make sure ASE writes forces: it writes calculator results; easiest is set arrays.
        # We'll store VASP forces explicitly as an array named "forces" so extxyz keeps it.
        if forces is not None:
            import numpy as np
            atoms.arrays["forces"] = np.array(forces, dtype=float)

        frames.append(atoms)

    out_path = Path(out_extxyz).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError(
            "No frames collected. Check that jobs are completed and vasp_ok=true "
            "and that vasprun.xml/OUTCAR exist."
        )

    # Write multi-frame extxyz
    write(str(out_path), frames, format="extxyz")

    return CollectReport(
        written_frames=len(frames),
        skipped_not_completed=skipped_not_completed,
        skipped_failed=skipped_failed,
        skipped_missing_output=skipped_missing_output,
        jobdirs_total=len(jobdirs),
        out_path=str(out_path),
    )