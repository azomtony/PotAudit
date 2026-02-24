from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from ase.io import read, write


@dataclass(frozen=True)
class PrepReport:
    prepared: List[int]
    skipped: List[int]


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_indices(indices_path: Path) -> List[int]:
    idx: List[int] = []
    for ln in indices_path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        idx.append(int(s))
    return idx


def _state_path(jobdir: Path) -> Path:
    return jobdir / "state.json"


def _elements_first_appearance(atoms) -> List[str]:
    """
    Return unique element symbols in the order they first appear in the Atoms object.
    This is a deterministic, ASE-based way to define POTCAR concatenation order.
    """
    seen = set()
    order: List[str] = []
    for s in atoms.get_chemical_symbols():
        if s not in seen:
            seen.add(s)
            order.append(s)
    return order

def _default_templates_dir() -> Path:
    """
    Repo layout:
      <repo>/src/potaudit/prep_vasp.py  (this file)
      <repo>/templates/vasp/INCAR
    """
    repo_root = Path(__file__).resolve().parents[2]  # potaudit -> src -> repo
    return repo_root / "templates" / "vasp"
from ase import Atoms

def _group_atoms_by_element(atoms) -> Atoms:
    """
    Return a new Atoms object where atoms are grouped by element
    (all C first, then H, then O, ...), preserving cell and pbc.
    Element order follows first appearance in the original atoms.
    """
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    elems = []
    seen = set()
    for s in syms:
        if s not in seen:
            seen.add(s)
            elems.append(s)

    new_syms = []
    new_pos = []
    for e in elems:
        for s, r in zip(syms, pos):
            if s == e:
                new_syms.append(s)
                new_pos.append(r)

    new_atoms = Atoms(symbols=new_syms, positions=new_pos, cell=cell, pbc=pbc)
    new_atoms.info = dict(atoms.info)  # keep metadata if you want
    return new_atoms
def _render_submit_script(
    *,
    template_path: Path,
    out_path: Path,
    job_name: str,
) -> None:
    txt = template_path.read_text()
    txt = txt.replace("__JOB_NAME__", job_name)
    out_path.write_text(txt)
    out_path.chmod(0o755)

def _build_potcar(
    *,
    potcar_root: Path,
    elements: List[str],
    out_path: Path,
    potcar_suffix: str = "",
) -> None:
    """
    Build combined POTCAR by concatenating:
      <potcar_root>/<Element{potcar_suffix}>/POTCAR

    Examples:
      potcar_suffix=""        -> C/POTCAR
      potcar_suffix="_GW"     -> C_GW/POTCAR
      potcar_suffix="_pv"     -> Ag_pv/POTCAR
      potcar_suffix="_d"      -> As_d/POTCAR
      potcar_suffix="_d_GW"   -> At_d_GW/POTCAR
      potcar_suffix="_sv_GW"  -> Au_sv_GW/POTCAR
    """
    parts: List[Path] = []
    for el in elements:
        part = potcar_root / f"{el}{potcar_suffix}" / "POTCAR"
        if not part.exists():
            raise FileNotFoundError(
                f"Missing POTCAR part: element='{el}', suffix='{potcar_suffix}'. Expected: {part}"
            )
        parts.append(part)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as w:
        for p in parts:
            w.write(p.read_bytes())


def prep_vasp(
    *,
    extxyz_path: str,
    indices: Iterable[int],
    out_root: str,
    templates_dir: Optional[str] = None,
    potcar_root: Optional[str] = None,
    potcar_suffix: str = "",
    force: bool = False,
) -> PrepReport:
    """
    Prepare per-frame VASP single-point folders:
      - POSCAR (from extxyz frame)
      - INCAR, KPOINTS (from templates_dir)
      - POTCAR (built from potcar_root + element order from ASE), optional
      - state.json (tracks what was generated)

    Idempotency:
      - If state.json exists and force=False, we skip that frame.
    """
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    tdir = _default_templates_dir() if templates_dir is None else Path(templates_dir).resolve()
    incar_t = tdir / "INCAR"
    kpoints_t = tdir / "KPOINTS"
    sub_vasp_t = tdir / "sub_vasp.sh"  
    if not incar_t.exists():
        raise FileNotFoundError(f"Missing INCAR template: {incar_t}")
    if not kpoints_t.exists():
        raise FileNotFoundError(f"Missing KPOINTS template: {kpoints_t}")
    if not sub_vasp_t.exists():
        raise FileNotFoundError(f"Missing sub_vasp.sh template: {sub_vasp_t}")

    potroot_p = Path(potcar_root).resolve() if potcar_root else None
    if potroot_p and not potroot_p.exists():
        raise FileNotFoundError(f"POTCAR root not found: {potroot_p}")

    frames = read(extxyz_path, index=":")
    n_total = len(frames)
    if n_total == 0:
        raise ValueError(f"No frames found in {extxyz_path}")

    prepared: List[int] = []
    skipped: List[int] = []

    for i in indices:
        if i < 0 or i >= n_total:
            raise ValueError(f"Frame index {i} out of range (0..{n_total-1})")

        jobdir = out_root_p / f"{i:06d}"
        jobdir.mkdir(parents=True, exist_ok=True)
        state_p = _state_path(jobdir)

        if state_p.exists() and not force:
            skipped.append(i)
            continue

        atoms = frames[i]
        atoms = _group_atoms_by_element(atoms)

        # Write POSCAR
        poscar_p = jobdir / "POSCAR"
        write(poscar_p, atoms, format="vasp")

        # Copy templates
        shutil.copy(incar_t, jobdir / "INCAR")
        shutil.copy(kpoints_t, jobdir / "KPOINTS")
        _render_submit_script(
            template_path=sub_vasp_t,
            out_path=jobdir / "sub_vasp.sh",
            job_name=jobdir.name,  # "000123"
        )

        # Build POTCAR (optional)
        elements_order = _elements_first_appearance(atoms)
        potcar_p = jobdir / "POTCAR"
        if potroot_p:
            _build_potcar(
                potcar_root=potroot_p,
                elements=elements_order,
                out_path=potcar_p,
                potcar_suffix=potcar_suffix,
            )

        # Write state.json
        state = {
            "frame_index": i,
            "prepared": True,
            "submitted": False,
            "completed": False,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "extxyz_path": str(Path(extxyz_path).resolve()),
            "templates_dir": str(tdir.resolve()),
            "poscar_elements": elements_order,
            "poscar_sha1": _sha1_file(poscar_p),
            "potcar_root": str(potroot_p) if potroot_p else None,
            "potcar_suffix": potcar_suffix,
            "potcar_sha1": _sha1_file(potcar_p) if potcar_p.exists() else None,
        }
        state_p.write_text(json.dumps(state, indent=2) + "\n")

        prepared.append(i)

    return PrepReport(prepared=prepared, skipped=skipped)


def prep_vasp_from_indices_file(
    *,
    extxyz_path: str,
    indices_file: str,
    out_root: str,
    templates_dir: str,
    potcar_root: Optional[str] = None,
    potcar_suffix: str = "",
    force: bool = False,
) -> PrepReport:
    idx = _load_indices(Path(indices_file))
    return prep_vasp(
        extxyz_path=extxyz_path,
        indices=idx,
        out_root=out_root,
        templates_dir=templates_dir,
        potcar_root=potcar_root,
        potcar_suffix=potcar_suffix,
        force=force,
    )