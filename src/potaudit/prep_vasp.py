from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write, iread


@dataclass(frozen=True)
class PrepReport:
    prepared: List[int]
    skipped: List[int]


@dataclass(frozen=True)
class PrepDirReport:
    prepared: List[str]
    skipped: List[str]


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


def _default_relax_templates_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "templates" / "vasp_relax"


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


def _format_incar_value(value: object) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    return str(value)


def _write_incar_with_overrides(
    *,
    template_path: Path,
    out_path: Path,
    overrides: Dict[str, object],
) -> None:
    """
    Write an INCAR from a template while replacing/appending key-value settings.
    Keeps comments and unrelated template settings intact.
    """
    remaining = {k.upper(): _format_incar_value(v) for k, v in overrides.items()}
    lines: List[str] = []
    key_re = re.compile(r"^(\s*)([A-Za-z][A-Za-z0-9_]*)\s*=.*$")

    for line in template_path.read_text().splitlines():
        m = key_re.match(line)
        if m:
            key = m.group(2).upper()
            if key in remaining:
                lines.append(f"{key:<6} = {remaining.pop(key)}")
                continue
        lines.append(line)

    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# PotAudit INCAR overrides")
        for key in sorted(remaining):
            lines.append(f"{key:<6} = {remaining[key]}")

    out_path.write_text("\n".join(lines) + "\n")


def _safe_job_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    name = name.strip("._-")
    return name or "structure"


def _extxyz_job_base(extxyz_path: Path, input_dir: Path) -> str:
    rel = extxyz_path.relative_to(input_dir)
    stem_parts = list(rel.with_suffix("").parts)
    return _safe_job_name("__".join(stem_parts))


def _as_atoms_list(frames) -> List[Atoms]:
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def _parse_extra_incar_settings(settings: Optional[Iterable[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in settings or []:
        if "=" not in item:
            raise ValueError(f"Invalid INCAR setting '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid INCAR setting '{item}'. Expected KEY=VALUE.")
        parsed[key] = value
    return parsed


def _bottom_layer_indices(
    atoms: Atoms,
    *,
    n_layers: int = 1,
    tolerance: float = 0.5,
    axis: str = "frac-c",
) -> List[int]:
    """
    Identify bottom slab layer(s) by clustering along the slab-normal coordinate.

    axis="frac-c" uses fractional coordinate along the cell c direction. This is
    appropriate for non-orthogonal slab cells as long as the vacuum/slab normal
    is represented by the third cell vector. Tolerance is still in Angstrom.
    """
    if n_layers <= 0:
        return []
    if tolerance < 0:
        raise ValueError("bottom layer tolerance must be >= 0")

    if axis == "frac-c":
        cell = np.asarray(atoms.get_cell(), dtype=float)
        ab_area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        volume = abs(float(np.linalg.det(cell)))
        if ab_area <= 0.0 or volume <= 0.0:
            raise ValueError("Cannot detect fractional-c layers without a valid 3D cell")
        coord_to_angstrom = volume / ab_area
        coord = atoms.get_scaled_positions(wrap=False)[:, 2]
    elif axis == "cart-z":
        coord_to_angstrom = 1.0
        coord = atoms.get_positions()[:, 2]
    else:
        raise ValueError(f"Unsupported bottom layer axis '{axis}'. Use 'frac-c' or 'cart-z'.")

    rows = sorted((float(value), i) for i, value in enumerate(coord))
    if not rows:
        return []

    layers: List[List[int]] = []
    current: List[int] = [rows[0][1]]
    prev_value = rows[0][0]

    for value, idx in rows[1:]:
        gap_angstrom = (value - prev_value) * coord_to_angstrom
        if gap_angstrom <= tolerance:
            current.append(idx)
        else:
            layers.append(current)
            current = [idx]
        prev_value = value

    layers.append(current)

    fixed: List[int] = []
    for layer in layers[:n_layers]:
        fixed.extend(layer)
    return sorted(fixed)


def _fix_bottom_layers(
    atoms: Atoms,
    *,
    n_layers: int,
    tolerance: float,
    axis: str,
) -> List[int]:
    fixed_indices = _bottom_layer_indices(
        atoms,
        n_layers=n_layers,
        tolerance=tolerance,
        axis=axis,
    )
    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return fixed_indices

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

            # lifecycle
            "prepared": True,
            "submitted": False,
            "completed": False,
            "failed": False,

            # slurm tracking
            "slurm_job_id": None,
            "slurm_state": None,
            "slurm_exit_code": None,

            # vasp validation
            "vasp_ok": None,
            "vasp_reason": None,
            "vasp_energy_toten_ev": None,
            "vasp_nions": None,

            # timestamps
            "created_at": datetime.utcnow().isoformat() + "Z",
            "submitted_at": None,
            "completed_at": None,
            "checked_at": None,

            # provenance
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


def prep_vasp_singlepoints_from_extxyz_dir(
    *,
    input_dir: str,
    out_root: str,
    templates_dir: Optional[str] = None,
    potcar_root: Optional[str] = None,
    potcar_suffix: str = "",
    pattern: str = "*_opt.extxyz",
    recursive: bool = False,
    index: str = ":",
    job_prefix: str = "sp",
    incar_set: Optional[Iterable[str]] = None,
    force: bool = False,
) -> PrepDirReport:
    """
    Prepare VASP single-point folders from extxyz files in a directory.

    By default this only reads *_opt.extxyz files, which keeps *_init.extxyz
    files available for later optimization tests.
    """
    input_dir_p = Path(input_dir).resolve()
    if not input_dir_p.exists() or not input_dir_p.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir_p}")

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

    extxyz_files = sorted(input_dir_p.rglob(pattern) if recursive else input_dir_p.glob(pattern))
    extxyz_files = [p for p in extxyz_files if p.is_file()]
    if not extxyz_files:
        raise FileNotFoundError(f"No extxyz files matched pattern '{pattern}' in {input_dir_p}")

    ase_index: object = int(index) if str(index).lstrip("-").isdigit() else index
    incar_overrides = _parse_extra_incar_settings(incar_set)
    prefix = _safe_job_name(job_prefix)

    prepared: List[str] = []
    skipped: List[str] = []
    job_counter = 0

    for extxyz_p in extxyz_files:
        frames = _as_atoms_list(read(str(extxyz_p), index=ase_index))
        if not frames:
            raise ValueError(f"No frames found in {extxyz_p}")

        for frame_ordinal, atoms in enumerate(frames):
            job_name = f"{prefix}_{job_counter:06d}"
            job_counter += 1

            jobdir = out_root_p / job_name
            jobdir.mkdir(parents=True, exist_ok=True)
            state_p = _state_path(jobdir)

            if state_p.exists() and not force:
                skipped.append(job_name)
                continue

            atoms = _group_atoms_by_element(atoms)

            poscar_p = jobdir / "POSCAR"
            write(poscar_p, atoms, format="vasp")

            if incar_overrides:
                _write_incar_with_overrides(
                    template_path=incar_t,
                    out_path=jobdir / "INCAR",
                    overrides=incar_overrides,
                )
            else:
                shutil.copy(incar_t, jobdir / "INCAR")
            shutil.copy(kpoints_t, jobdir / "KPOINTS")
            _render_submit_script(
                template_path=sub_vasp_t,
                out_path=jobdir / "sub_vasp.sh",
                job_name=job_name,
            )

            elements_order = _elements_first_appearance(atoms)
            potcar_p = jobdir / "POTCAR"
            if potroot_p:
                _build_potcar(
                    potcar_root=potroot_p,
                    elements=elements_order,
                    out_path=potcar_p,
                    potcar_suffix=potcar_suffix,
                )

            state = {
                "job_type": "single_point",
                "frame_index": frame_ordinal,

                # lifecycle
                "prepared": True,
                "submitted": False,
                "completed": False,
                "failed": False,

                # slurm tracking
                "slurm_job_id": None,
                "slurm_state": None,
                "slurm_exit_code": None,

                # vasp validation
                "vasp_ok": None,
                "vasp_reason": None,
                "vasp_energy_toten_ev": None,
                "vasp_nions": None,

                # timestamps
                "created_at": datetime.utcnow().isoformat() + "Z",
                "submitted_at": None,
                "completed_at": None,
                "checked_at": None,

                # provenance
                "extxyz_path": str(extxyz_p.resolve()),
                "extxyz_input_dir": str(input_dir_p),
                "extxyz_name": extxyz_p.name,
                "extxyz_index": str(index),
                "source_frame_ordinal": frame_ordinal,
                "templates_dir": str(tdir.resolve()),
                "incar_overrides": {k: _format_incar_value(v) for k, v in incar_overrides.items()},
                "poscar_elements": elements_order,
                "poscar_sha1": _sha1_file(poscar_p),
                "potcar_root": str(potroot_p) if potroot_p else None,
                "potcar_suffix": potcar_suffix,
                "potcar_sha1": _sha1_file(potcar_p) if potcar_p.exists() else None,
            }
            state_p.write_text(json.dumps(state, indent=2) + "\n")

            prepared.append(job_name)

    return PrepDirReport(prepared=prepared, skipped=skipped)


def prep_vasp_optimizations_from_extxyz_dir(
    *,
    input_dir: str,
    out_root: str,
    templates_dir: Optional[str] = None,
    potcar_root: Optional[str] = None,
    potcar_suffix: str = "",
    pattern: str = "*.extxyz",
    recursive: bool = False,
    index: str = ":",
    nsw: Optional[int] = None,
    ibrion: Optional[int] = None,
    isif: Optional[int] = None,
    ediffg: Optional[float] = None,
    incar_set: Optional[Iterable[str]] = None,
    fix_bottom_layers: int = 1,
    bottom_layer_tol: float = 0.5,
    bottom_layer_axis: str = "frac-c",
    force: bool = False,
) -> PrepDirReport:
    """
    Prepare VASP geometry-optimization folders from extxyz files in a directory.

    Each discovered extxyz file contributes the frames selected by ``index``.
    The default index ":" prepares all frames. Single-frame inputs get a job
    folder named after the file stem; multi-frame inputs get a frame suffix.
    """
    input_dir_p = Path(input_dir).resolve()
    if not input_dir_p.exists() or not input_dir_p.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir_p}")

    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    tdir = _default_relax_templates_dir() if templates_dir is None else Path(templates_dir).resolve()
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

    extxyz_files = sorted(input_dir_p.rglob(pattern) if recursive else input_dir_p.glob(pattern))
    extxyz_files = [p for p in extxyz_files if p.is_file()]
    if not extxyz_files:
        raise FileNotFoundError(f"No extxyz files matched pattern '{pattern}' in {input_dir_p}")

    ase_index: object = int(index) if str(index).lstrip("-").isdigit() else index
    incar_overrides: Dict[str, object] = {}
    if ibrion is not None:
        incar_overrides["IBRION"] = ibrion
    if nsw is not None:
        incar_overrides["NSW"] = nsw
    if isif is not None:
        incar_overrides["ISIF"] = isif
    if ediffg is not None:
        incar_overrides["EDIFFG"] = ediffg
    incar_overrides.update(_parse_extra_incar_settings(incar_set))

    prepared: List[str] = []
    skipped: List[str] = []

    for extxyz_p in extxyz_files:
        frames = _as_atoms_list(read(str(extxyz_p), index=ase_index))
        if not frames:
            raise ValueError(f"No frames found in {extxyz_p}")

        base_name = _extxyz_job_base(extxyz_p, input_dir_p)
        for frame_ordinal, atoms in enumerate(frames):
            if len(frames) == 1:
                job_name = base_name
            else:
                job_name = f"{base_name}__frame_{frame_ordinal:06d}"

            jobdir = out_root_p / job_name
            jobdir.mkdir(parents=True, exist_ok=True)
            state_p = _state_path(jobdir)

            if state_p.exists() and not force:
                skipped.append(job_name)
                continue

            atoms = _group_atoms_by_element(atoms)
            fixed_atom_indices = _fix_bottom_layers(
                atoms,
                n_layers=fix_bottom_layers,
                tolerance=bottom_layer_tol,
                axis=bottom_layer_axis,
            )

            poscar_p = jobdir / "POSCAR"
            write(poscar_p, atoms, format="vasp")

            _write_incar_with_overrides(
                template_path=incar_t,
                out_path=jobdir / "INCAR",
                overrides=incar_overrides,
            )
            shutil.copy(kpoints_t, jobdir / "KPOINTS")
            _render_submit_script(
                template_path=sub_vasp_t,
                out_path=jobdir / "sub_vasp.sh",
                job_name=job_name,
            )

            elements_order = _elements_first_appearance(atoms)
            potcar_p = jobdir / "POTCAR"
            if potroot_p:
                _build_potcar(
                    potcar_root=potroot_p,
                    elements=elements_order,
                    out_path=potcar_p,
                    potcar_suffix=potcar_suffix,
                )

            state = {
                "job_type": "optimization",
                "frame_index": frame_ordinal,

                # lifecycle
                "prepared": True,
                "submitted": False,
                "completed": False,
                "failed": False,

                # slurm tracking
                "slurm_job_id": None,
                "slurm_state": None,
                "slurm_exit_code": None,

                # vasp validation
                "vasp_ok": None,
                "vasp_reason": None,
                "vasp_energy_toten_ev": None,
                "vasp_nions": None,

                # timestamps
                "created_at": datetime.utcnow().isoformat() + "Z",
                "submitted_at": None,
                "completed_at": None,
                "checked_at": None,

                # provenance
                "extxyz_path": str(extxyz_p.resolve()),
                "extxyz_input_dir": str(input_dir_p),
                "extxyz_index": str(index),
                "source_frame_ordinal": frame_ordinal,
                "templates_dir": str(tdir.resolve()),
                "incar_overrides": {k: _format_incar_value(v) for k, v in incar_overrides.items()},
                "fix_bottom_layers": fix_bottom_layers,
                "bottom_layer_tol": bottom_layer_tol,
                "bottom_layer_axis": bottom_layer_axis,
                "fixed_atom_indices": fixed_atom_indices,
                "poscar_elements": elements_order,
                "poscar_sha1": _sha1_file(poscar_p),
                "potcar_root": str(potroot_p) if potroot_p else None,
                "potcar_suffix": potcar_suffix,
                "potcar_sha1": _sha1_file(potcar_p) if potcar_p.exists() else None,
            }
            state_p.write_text(json.dumps(state, indent=2) + "\n")

            prepared.append(job_name)

    return PrepDirReport(prepared=prepared, skipped=skipped)


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

def select_and_prep_vasp(
    *,
    extxyz_path: str,
    out_root: str,
    templates_dir: Optional[str] = None,
    potcar_root: Optional[str] = None,
    potcar_suffix: str = "",
    max_frames: int = 50,
    stride: Optional[int] = None,
    start: int = 0,
    stop: Optional[int] = None,
    force: bool = False,
) -> PrepReport:
    """
    Select frames + prepare VASP folders in ONE pass over extxyz.

    Selection rules match select_frames():
      - If stride is provided: pick start, start+stride, ...
      - If stride is None: auto-stride to keep <= max_frames within [start, stop)
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

    # -------- pass 1: count frames (needed for auto-stride / negative start/stop) --------
    n_total = 0
    for _ in iread(extxyz_path):
        n_total += 1
    if n_total == 0:
        raise ValueError(f"No frames found in {extxyz_path}")

    if start < 0:
        start = max(0, n_total + start)

    if stop is None:
        stop = n_total
    elif stop < 0:
        stop = max(0, n_total + stop)

    start = max(0, min(start, n_total))
    stop = max(0, min(stop, n_total))
    if stop <= start:
        raise ValueError(f"Invalid range: start={start}, stop={stop}, n_frames={n_total}")

    span = stop - start
    if stride is None:
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        stride = max(1, span // max_frames)
    if stride <= 0:
        raise ValueError("stride must be > 0")

    # -------- pass 2: stream + prep selected frames --------
    prepared: List[int] = []
    skipped: List[int] = []

    selected_count = 0
    for i, atoms in enumerate(iread(extxyz_path)):
        if i < start or i >= stop:
            continue
        if (i - start) % stride != 0:
            continue

        # hard cap
        if selected_count >= max_frames:
            break
        selected_count += 1

        jobdir = out_root_p / f"{i:06d}"
        jobdir.mkdir(parents=True, exist_ok=True)
        state_p = _state_path(jobdir)

        if state_p.exists() and not force:
            skipped.append(i)
            continue

        atoms = _group_atoms_by_element(atoms)

        # POSCAR
        poscar_p = jobdir / "POSCAR"
        write(poscar_p, atoms, format="vasp")

        # templates
        shutil.copy(incar_t, jobdir / "INCAR")
        shutil.copy(kpoints_t, jobdir / "KPOINTS")
        # Copy templates
        _render_submit_script(
            template_path=sub_vasp_t,
            out_path=jobdir / "sub_vasp.sh",
            job_name=jobdir.name,  # "000123"
        )

        # POTCAR (optional)
        elements_order = _elements_first_appearance(atoms)
        potcar_p = jobdir / "POTCAR"
        if potroot_p:
            _build_potcar(
                potcar_root=potroot_p,
                elements=elements_order,
                out_path=potcar_p,
                potcar_suffix=potcar_suffix,
            )

        # state.json
        state = {
            "frame_index": i,

            # lifecycle
            "prepared": True,
            "submitted": False,
            "completed": False,
            "failed": False,

            # slurm tracking
            "slurm_job_id": None,
            "slurm_state": None,
            "slurm_exit_code": None,

            # vasp validation
            "vasp_ok": None,
            "vasp_reason": None,
            "vasp_energy_toten_ev": None,
            "vasp_nions": None,

            # timestamps
            "created_at": datetime.utcnow().isoformat() + "Z",
            "submitted_at": None,
            "completed_at": None,
            "checked_at": None,

            # provenance
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
