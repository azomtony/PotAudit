from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read, write


@dataclass(frozen=True)
class UMAAnnotateReport:
    written_frames: int
    out_path: str


def _build_uma_calculator(
    *,
    model_name: str,
    task_name: str = "omol",
    device: str = "cuda",
    ft:bool = True,
):
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    if ft:
        print(f"Loading fine-tuned model from {model_name} on device {device}...")
        predictor = load_predict_unit(model_name, device=device)
    else:
        print(f"Loading pretrained model {model_name} on device {device}...")   
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    calc = FAIRChemCalculator(predictor, task_name=task_name)
    return calc


def annotate_extxyz_with_uma(
    *,
    in_extxyz: str,
    out_extxyz: str,
    model_name: str,
    task_name: str = "omol",
    device: str = "cuda",
    overwrite: bool = False,
    add_deltas: bool = True,
    ft:bool = True,
) -> UMAAnnotateReport:
    frames = read(in_extxyz, index=":")
    if len(frames) == 0:
        raise ValueError(f"No frames found in {in_extxyz}")

    out_path = Path(out_extxyz).resolve()
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to replace it.")

    calc = _build_uma_calculator(model_name=model_name, task_name=task_name, device=device, ft=ft)

    annotated = []

    for atoms in frames:
        a = atoms.copy()

        # --- SINGLE POINT with UMA ---
        a.calc = calc
        e_uma = float(a.get_potential_energy())      # eV
        f_uma = np.asarray(a.get_forces(), dtype=float)  # eV/Å

        # Store UMA predictions under explicit keys
        a.info["uma_energy"] = e_uma
        a.arrays["uma_forces"] = f_uma

        if add_deltas:
            # Compare to VASP values already in the extxyz (energy/forces)
            e_vasp = a.info.get("energy", None)
            if e_vasp is not None:
                a.info["delta_energy"] = e_uma - float(e_vasp)

            f_vasp = a.arrays.get("forces", None)
            if f_vasp is not None:
                dv = f_uma - np.asarray(f_vasp, dtype=float)
                a.info["force_rmse"] = float(np.sqrt(np.mean(dv * dv)))

        # CRITICAL: detach calculator so ASE writer won't overwrite `energy`/`forces`
        a.calc = None

        annotated.append(a)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), annotated, format="extxyz")

    return UMAAnnotateReport(written_frames=len(annotated), out_path=str(out_path))