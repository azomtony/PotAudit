from __future__ import annotations

from pathlib import Path
import numpy as np
from ase.io import read, write


def merge_vasp_and_uma_extxyz(
    *,
    vasp_extxyz: str,
    uma_extxyz: str,
    out_extxyz: str,
    overwrite: bool = False,
):
    v_frames = read(vasp_extxyz, index=":")
    u_frames = read(uma_extxyz, index=":")

    if len(v_frames) != len(u_frames):
        raise RuntimeError(
            f"Frame count mismatch: VASP={len(v_frames)} UMA={len(u_frames)}"
        )

    out_path = Path(out_extxyz).resolve()
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Use overwrite=True.")

    merged = []

    for i, (v, u) in enumerate(zip(v_frames, u_frames)):

        if len(v) != len(u):
            raise RuntimeError(
                f"Atom count mismatch at frame {i}: "
                f"VASP={len(v)} UMA={len(u)}"
            )

        if "REF_forces" not in v.arrays:
            raise RuntimeError(
                f"VASP frame {i} missing REF_forces. "
                f"Found arrays={list(v.arrays.keys())}"
            )

        if "uma_forces" not in u.arrays:
            raise RuntimeError(
                f"UMA frame {i} missing uma_forces. "
                f"Found arrays={list(u.arrays.keys())}"
            )

        if "REF_energy" not in v.info:
            raise RuntimeError(
                f"VASP frame {i} missing REF_energy. "
                f"Found info={list(v.info.keys())}"
            )

        if "uma_energy" not in u.info:
            raise RuntimeError(
                f"UMA frame {i} missing uma_energy. "
                f"Found info={list(u.info.keys())}"
            )

        m = v.copy()

        # Keep VASP exactly as-is
        # Append UMA predictions
        m.arrays["uma_forces"] = np.asarray(
            u.arrays["uma_forces"], dtype=float
        )
        m.info["uma_energy"] = float(u.info["uma_energy"])

        # Make sure no accidental duplicate force arrays exist
        for k in ["forces"]:
            if k in m.arrays:
                del m.arrays[k]

        merged.append(m)

    write(str(out_path), merged, format="extxyz")