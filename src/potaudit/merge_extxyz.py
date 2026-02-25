from ase.io import read, write
import numpy as np
from pathlib import Path


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

    for v, u in zip(v_frames, u_frames):

        if len(v) != len(u):
            raise RuntimeError("Atom count mismatch between VASP and UMA frame")

        m = v.copy()

        # copy UMA info
        m.info["uma_energy"] = u.info["uma_energy"]
        m.arrays["uma_forces"] = np.asarray(u.arrays["uma_forces"], dtype=float)

        merged.append(m)

    write(str(out_path), merged, format="extxyz")