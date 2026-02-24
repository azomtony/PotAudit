from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ase.io import read

@dataclass(frozen=True)
class Selection:
    """Selected frame indices from a trajectory."""
    indices: List[int]

def select_frames(
        extxyz_path: str,
        *,
        max_frames: int=50,
        stride:Optional[int]=None,
        start: int=0,
        stop: Optional[int]=None,
) -> Selection:
    """
    Select frames from an extxyz with deterministic rules.

    Rules:
      - If stride is provided: take frames start, start+stride, ...
      - If stride is None: compute an "auto stride" to keep <= max_frames frames
      - stop is exclusive (Python slicing convention)
      - Always returns sorted unique indices

    This function is intentionally simple + robust for v1.
    """
    frames = read(extxyz_path, index=':')
    n_total = len(frames)

    if n_total == 0:
        raise ValueError(f"No frames found in {extxyz_path}")
    if start<0:
        start = max(0, n_total + start)
    if stop is None:
        stop = n_total
    elif stop < 0:
        stop = max(0, n_total + stop)
    if start >= stop:
        raise ValueError(f"Invalid start/stop: {start} >= {stop} (total frames: {n_total})")
    
    span=stop-start

    if stride is None:
        # Auto stride: pick evenly spaced frames, capped by max_frames
        if max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {max_frames}")
        stride=max(1, span // max_frames)

    if stride <= 0:
        raise ValueError(f"Stride must be > 0, got {stride}")

    idx=list(range(start, stop, stride))

    #Hard cap
    if len(idx) > max_frames:
        idx=idx[:max_frames]

    return Selection(indices=idx)
    