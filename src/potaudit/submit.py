from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# If you have status_update() in potaudit.status, we can call it in --watch mode.
# If you prefer not to import here, you can pass a callback instead.
try:
    from .status import status_update  # type: ignore
except Exception:  # pragma: no cover
    status_update = None  # optional


@dataclass(frozen=True)
class SubmitReport:
    submitted: List[str]   # job folder names (e.g., "000123")
    skipped: List[str]     # already submitted/completed/etc
    inflight: int
    capacity: int
    remaining_ready: int   # ready-to-submit jobs still left after this pass


@dataclass(frozen=True)
class ResubmitReport:
    resubmitted: List[str]
    skipped: List[Tuple[str, str]]
    inflight: int
    capacity: int
    eligible: int
    status_error: Optional[str] = None


_LOG_RESUBMITTABLE_VASP_REASONS = {
    "missing_or_small_OUTCAR",
    "no_timing_footer",
    "no_TOTEN_found",
    "no_force_block",
}

_SLURM_FAILURE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("node_fail", re.compile(r"\bNODE_FAIL\b|node failure|nodes? .*(failed|down)", re.I)),
    ("time_limit", re.compile(r"time limit|DUE TO TIME LIMIT", re.I)),
    ("oom", re.compile(r"OUT_OF_MEMORY|oom[-_ ]?kill|killed process|detected .* oom", re.I)),
    ("cancelled", re.compile(r"\bCANCELLED\b|job .* cancelled", re.I)),
    ("srun_error", re.compile(r"\bsrun:\s+error:|\bslurmstepd:\s+error:", re.I)),
    ("step_aborted", re.compile(r"job step aborted|unable to create step|launch failed", re.I)),
)


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _read_state(p: Path) -> Dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _write_state(p: Path, state: Dict) -> None:
    p.write_text(json.dumps(state, indent=2) + "\n")


def _state_path(jobdir: Path) -> Path:
    return jobdir / "state.json"


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    return r.stdout.strip()


def _parse_sbatch_jobid(s: str) -> str:
    # "Submitted batch job 29496284"
    m = re.search(r"Submitted batch job\s+(\d+)", s)
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from: {s}")
    return m.group(1)


def _squeue_jobids() -> Set[str]:
    """
    Returns set of job IDs currently in Slurm queue for *this user*.
    This is safer than trusting state.json alone, especially after restarts.
    """
    try:
        out = _run(["squeue", "-h", "-o", "%A"])
    except Exception:
        return set()
    ids: Set[str] = set()
    for ln in out.splitlines():
        s = ln.strip()
        if s:
            ids.add(s)
    return ids


def _inflight_count(out_root: Path, crosscheck_squeue: bool = True) -> int:
    """
    Inflight = jobs that are submitted but not completed.
    If crosscheck_squeue=True, only count those whose jobid is still in squeue.
    This avoids "inflight stuck at 50" due to stale state.json.
    """
    q = _squeue_jobids() if crosscheck_squeue else set()

    n = 0
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        st = _read_state(_state_path(d))
        if not st:
            continue
        if st.get("submitted") and not st.get("completed"):
            jid = st.get("slurm_job_id")
            # If no jid, treat as inflight only if you *want* to be conservative.
            # Better: treat as NOT inflight so we can resubmit or flag it.
            if not jid:
                continue
            if (not crosscheck_squeue) or (str(jid) in q):
                n += 1
    return n


def _ready_to_submit_count(out_root: Path) -> int:
    """
    Count jobs that look eligible:
      prepared=True, submitted=False, completed=False
    """
    n = 0
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        st = _read_state(_state_path(d))
        if not st:
            continue
        if st.get("prepared", False) and (not st.get("submitted", False)) and (not st.get("completed", False)):
            n += 1
    return n


def _normalize_slurm_state(state: Optional[str]) -> str:
    if not state:
        return ""
    return state.strip().upper().split()[0].split("+")[0]


def _job_log_candidates(jobdir: Path, jobid: Optional[object]) -> List[Path]:
    candidates: List[Path] = []
    names = [
        "vasp_stdout.txt",
        "stderr.txt",
        "stdout.txt",
        "job.err",
        "job.out",
    ]
    for name in names:
        p = jobdir / name
        if p.exists():
            candidates.append(p)

    for pattern in ("slurm-*.out", "*.err"):
        candidates.extend(sorted(jobdir.glob(pattern)))

    if jobid:
        candidates.extend(sorted(jobdir.glob(f"*{jobid}*.out")))
        candidates.extend(sorted(jobdir.glob(f"*{jobid}*.err")))

    seen: Set[Path] = set()
    unique: List[Path] = []
    for p in candidates:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _slurm_failure_from_logs(jobdir: Path, st: Dict) -> Optional[str]:
    for p in _job_log_candidates(jobdir, st.get("slurm_job_id")):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if not txt:
            continue
        tail = txt[-200000:]
        for label, pattern in _SLURM_FAILURE_PATTERNS:
            if pattern.search(tail):
                return f"slurm_log_{label}:{p.name}"
    return None


def _slurm_resubmit_reason(jobdir: Path, st: Dict) -> Optional[str]:
    """
    Return a reason string when a job is safe to resubmit as a Slurm failure.

    Successful VASP jobs and VASP/output/convergence failures are intentionally
    not eligible unless Slurm/srun logs show infrastructure trouble. This catches
    batch scripts that end as COMPLETED 0:0 even though the VASP step died.
    """
    if not st.get("submitted"):
        return None
    if not st.get("prepared", False):
        return None
    if st.get("submitted") and not st.get("completed", False):
        return None
    if not st.get("failed"):
        return None
    if st.get("vasp_ok") is True:
        return None

    reason = str(st.get("vasp_reason") or "")
    state = _normalize_slurm_state(st.get("slurm_state"))

    if reason.startswith("slurm_") and state and state != "COMPLETED":
        return reason

    if reason in _LOG_RESUBMITTABLE_VASP_REASONS:
        return _slurm_failure_from_logs(jobdir, st)

    return None


def _eligible_resubmit_count(out_root: Path) -> int:
    n = 0
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        st = _read_state(_state_path(d))
        if _slurm_resubmit_reason(d, st) is not None:
            n += 1
    return n


def _resubmit_skip_reason(jobdir: Path, st: Dict) -> str:
    if not st:
        return "missing_state"
    if not st.get("prepared", False):
        return "not_prepared"
    if st.get("submitted") and not st.get("completed", False):
        return "inflight"
    if st.get("vasp_ok") is True:
        return "completed_ok"
    if not st.get("failed", False):
        return "not_failed"

    reason = str(st.get("vasp_reason") or "")
    state = _normalize_slurm_state(st.get("slurm_state"))
    if not reason.startswith("slurm_"):
        if reason in _LOG_RESUBMITTABLE_VASP_REASONS:
            return f"vasp_failure_no_slurm_log:{reason or 'unknown'}"
        return f"vasp_failure:{reason or 'unknown'}"
    if state == "COMPLETED":
        log_reason = _slurm_failure_from_logs(jobdir, st)
        if log_reason:
            return f"unexpected_skip:{log_reason}"
        return "completed_slurm_state"
    if not state:
        return "missing_slurm_state"
    return f"not_resubmittable:{reason}"


def _set_sbatch_directive(lines: List[str], key: str, value: object) -> List[str]:
    flag = f"--{key}"
    rendered = f"#SBATCH {flag}={value}"
    pattern = re.compile(rf"^(\s*#SBATCH\s+){re.escape(flag)}(?:\s+|=).*$")

    out: List[str] = []
    replaced = False
    insert_at: Optional[int] = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#SBATCH"):
            insert_at = i + 1
        if pattern.match(line):
            if not replaced:
                out.append(rendered)
                replaced = True
            continue
        out.append(line)

    if not replaced:
        if insert_at is None:
            insert_at = 1 if out and out[0].startswith("#!") else 0
        out.insert(insert_at, rendered)

    return out


def _update_sbatch_resources(
    script_path: Path,
    *,
    partition: Optional[str] = None,
    nodes: Optional[int] = None,
    ntasks: Optional[int] = None,
    exclude: Optional[str] = None,
) -> bool:
    updates = {
        "partition": partition,
        "nodes": nodes,
        "ntasks": ntasks,
        "exclude": exclude,
    }
    active = {k: v for k, v in updates.items() if v is not None}
    if not active:
        return False

    original = script_path.read_text()
    lines = original.splitlines()
    trailing_newline = original.endswith("\n")

    for key, value in active.items():
        lines = _set_sbatch_directive(lines, key, value)

    new_txt = "\n".join(lines)
    if trailing_newline or not new_txt:
        new_txt += "\n"

    if new_txt == original:
        return False

    script_path.write_text(new_txt)
    script_path.chmod(0o755)
    return True

def resubmit_jobs(
    *,
    out_root: str,
    max_inflight: int = 100,
    limit: Optional[int] = None,
    partition: Optional[str] = None,
    nodes: Optional[int] = None,
    ntasks: Optional[int] = None,
    exclude: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = True,
    crosscheck_squeue: bool = True,
    refresh_status: bool = True,
) -> ResubmitReport:
    """
    Resubmit failed VASP jobs only when the recorded failure is a Slurm issue.

    VASP/output/convergence failures and successful COMPLETED jobs are skipped.
    Optional Slurm resource overrides edit each eligible job's sub_vasp.sh just
    before sbatch is called.
    """
    out_root_p = Path(out_root).resolve()
    if not out_root_p.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_p}")

    status_error: Optional[str] = None
    if refresh_status and status_update is not None:
        try:
            status_update(out_root=str(out_root_p))
        except Exception as e:
            status_error = str(e)
            if verbose:
                print(f"[PotAudit] status_update failed before resubmit (continuing): {e}")

    inflight = _inflight_count(out_root_p, crosscheck_squeue=crosscheck_squeue)
    capacity = max(0, int(max_inflight) - inflight)
    if limit is not None:
        capacity = min(capacity, max(0, int(limit)))

    eligible = _eligible_resubmit_count(out_root_p)
    resubmitted: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for jobdir in sorted(out_root_p.iterdir()):
        if not jobdir.is_dir():
            continue

        st_path = _state_path(jobdir)
        st = _read_state(st_path)
        reason = _slurm_resubmit_reason(jobdir, st)
        if reason is None:
            skipped.append((jobdir.name, _resubmit_skip_reason(jobdir, st)))
            continue

        if capacity == 0:
            skipped.append((jobdir.name, "capacity"))
            continue

        sub_sh = jobdir / "sub_vasp.sh"
        if not sub_sh.exists():
            skipped.append((jobdir.name, "missing_sub_vasp.sh"))
            continue

        if dry_run:
            resubmitted.append(jobdir.name)
            capacity -= 1
            inflight += 1
            continue

        resources_changed = _update_sbatch_resources(
            sub_sh,
            partition=partition,
            nodes=nodes,
            ntasks=ntasks,
            exclude=exclude,
        )

        previous_attempt = {
            "slurm_job_id": st.get("slurm_job_id"),
            "slurm_state": st.get("slurm_state"),
            "slurm_exit_code": st.get("slurm_exit_code"),
            "vasp_reason": st.get("vasp_reason"),
            "submitted_at": st.get("submitted_at"),
            "completed_at": st.get("completed_at"),
            "checked_at": st.get("checked_at"),
        }

        out = _run(["sbatch", "sub_vasp.sh"], cwd=jobdir)
        jid = _parse_sbatch_jobid(out)

        history = list(st.get("resubmit_history") or [])
        history.append(previous_attempt)
        st["resubmit_history"] = history
        st["resubmit_count"] = int(st.get("resubmit_count") or 0) + 1
        st["last_resubmit_reason"] = reason
        st["resources_changed_on_resubmit"] = resources_changed

        st["submitted"] = True
        st["completed"] = False
        st["failed"] = False
        st["slurm_job_id"] = jid
        st["slurm_state"] = "SUBMITTED"
        st["slurm_exit_code"] = None
        st["vasp_ok"] = None
        st["vasp_reason"] = None
        st["vasp_energy_toten_ev"] = None
        st["vasp_nions"] = None
        st["submitted_at"] = _utcnow()
        st["resubmitted_at"] = st["submitted_at"]
        st["completed_at"] = None
        st["checked_at"] = _utcnow()
        st["sbatch_stdout"] = out
        _write_state(st_path, st)

        resubmitted.append(jobdir.name)
        capacity -= 1
        inflight += 1

    return ResubmitReport(
        resubmitted=resubmitted,
        skipped=skipped,
        inflight=inflight,
        capacity=capacity,
        eligible=eligible,
        status_error=status_error,
    )


def submit_jobs(
    *,
    out_root: str,
    max_inflight: int = 100,
    watch: bool = False,
    poll_sec: int = 60,
    limit: Optional[int] = None,
    verbose: bool = True,
    crosscheck_squeue: bool = True,
    refresh_status_in_watch: bool = True,
) -> SubmitReport:
    """
    Submit per-frame VASP jobs under out_root/<jobdir>/sub_vasp.sh

    Key improvements vs your version:
      - inflight is computed with optional squeue cross-check (avoids stale state.json inflight).
      - watch mode DOES NOT exit just because "submitted=0" once (it may be at capacity).
      - prints which job folders were submitted in that pass.
      - reports remaining ready-to-submit jobs so you can see if it’s done.
      - (optional) calls status_update() each watch loop to mark completed jobs and free capacity.

    Notes:
      - If you ran submit previously and got partial/stale state, crosscheck_squeue=True is safer.
      - If sbatch succeeds but you want more metadata, you can also store submit stdout in state.json.
    """
    out_root_p = Path(out_root).resolve()
    if not out_root_p.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_p}")

    def one_pass() -> SubmitReport:
        inflight = _inflight_count(out_root_p, crosscheck_squeue=crosscheck_squeue)
        capacity = max(0, int(max_inflight) - inflight)
        if limit is not None:
            capacity = min(capacity, max(0, int(limit)))

        submitted: List[str] = []
        skipped: List[str] = []

        # quick ready count before submitting
        remaining_ready_before = _ready_to_submit_count(out_root_p)

        if capacity == 0:
            rep = SubmitReport(
                submitted=submitted,
                skipped=skipped,
                inflight=inflight,
                capacity=capacity,
                remaining_ready=remaining_ready_before,
            )
            return rep

        for jobdir in sorted(out_root_p.iterdir()):
            if capacity == 0:
                break
            if not jobdir.is_dir():
                continue

            st_path = _state_path(jobdir)
            st = _read_state(st_path)

            # must be prepared
            if not st.get("prepared", False):
                skipped.append(jobdir.name)
                continue

            # already done or already submitted
            if st.get("completed", False) or st.get("submitted", False):
                skipped.append(jobdir.name)
                continue

            sub_sh = jobdir / "sub_vasp.sh"
            if not sub_sh.exists():
                raise FileNotFoundError(f"Missing sub_vasp.sh in {jobdir}")

            # submit
            out = _run(["sbatch", "sub_vasp.sh"], cwd=jobdir)
            jid = _parse_sbatch_jobid(out)

            st["submitted"] = True
            st["slurm_job_id"] = jid
            st["slurm_state"] = "SUBMITTED"
            st["submitted_at"] = _utcnow()
            st["checked_at"] = _utcnow()
            st["sbatch_stdout"] = out
            _write_state(st_path, st)

            submitted.append(jobdir.name)
            capacity -= 1
            inflight += 1

        remaining_ready_after = _ready_to_submit_count(out_root_p)

        return SubmitReport(
            submitted=submitted,
            skipped=skipped,
            inflight=inflight,
            capacity=capacity,
            remaining_ready=remaining_ready_after,
        )

    if not watch:
        rep = one_pass()
        if verbose:
            if rep.submitted:
                print(f"[PotAudit] submitted {len(rep.submitted)}: {', '.join(rep.submitted[:10])}"
                      + (" ..." if len(rep.submitted) > 10 else ""))
            print(f"[PotAudit] inflight={rep.inflight} remaining_ready={rep.remaining_ready}")
        return rep

    # Watch mode: keep topping up until no more work remains.
    # We do NOT exit just because one pass submits 0; could be at capacity.
    while True:
        if refresh_status_in_watch and status_update is not None:
            try:
                srep = status_update(out_root=str(out_root_p))
                if verbose and srep.updated:
                    print(f"[PotAudit] status: updated={srep.updated} ok={srep.ok} bad={srep.bad} "
                          f"running={srep.running} pending={srep.pending}")
            except Exception as e:
                if verbose:
                    print(f"[PotAudit] status_update failed (continuing): {e}")

        rep = one_pass()

        if verbose:
            if rep.submitted:
                print(f"[PotAudit] submitted {len(rep.submitted)}: {', '.join(rep.submitted)}")
            print(f"[PotAudit] inflight={rep.inflight} remaining_ready={rep.remaining_ready}")

        # Done condition: nothing left ready to submit AND no inflight jobs.
        if rep.remaining_ready == 0 and rep.inflight == 0:
            return rep

        # If no capacity (or no eligible jobs right now), just wait and re-check.
        time.sleep(max(5, int(poll_sec)))
