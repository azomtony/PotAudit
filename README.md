# PotAudit
Foundational model validation using VASP and ASE

## Convert Structure Files

Convert between VASP POSCAR/CONTCAR files and LAMMPS data files:

```bash
potaudit convert --in POSCAR --out lmp
potaudit convert --in lmp --out POSCAR
```

Formats are inferred from names like `POSCAR`, `CONTCAR`, `.vasp`, `.lmp`, and
`.data`. For ambiguous paths, add `--in-format poscar` or `--out-format lmp`.
When reading a LAMMPS data file without reliable masses, provide atom-type order
with `--lammps-species "Si O"` so type 1 maps to Si and type 2 maps to O.

## Prepare VASP Optimizations From Extxyz Files

Create one VASP relaxation job folder per extxyz structure/frame in a directory:

```bash
potaudit prep-vasp-opt-dir \
  --input-dir path/to/extxyz_dir \
  --out-root runs/case001/vasp_opt \
  --templates-dir templates/vasp_sio2_relax \
  --potcar-root /path/to/potpaw_PBE.64
```

By default this reads all frames from each `*.extxyz` file and uses the
relaxation templates in `templates/vasp_relax`. Use `--index 0` or `--index -1`
to prepare only one frame per extxyz file.

For slab relaxations, `prep-vasp-opt-dir` automatically fixes the bottom slab
layer by writing VASP selective-dynamics flags in `POSCAR`. The default layer
detector uses fractional coordinate along the cell `c` direction, so it works
for non-orthogonal slab cells when the slab/vacuum direction is the third cell
vector. Use `--fix-bottom-layers 2` to fix more layers,
`--bottom-layer-tol 0.4` to tune layer clustering, or `--no-fix-bottom` to relax
all atoms.

## Template Sets

Molecule/default single-point templates live in `templates/vasp`. SiO2 slab
single-point templates live in `templates/vasp_sio2`:

```bash
potaudit prep-vasp-sp-dir \
  --input-dir path/to/extxyz_dir \
  --out-root runs/sio2_sp \
  --templates-dir templates/vasp_sio2 \
  --potcar-root /path/to/potpaw_PBE.64
```

`prep-vasp-sp-dir` defaults to `--pattern '*_opt.extxyz'`, so paired
`*_init.extxyz` files are ignored for single-point evaluation. It creates short
job folders like `sp_000000` while storing the original long source filename in
`state.json`.

SiO2 slab relaxation templates live in `templates/vasp_sio2_relax`:

```bash
potaudit prep-vasp-opt-dir \
  --input-dir path/to/extxyz_dir \
  --out-root runs/sio2_relax \
  --templates-dir templates/vasp_sio2_relax \
  --potcar-root /path/to/potpaw_PBE.64
```

## Resubmit Slurm-Failed VASP Jobs

After `potaudit status` has marked terminal jobs, resubmit only failures caused
by Slurm:

```bash
potaudit resubmit \
  --out-root runs/case001/vasp_opt \
  --partition Standard.2.0 \
  --nodes 1 \
  --cores 64 \
  --exclude node001,node002
```

Use `--dry-run` first to see which jobs would be resubmitted. Completed
successful jobs are never resubmitted. Incomplete VASP outputs such as missing
`OUTCAR` files or missing timing footers are treated as retryable runtime
failures. Pass `--strict-slurm-log` to resubmit those incomplete outputs only
when the job logs also show Slurm/srun trouble such as node failure, timeout,
cancellation, OOM, PRTE remote-daemon communication loss, or job-step launch
errors.

## Submit DP-GEN FP Tasks

Submit unfinished DP-GEN first-principles task folders under an `02.fp`
directory:

```bash
dpgen-fp \
  --fp-dir iter.000001/02.fp \
  --partition Standard.2.0 \
  --time 12:00:00 \
  --ntasks-per-node 64
```

Use `--dry-run` to write `submit.slurm` files and preview submissions without
calling `sbatch`. By default the generated script purges modules, loads
`vasp/gnu14.6.5.0`, sets `OMP_NUM_THREADS=1`, and runs
`mpirun -np ${SLURM_NTASKS} vasp_std 1>>fp.log 2>>fp.log`. Set
`POTAUDIT_DPGEN_FP_EXCLUDE` to override the default excluded nodes
`chpc129,chpc098`, or pass `--exclude` directly. The same tool is also available as `potaudit dpgen-fp ...`
when you want to run it through the main PotAudit command.

## Copy DP-GEN FP Outputs

Copy existing FP outputs from an old scratch directory into the matching local
`task.*` folders:

```bash
potaudit dpgen-fp-copy \
  --local-fp iter.000001/02.fp \
  --remote-fp /scratch/old/iter.000001/02.fp \
  --style vasp \
  --search
```

This is a dry run by default. Add `--apply` to copy files, and `--overwrite` to
replace existing local files. `dpgen-fp-sync` is available as an alias.

## Check DP-GEN FP Convergence

Report VASP FP tasks that exited normally but hit the electronic SCF limit
instead of converging:

```bash
potaudit dpgen-fp-converge \
  --fp-dir iter.000001/02.fp \
  --only-problems
```

The checker does not copy or submit anything. It reads `OUTCAR`/`OSZICAR`,
looks for `aborting loop because EDIFF is reached`, and groups task status by
DP-GEN system id:

```text
------task.000-----
task.000.000001 converged steps_completed=17/200 ionic_steps=1
task.000.000002 not converged steps_completed=200/200 ionic_steps=1 reason=scf_hit_NELM_steps=1;no_EDIFF_reached
```

Use `--nelm 200` to override the parsed limit, or `--check-ionic` to also flag
relaxations that did not satisfy `EDIFFG`.
