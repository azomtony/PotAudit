# PotAudit
Foundational model validation using VASP and ASE

## Prepare VASP Optimizations From Extxyz Files

Create one VASP relaxation job folder per extxyz structure/frame in a directory:

```bash
potaudit prep-vasp-opt-dir \
  --input-dir path/to/extxyz_dir \
  --out-root runs/case001/vasp_opt \
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
