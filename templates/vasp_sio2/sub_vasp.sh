#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=Standard.2.0
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64

set -euo pipefail

module purge
module load vasp/intel.6.5.0

ulimit -s unlimited

export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"

# Intel MPI <-> Slurm PMI
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export I_MPI_HYDRA_BOOTSTRAP=slurm

# Intel MPI fabrics
export I_MPI_FABRICS=shm:ofi

# OFI provider
export FI_PROVIDER=tcp

echo "[VASP] host=$(hostname) ntasks=${SLURM_NTASKS} start=$(date -Is)"

srun --mpi=pmi2 vasp_std > vasp_stdout.txt

echo "[VASP] done=$(date -Is)" | tee vasp_done.txt
