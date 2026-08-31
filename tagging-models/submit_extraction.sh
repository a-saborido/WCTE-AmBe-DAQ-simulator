#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=wcte_ambe_ntag_extract
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=24000
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

print_finish_time() {
  local exit_code=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Finished submit_extraction.sh with exit code ${exit_code}"
}

trap print_finish_time EXIT

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="${SCRIPT_DIR}"
LOGDIR="${REPO_DIR}/logs"
CONDA_SH=/scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
CONDA_ENV=/scratch/saborido/conda-env/caverns

OUTDIR="${REPO_DIR}/outputs/ntag_bdt_out_TRAIN_combined"

mkdir -p "${LOGDIR}" "${OUTDIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
source /home/saborido/setup.sh
set -u

cd "${REPO_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting submit_extraction.sh"
echo "Using input samples from candidates_extraction/config.py"

srun python -u wcte_ambe_neutron_bdt.py extract \
  --outdir "${OUTDIR}" \
  --geometry-file data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt
