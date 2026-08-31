#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=wcsim_wcte_AmBe
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=24000
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((100000 + ${SLURM_JOB_ID:-0} * 1000 + TASK_ID))

MACDIR="${SCRIPT_DIR}/macs"
OUTDIR="${SCRIPT_DIR}/output/2390"

mkdir -p "${SCRIPT_DIR}/logs" "${MACDIR}" "${OUTDIR}"

OUTFILE=$(printf "%s/wcte_ambe_%03d.root" "${OUTDIR}" "${TASK_ID}")
MACFILE=$(printf "%s/wcte_ambe_job_%03d.mac" "${MACDIR}" "${TASK_ID}")

sed \
  -e "s/SEED_PLACEHOLDER/${SEED}/" \
  -e "s|OUTFILE_PLACEHOLDER|${OUTFILE}|" \
  "${SCRIPT_DIR}/macros/wcte_ambe_template.mac" > "${MACFILE}"

srun WCSim "${MACFILE}" "${SCRIPT_DIR}/macros/tuning_parameters.mac"
