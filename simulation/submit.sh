#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=wcsim_wcte_AmBe
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:30:00
#SBATCH --mem=24000
#SBATCH --array=0-99
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((100000 + TASK_ID))

MACDIR=macs
OUTDIR=output

mkdir -p logs "${MACDIR}" "${OUTDIR}"

OUTFILE=$(printf "%s/wcte_ambe_%03d.root" "${OUTDIR}" "${TASK_ID}")
MACFILE=$(printf "%s/wcte_ambe_job_%03d.mac" "${MACDIR}" "${TASK_ID}")

sed \
  -e "s/SEED_PLACEHOLDER/${SEED}/" \
  -e "s|OUTFILE_PLACEHOLDER|${OUTFILE}|" \
  macros/wcte_ambe_template.mac > "${MACFILE}"

srun WCSim "${MACFILE}" macros/tuning_parameters.mac