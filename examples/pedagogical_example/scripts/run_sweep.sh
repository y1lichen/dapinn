#!/usr/bin/env bash
set -euo pipefail

# Simple sweep script to compare Standard PINN vs DAPINN across sample sizes
# Usage: ./run_sweep.sh

WORKDIR=$(pwd)
CONFIG="examples/pedagogical_example/configs/default.py"

# Ensure repo root is on PYTHONPATH so `import examples...` works
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

# Measurements to test (customize as needed)
SIZES=(10 25 50 100 200)

SEED=42
NOISE=0.0

for N in "${SIZES[@]}"; do
  # Standard PINN (no corrector)
  SUBDIR="results/pedagogical/measurements/pedagogical_sample${N}_PINN"
  echo "Running Standard PINN sample_size=${N} -> ${SUBDIR}"
  python3 -m examples.pedagogical_example.main \
    --config=${CONFIG} \
    --mode=train \
    --workdir="${WORKDIR}" \
    --save_subdir="${SUBDIR}" \
    --sample_size=${N} \
    --use_corrector=False \
    --run_pretrain=False \
    --run_finetune=True \
    --seed=${SEED} \
    --noise=${NOISE}

  # DAPINN: pretrain + finetune (enable pretraining so pretrained weights are produced)
  SUBDIR="results/pedagogical/measurements/pedagogical_sample${N}_DAPINN"
  echo "Running DAPINN sample_size=${N} -> ${SUBDIR}"
  python3 -m examples.pedagogical_example.main \
    --config=${CONFIG} \
    --mode=train \
    --workdir="${WORKDIR}" \
    --save_subdir="${SUBDIR}" \
    --sample_size=${N} \
    --use_corrector=True \
    --run_pretrain=True \
    --run_finetune=True \
    --load_pretrained=True \
    --seed=${SEED} \
    --noise=${NOISE}
done

echo "All runs finished. Results stored under ./results/..."

# --- Auto-collect and plot ---
CSV_PATH="results/pedagogical/measurements/pedagogical_summary.csv"
PLOTS_DIR="results/pedagogical/measurements/plots"

echo "Collecting results into ${CSV_PATH} ..."
python3 examples/pedagogical_example/utils/collect_measurements_results.py \
  --workdir "${WORKDIR}" \
  --pattern "results/pedagogical/measurements/pedagogical_sample*_*" \
  --out "${CSV_PATH}"

echo "Plotting comparisons into ${PLOTS_DIR} ..."
python3 examples/pedagogical_example/utils/plot_measurements_comparison.py \
  --csv "${CSV_PATH}" \
  --outdir "${PLOTS_DIR}"

echo "Collect + Plot finished. Summary: ${CSV_PATH}, plots at ${PLOTS_DIR}"
