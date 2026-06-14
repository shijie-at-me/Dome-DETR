#!/usr/bin/env bash
# ============================================================================
# Single-model evaluation (--test-only). You pick the experiment by 3 inputs;
# the config / checkpoint / log paths are all derived automatically.
#
#   bash scripts/test.sh <dataset> <size> <type>
#       dataset : aitod | visdrone
#       size    : s | m | l
#       type    : baseline | <custom variant, e.g. mult, additive_B0.2, ...>
#
# Examples:
#     bash scripts/test.sh visdrone m baseline
#     bash scripts/test.sh aitod s mult
#
# Derived paths (size/dataset case is normalized for you):
#     config  : configs/dome/Dome-<S|M|L>-<AITOD|VisDrone>.yml
#     outdir  : output/aiiou-<size>-<dataset>/<type>
#     ckpt    : <outdir>/best_stg2.pth          (override with CKPT=...)
#     evaldir : <outdir>/eval                   (eval-only artifacts live here,
#     log     : <outdir>/eval/test.log           so they don't pollute <outdir>)
#
# Runs single-GPU with plain `python` (no torchrun). All stdout/stderr is shown
# live in the terminal AND tee'd (appended) to the log.
#
# Override knobs via env, e.g.:
#     DEVICES=1 BATCH_SIZE=16 SAVE_TP_FP_ANALYSIS=true bash scripts/test.sh visdrone m baseline
# ============================================================================
# This script lives in scripts/; cd to the repo root (its parent) so all the
# relative paths below (configs/, train.py, output/) resolve correctly.
cd "$(dirname "$0")/.." || exit 1

# --- inputs (positional, with defaults) -------------------------------------
DATASET=${1:-visdrone}            # aitod | visdrone
SIZE=${2:-m}                      # s | m | l
TYPE=${3:-baseline}               # baseline | <custom variant name>

# --- normalize + validate ---------------------------------------------------
dataset_l=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
size_l=$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')
size_u=$(echo "$SIZE" | tr '[:lower:]' '[:upper:]')

case "$dataset_l" in
  aitod)    dataset_proper="AITOD" ;;
  visdrone) dataset_proper="VisDrone" ;;
  *) echo "ERROR: dataset must be 'aitod' or 'visdrone' (got '$DATASET')"; exit 1 ;;
esac
case "$size_u" in
  S|M|L) ;;
  *) echo "ERROR: size must be s, m, or l (got '$SIZE')"; exit 1 ;;
esac

# --- auto-derived paths -----------------------------------------------------
CONFIG="configs/dome/Dome-${size_u}-${dataset_proper}.yml"
OUTDIR="output/aiiou-${size_l}-${dataset_l}/${TYPE}"
EVALDIR="${OUTDIR}/eval"          # keep eval artifacts out of the training outdir
CKPT=${CKPT:-${OUTDIR}/best_stg2.pth}
LOG=${LOG:-${EVALDIR}/test.log}

DEVICES=${DEVICES:-0}             # CUDA_VISIBLE_DEVICES (single card)
BATCH_SIZE=${BATCH_SIZE:-32}      # val batch size (override: BATCH_SIZE=...)
PYTHON=${PYTHON:-C:/Shijie_Li/.venv/Scripts/python.exe}   # interpreter (override: PYTHON=...)
SAVE_TP_FP_ANALYSIS=${SAVE_TP_FP_ANALYSIS:-false}   # true | false: dump TP/FP analysis

# --- sanity checks ----------------------------------------------------------
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[ -f "$CKPT" ]   || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
mkdir -p "$EVALDIR"

# --- run --------------------------------------------------------------------
{
  echo "============================================================"
  echo "[$(date '+%F %T')] test-only"
  echo "  dataset/size/type: ${dataset_l} / ${size_u} / ${TYPE}"
  echo "  config: ${CONFIG}"
  echo "  ckpt:   ${CKPT}"
  echo "  device: ${DEVICES} (single-GPU, no torchrun)"
  echo "  tp_fp_analysis: ${SAVE_TP_FP_ANALYSIS}"
  echo "============================================================"

  # Single GPU: call python directly. train.py's setup_distributed() falls back
  # to non-distributed mode when the torchrun env vars are absent.
  CUDA_VISIBLE_DEVICES=${DEVICES} SAVE_TP_FP_ANALYSIS=${SAVE_TP_FP_ANALYSIS} "${PYTHON}" \
      train.py -c "${CONFIG}" --test-only -r "${CKPT}" --output-dir "${EVALDIR}" \
      -u val_dataloader.total_batch_size=${BATCH_SIZE}
} 2>&1 | tee -a "${LOG}"

echo "Done. Output appended to ${LOG}"
