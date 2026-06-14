#!/usr/bin/env bash
# ============================================================================
# AI-IoU target-variant sweep. You pick the experiment by dataset/size (same
# positional interface as test.sh); config + output paths are derived to MATCH
# test.sh, so after training you evaluate any variant with:
#     bash scripts/test.sh <dataset> <size> <type>
#
#   bash scripts/train_aiiou_sweep.sh <dataset> <size> [type]
#       dataset : aitod | visdrone
#       size    : s | m | l
#       type    : (optional) run ONLY this one variant; omit to run the whole
#                 VARIANTS sweep below.
#
# Examples:
#     bash scripts/train_aiiou_sweep.sh                     # default: full sweep, Dome-S-AITOD, full schedule
#     bash scripts/train_aiiou_sweep.sh aitod s mult        # only the 'mult' variant
#     EPOCHS=80 bash scripts/train_aiiou_sweep.sh visdrone s    # shortened (scaled) ablation instead
#
# Derived paths (case normalized for you), identical to test.sh:
#     config : configs/dome/Dome-<S|M|L>-<AITOD|VisDrone>.yml
#     outdir : output/aiiou-<size>-<dataset>/<type>      (per variant)
#
# Each variant trains a fresh model differing ONLY in the classification soft
# target (DomeCriterion.aiiou_variant); everything else (data/optim/schedule/
# seed) is identical, so any AP difference is attributable to the target.
# Re-running resumes each variant from its last.pth automatically.
#
# Override knobs via env, e.g.:
#     DEVICES=6,7 GPUS=2 EPOCHS=80 bash scripts/train_aiiou_sweep.sh visdrone m
#
# BATCH SIZE: default bs=8 (canonical). Set BS=16 to train at global batch 16
# (doubled lrs, linear scaling rule) for faster sweeps:
#     BS=16 bash scripts/train_aiiou_sweep.sh visdrone m
# This selects configs/dome/Dome-<S|M|L>-<dataset>-bs16.yml (must exist; see
# Dome-M-VisDrone-bs16.yml) and writes to a SEPARATE output/aiiou-<size>-<dataset>-bs16/
# tree so bs8/bs16 checkpoints never collide. Re-validate the winning variant at
# the canonical bs=8 config before finalizing paper numbers.
#
# NOTE on EPOCHS: simply lowering `epoches` would desync the epoch-coupled
# schedule (LR milestones, strong-aug stop). When EPOCHS is set, this script
# reads the chosen config's baseline schedule and scales ALL epoch-coupled
# knobs PROPORTIONALLY (lr_scheduler.milestones, aug policy.epoch, collate
# stop_epoch). Shape is preserved so the variant ranking stays fair; still
# re-run the top variant(s) at full schedule before finalizing.
# (ema.warmups / lr warmup_duration are iteration-based and left unchanged.)
# ============================================================================
# This script lives in scripts/; cd to the repo root (its parent) so all the
# relative paths below (configs/, train.py, output/) resolve correctly.
cd "$(dirname "$0")/.." || exit 1
set -f          # no filename globbing (so list overrides like milestones=[36,60] stay literal)

# --- inputs (positional, mirroring test.sh) ---------------------------------
# Default target: Dome-S on AITOD at the full config schedule.
DATASET=${1:-aitod}               # aitod | visdrone
SIZE=${2:-s}                      # s | m | l
TYPE_ARG=${3:-}                   # optional: single variant name; empty = full sweep

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

# --- batch-size selection ---------------------------------------------------
# BS=8  -> canonical config  Dome-<S|M|L>-<AITOD|VisDrone>.yml       (total_batch_size 8)
# BS=16 -> doubled-batch cfg Dome-<S|M|L>-<AITOD|VisDrone>-bs16.yml  (total_batch_size 16,
#          ALL lrs doubled per the linear scaling rule). That wrapper merely
#          __include__s the bs=8 config, so the epoch-coupled schedule is shared
#          and is still read below from the canonical (bs=8) file.
BS=${BS:-8}
case "$BS" in 8|16) ;; *) echo "ERROR: BS must be 8 or 16 (got '$BS')"; exit 1 ;; esac

# --- auto-derived paths (match test.sh) -------------------------------------
# BASE_CONFIG is always the canonical bs=8 file: it is the source of truth for
# the epoch-coupled schedule (the bs16 wrapper does not redeclare those keys).
BASE_CONFIG=configs/dome/Dome-${size_u}-${dataset_proper}.yml
if [ "$BS" -eq 16 ]; then
  # separate output tree so bs8/bs16 checkpoints never share a last.pth (auto-resume)
  CONFIG=${CONFIG:-configs/dome/Dome-${size_u}-${dataset_proper}-bs16.yml}
  OUT_ROOT=${OUT_ROOT:-output/aiiou-${size_l}-${dataset_l}-bs16}
else
  CONFIG=${CONFIG:-$BASE_CONFIG}
  OUT_ROOT=${OUT_ROOT:-output/aiiou-${size_l}-${dataset_l}}
fi

DEVICES=${DEVICES:-0,1}            # CUDA_VISIBLE_DEVICES
GPUS=${GPUS:-2}                    # nproc_per_node (must match #DEVICES)
PORT=${PORT:-7790}
SEED=${SEED:-0}                    # fixed across variants for fair comparison
EPOCHS=${EPOCHS:-}                 # empty = full config schedule; set e.g. 80 to shorten (scaled)

[ -f "$BASE_CONFIG" ] || { echo "ERROR: base config not found: $BASE_CONFIG"; exit 1; }
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG"
  [ "$BS" -eq 16 ] && echo "  (BS=16 needs a bs16 wrapper config; create one like configs/dome/Dome-M-VisDrone-bs16.yml:" && \
    echo "   __include__ the bs=8 file, set train/val total_batch_size to 16/32, and double all lrs.)"
  exit 1
fi

# "name|<DomeCriterion overrides, space-separated, WITHOUT the DomeCriterion. prefix>"
# Comment/uncomment rows to choose the sweep. baseline + mult + additive first.
VARIANTS=(
  # --- first wave: baseline + the two oracle-Pareto endpoints -------------
  "baseline|aiiou_variant=none"
  "additive_B0.2|aiiou_variant=additive aiiou_B=0.2"               # dual-win: AP@.5 up AND mAP up
  "obj_sref32_g1|aiiou_variant=obj aiiou_s_ref=32 aiiou_gamma=1"   # max objectness / AP@.5
  # --- second wave (uncomment as needed) ----------------------------------
  # "additive_B0.3|aiiou_variant=additive aiiou_B=0.3"
  # "obj_sref32_g2|aiiou_variant=obj aiiou_s_ref=32 aiiou_gamma=2"
  # "obj_sref16_g2|aiiou_variant=obj aiiou_s_ref=16 aiiou_gamma=2" # mAP-safe (only <16px)
  # "mult|aiiou_variant=mult"
  # "smooth|aiiou_variant=smooth"
)

# --- select variants: single (3rd arg) or full sweep -----------------------
SELECTED=()
if [ -n "$TYPE_ARG" ]; then
  for row in "${VARIANTS[@]}"; do
    [ "${row%%|*}" = "$TYPE_ARG" ] && SELECTED=("$row")
  done
  if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "ERROR: type '$TYPE_ARG' not found in VARIANTS. Available:"
    for row in "${VARIANTS[@]}"; do echo "    ${row%%|*}"; done
    exit 1
  fi
else
  SELECTED=("${VARIANTS[@]}")
fi

# start_eval: skip validation before this epoch (defined on the FULL-schedule
# scale, e.g. 90 of 160). Big AITOD speedup since early stage-1 eval is wasted:
# stage-1 AP is ~monotonic, so best_stg1 == the last evaluated stage-1 epoch.
# 90 still leaves ~30 epochs of stage-1 eval margin below stop_epoch (108) to
# robustly capture best_stg1. Scales with EPOCHS and is clamped below stop_epoch
# so best_stg1 is still produced and all of stage-2 is evaluated. 0 = off.
START_EVAL=${START_EVAL:-78}

# --- read baseline schedule FROM THE CONFIG (needed for scaling / clamping) --
SCHED_OV=""
if [ -n "$EPOCHS" ] || [ "${START_EVAL:-0}" -gt 0 ]; then
  # always read the schedule from BASE_CONFIG: the bs16 wrapper only redeclares
  # batch sizes + lrs and inherits epoches/milestones/stop_epoch via __include__.
  base_ep=$(grep -E '^epoches:' "$BASE_CONFIG" | grep -oE '[0-9]+' | head -1)
  base_m1=$(grep -E 'milestones:' "$BASE_CONFIG" | grep -oE '[0-9]+' | sed -n 1p)
  base_m2=$(grep -E 'milestones:' "$BASE_CONFIG" | grep -oE '[0-9]+' | sed -n 2p)
  base_stop=$(grep -E 'stop_epoch:' "$BASE_CONFIG" | grep -oE '[0-9]+' | head -1)
  base_aug=$(awk '/policy:/{f=1} f&&/epoch:/{print $2; exit}' "$BASE_CONFIG")
  : "${base_aug:=$base_stop}"   # fall back to stop_epoch if policy.epoch absent
  if [ -z "$base_ep" ] || [ -z "$base_m1" ] || [ -z "$base_m2" ] || [ -z "$base_stop" ]; then
    echo "ERROR: could not parse baseline schedule from $BASE_CONFIG (epoches/milestones/stop_epoch)"; exit 1
  fi
fi

# proportional schedule scaling when EPOCHS is set
if [ -n "$EPOCHS" ]; then
  sc () { awk -v a="$1" -v e="$EPOCHS" -v b="$base_ep" 'BEGIN{printf "%d", int(a*e/b + 0.5)}'; }
  M1=$(sc "$base_m1"); M2=$(sc "$base_m2"); STOP=$(sc "$base_stop"); AUG=$(sc "$base_aug")
  SCHED_OV=" epoches=${EPOCHS} lr_scheduler.milestones=[${M1},${M2}]"
  SCHED_OV="${SCHED_OV} train_dataloader.dataset.transforms.policy.epoch=${AUG}"
  SCHED_OV="${SCHED_OV} train_dataloader.collate_fn.stop_epoch=${STOP}"
  echo "[sched] EPOCHS=${EPOCHS} scaled from config baseline ${base_ep}:"
  echo "        milestones [${base_m1},${base_m2}] -> [${M1},${M2}];" \
       "aug_stop ${base_aug} -> ${AUG}; collate stop_epoch ${base_stop} -> ${STOP}"
  eff_stop=$STOP
else
  eff_stop=$base_stop
fi

# start_eval override (scaled + clamped below the effective stop_epoch)
if [ "${START_EVAL:-0}" -gt 0 ]; then
  se=$START_EVAL
  [ -n "$EPOCHS" ] && se=$(awk -v a="$START_EVAL" -v e="$EPOCHS" -v b="$base_ep" 'BEGIN{printf "%d", int(a*e/b + 0.5)}')
  if [ "$se" -ge "$eff_stop" ]; then se=$((eff_stop - 1)); fi
  [ "$se" -lt 0 ] && se=0
  SCHED_OV="${SCHED_OV} start_eval=${se}"
  echo "[eval ] start_eval=${se} (skip eval before epoch ${se}; stage-2 from ${eff_stop} always evaluated)"
fi

run_one () {
  local name="$1"; local overrides="$2"
  local outdir="${OUT_ROOT}/${name}"
  mkdir -p "$outdir"

  # build -u args: DomeCriterion-prefixed variant overrides + (unprefixed) schedule overrides
  local u_args=""
  for kv in $overrides; do u_args="${u_args} DomeCriterion.${kv}"; done
  u_args="${u_args}${SCHED_OV}"

  # auto-resume if a checkpoint exists
  local resume_arg=""
  [ -f "${outdir}/last.pth" ] && resume_arg="-r ${outdir}/last.pth"

  echo "============================================================"
  echo "[$(date '+%F %T')] ${dataset_l}/${size_u} bs=${BS} variant=${name}"
  echo "  config:    ${CONFIG}"
  echo "  overrides: ${u_args}${resume_arg:+   (resume)}"
  echo "  outdir:    ${outdir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${DEVICES} torchrun --master_port=${PORT} --nproc_per_node=${GPUS} \
      train.py -c "${CONFIG}" --use-amp --seed=${SEED} --output-dir "${outdir}" \
      ${resume_arg} -u ${u_args} \
      2>&1 | tee -a "${outdir}/train.log"

  local status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "!! variant ${name} FAILED (exit ${status}); continuing to next." | tee -a "${outdir}/train.log"
  fi
}

for row in "${SELECTED[@]}"; do
  name="${row%%|*}"; ov="${row#*|}"
  run_one "$name" "$ov"
done

echo
echo "ALL VARIANTS DONE. Per-variant logs/checkpoints under ${OUT_ROOT}/<name>/."
echo "Evaluate any variant with:  bash scripts/test.sh ${dataset_l} ${size_l} <name>"
echo "Collect final AP per variant, e.g.:"
echo "    grep -hE 'Average Precision.*area=   all|aiiou|Epoch' ${OUT_ROOT}/*/train.log | tail"
