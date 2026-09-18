#!/usr/bin/env bash
# The training runs of the paper, by name, one after another in the order given.
#
#   bash scripts/experiments.sh --list                      # the names and their configs
#   bash scripts/experiments.sh dfine_s_visdrone ours_s_aitod
#   bash scripts/experiments.sh baselines                   # every D-FINE baseline (S, M, L x VisDrone, AI-TOD)
#   bash scripts/experiments.sh ours                        # our method (S, M, L x VisDrone, AI-TOD)
#   bash scripts/experiments.sh all                         # baselines, then ours
#   bash scripts/experiments.sh ablation_aitod              # the AI-TOD ablation rows 2 to 5 (configs/dome/ablation)
#   bash scripts/experiments.sh stride_study                # the fine level with and without its depthwise blocks
#   bash scripts/experiments.sh --dry-run all               # print the commands only
#
# Environment:
#   GPUS=2      torchrun on that many GPUs (default 1: plain python; CUDA_VISIBLE_DEVICES picks the card)
#   SEED=0      the seed of every run
#   EXTRA=...   more train.py arguments for every run, e.g. EXTRA="-u train_dataloader.num_workers=4"
#   REPORT=1    after each run, write its RESULTS.md and curves.png with tools/analysis/run_report.py
#   NOTIFY_URL=https://ntfy.sh/<topic>   POST a line there when a run finishes or fails and when the
#               queue ends (ntfy: subscribe to the topic in the phone app; any URL taking a POST body works)
#
# Each run writes into its config's output_dir/<date>_<time> (train.py does that) and its console
# into logs/<name>-<date>_<time>.log. A failing run stops the sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
# python's stdout is block-buffered into the tee pipe below; without this the logs/ copy lags by pages
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ the registry
declare -A CONFIGS=(
    [dfine_s_visdrone]=configs/dome/DFine-S-VisDrone.yml
    [dfine_m_visdrone]=configs/dome/DFine-M-VisDrone.yml
    [dfine_l_visdrone]=configs/dome/DFine-L-VisDrone.yml
    [dfine_s_aitod]=configs/dome/DFine-S-AITOD.yml
    [dfine_m_aitod]=configs/dome/DFine-M-AITOD.yml
    [dfine_l_aitod]=configs/dome/DFine-L-AITOD.yml
    [ours_s_visdrone]=configs/dome/DFine-S-VisDrone-Ours.yml
    [ours_m_visdrone]=configs/dome/DFine-M-VisDrone-Ours.yml
    [ours_l_visdrone]=configs/dome/DFine-L-VisDrone-Ours.yml
    [ours_s_aitod]=configs/dome/DFine-S-AITOD-Ours.yml
    [ours_m_aitod]=configs/dome/DFine-M-AITOD-Ours.yml
    [ours_l_aitod]=configs/dome/DFine-L-AITOD-Ours.yml
    # the AI-TOD ablation, one switch per row on the S baseline; rows 1 and 6 are dfine_s_aitod and ours_s_aitod
    [abl_aitod_2_budget]=configs/dome/ablation/DFine-S-AITOD-2-budget.yml
    [abl_aitod_3_enc_quality]=configs/dome/ablation/DFine-S-AITOD-3-enc-quality.yml
    [abl_aitod_4_fine]=configs/dome/ablation/DFine-S-AITOD-4-fine.yml
    [abl_aitod_5_min_cells]=configs/dome/ablation/DFine-S-AITOD-5-min-cells.yml
    [abl_aitod_6_fine_key]=configs/dome/ablation/DFine-S-AITOD-6-fine-key.yml
    # the stride study: the fine level with and without its two depthwise blocks, both at the
    # baseline's fixed 300 queries, which settles fine_blocks for row 3
    [abl_aitod_2_fine]=configs/dome/ablation/DFine-S-AITOD-2-fine.yml
    [abl_aitod_2b_fine_light]=configs/dome/ablation/DFine-S-AITOD-2b-fine-light.yml
    # ladder v2 (2026-09-15): row 1b, the baseline with the stride-4 fusion block light, prices the ELAN alone
    [abl_aitod_1b_fusion_light]=configs/dome/ablation/DFine-S-AITOD-1b-fusion-light.yml
    # fine-level design: the fine map widened to 256 and split 8 ways, one group per head, on the ELAN baseline
    [abl_aitod_fine_grouped]=configs/dome/ablation/DFine-S-AITOD-fine-grouped.yml
    # fine-level design rows (2026-09-16): fine+key (validated levers), and +min_cells as its own ablation
    [abl_aitod_fine_key]=configs/dome/ablation/DFine-S-AITOD-fine-key.yml
    [abl_aitod_fine_key_mincells]=configs/dome/ablation/DFine-S-AITOD-fine-key-mincells.yml
    [abl_aitod_fine_dq]=configs/dome/ablation/DFine-S-AITOD-fine-dq.yml
    # our method (2026-09-17): fine + dynamic query + repsep stride-4 fusion + objectness enc target,
    # and the same + Rank & Sort loss (the expected headline row)
    [abl_aitod_ours]=configs/dome/ablation/DFine-S-AITOD-ours.yml
    [abl_aitod_ours_rank]=configs/dome/ablation/DFine-S-AITOD-ours-rank.yml
    # obj=1 isolation: ours (obj) vs ours-vfl (same, encoder kept on VFL); their difference is obj=1 alone
    [abl_aitod_ours_vfl]=configs/dome/ablation/DFine-S-AITOD-ours-vfl.yml
)
BASELINES=(dfine_s_visdrone dfine_m_visdrone dfine_l_visdrone dfine_s_aitod dfine_m_aitod dfine_l_aitod)
OURS=(ours_s_visdrone ours_m_visdrone ours_l_visdrone ours_s_aitod ours_m_aitod ours_l_aitod)
ABLATION_AITOD=(abl_aitod_2_budget abl_aitod_3_enc_quality abl_aitod_4_fine abl_aitod_5_min_cells)
STRIDE_STUDY=(abl_aitod_2_fine abl_aitod_2b_fine_light)

GPUS=${GPUS:-1}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}
REPORT=${REPORT:-0}
NOTIFY_URL=${NOTIFY_URL:-}
DRY_RUN=0

# ------------------------------------------------------------------ functions
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

list_experiments() {
    for name in "${BASELINES[@]}" "${OURS[@]}" "${ABLATION_AITOD[@]}" "${STRIDE_STUDY[@]}"; do printf '  %-18s %s\n' "$name" "${CONFIGS[$name]}"; done
}

# the output_dir of a config, resolved through its includes, so the run's directory can be found afterwards
output_dir_of() {
    python - "$1" <<'PY'
import sys
from src.core.yaml_utils import load_config
print(load_config(sys.argv[1])["output_dir"])
PY
}

# the run directory train.py created: the newest under the config's output_dir
latest_run_of() {
    local out
    out=$(output_dir_of "$1")
    ls -1dt "$out"/*/ 2>/dev/null | head -1 | sed 's#/$##'
}

# train one config: python or torchrun, the console tee'd into logs/
train() {
    local name=$1 config=$2 stamp
    stamp=$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p logs
    local -a cmd
    if [ "$GPUS" -gt 1 ]; then
        cmd=(torchrun --master_port="${MASTER_PORT:-7789}" --nproc_per_node="$GPUS" train.py)
    else
        cmd=(python train.py)
    fi
    # shellcheck disable=SC2206  # EXTRA is meant to split into arguments
    cmd+=(-c "$config" --use-amp --seed "$SEED" $EXTRA)
    echo "== $name: ${cmd[*]}"
    if [ "$DRY_RUN" = 1 ]; then return 0; fi
    "${cmd[@]}" 2>&1 | tee "logs/${name}-${stamp}.log"
}

# one line to NOTIFY_URL, when set; a failure to deliver never stops the queue
notify() {
    [ -n "$NOTIFY_URL" ] || return 0
    curl -fsS -m 20 -H "Title: $(hostname) gpu${CUDA_VISIBLE_DEVICES:-?}" -d "$1" "$NOTIFY_URL" >/dev/null 2>&1 || true
}

# the best AP of a config's newest run, from its RESULTS.md when there is one, else its log.txt
best_ap_of() {
    local run
    run=$(latest_run_of "$1")
    [ -n "$run" ] || return 0
    if [ -f "$run/RESULTS.md" ]; then
        grep -m1 '^| best AP' "$run/RESULTS.md" | sed 's/[|*]//g; s/  */ /g; s/^ //'
    elif [ -f "$run/log.txt" ]; then
        python - "$run/log.txt" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if "test_coco_eval_bbox" in r]
if rows:
    best = max(rows, key=lambda r: r["test_coco_eval_bbox"][0])
    print(f"best AP {100 * best['test_coco_eval_bbox'][0]:.1f} at epoch {best['epoch']} ({len(rows)} evaluated epochs)")
PY
    fi
}

# summarize the newest run of a config into RESULTS.md and curves.png
report() {
    local run
    run=$(latest_run_of "$1")
    if [ -n "$run" ]; then
        echo "== report: $run"
        python tools/analysis/run_report.py "$run"
    fi
}

run_experiment() {
    local name=$1
    local config=${CONFIGS[$name]:-}
    if [ -z "$config" ]; then
        echo "unknown experiment '$name'; the names are:" >&2
        list_experiments >&2
        exit 2
    fi
    if ! train "$name" "$config"; then
        notify "FAILED $name; the queue stops (${names[*]})"
        exit 1
    fi
    if [ "$DRY_RUN" = 0 ]; then
        if [ "$REPORT" = 1 ]; then report "$config"; fi
        notify "finished $name: $(best_ap_of "$config")"
    fi
}

# a name, or a group of names, to the names it stands for
expand() {
    case "$1" in
        all) echo "${BASELINES[@]}" "${OURS[@]}" ;;
        baselines) echo "${BASELINES[@]}" ;;
        ours) echo "${OURS[@]}" ;;
        ablation_aitod) echo "${ABLATION_AITOD[@]}" ;;
        stride_study) echo "${STRIDE_STUDY[@]}" ;;
        *) echo "$1" ;;
    esac
}

# ------------------------------------------------------------------ main
if [ $# -eq 0 ]; then
    usage
    exit 1
fi
names=()
for arg in "$@"; do
    case "$arg" in
        -h | --help)
            usage
            exit 0
            ;;
        --list)
            list_experiments
            exit 0
            ;;
        --dry-run) DRY_RUN=1 ;;
        *)
            read -r -a more <<<"$(expand "$arg")"
            names+=("${more[@]}")
            ;;
    esac
done
if [ ${#names[@]} -eq 0 ]; then
    usage
    exit 1
fi

echo "runs, in order: ${names[*]}  (gpus $GPUS, seed $SEED${EXTRA:+, extra: $EXTRA})"
for name in "${names[@]}"; do run_experiment "$name"; done
echo "done: ${names[*]}"
if [ "$DRY_RUN" = 0 ]; then notify "queue done: ${names[*]}"; fi
