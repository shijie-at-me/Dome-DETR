#!/usr/bin/env bash
# Card A queue (2026-09-17): our method with obj=1, then the L AI-TOD baseline to hold the card.
#   bash scripts/queue_ours_a.sh <gpu>            # inside tmux; e.g. bash scripts/queue_ours_a.sh 0
#
# Trains, one after another:
#   abl_aitod_ours       fine level + dynamic query + repsep stride-4 fusion + objectness encoder
#                        target (enc_losses [obj, boxes]); 500 queries, 160 epochs (~12 h)
#   dfine_l_aitod seed0  the L AI-TOD baseline, 500 queries: the main table's L row, and it holds
#                        the card afterwards (the L run the early holder was, restarted from scratch)
# Card B runs abl_aitod_ours_vfl (the same without obj=1) and the L baseline's seed 1 (queue_ours_b.sh).
# The pair ours / ours-vfl isolates obj=1. A failed run does not stop the queue.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${1:?usage: bash scripts/queue_ours_a.sh <gpu>}
export REPORT=1
export NOTIFY_URL=${NOTIFY_URL:-https://ntfy.sh/dfine-saturn}

bash scripts/experiments.sh abl_aitod_ours
SEED=0 bash scripts/experiments.sh dfine_l_aitod
echo "ours-a queue done"
