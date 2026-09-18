#!/usr/bin/env bash
# Card B queue (2026-09-17): our method with the encoder kept on VFL (obj=1 off, the control for
# the objectness ablation), then the L AI-TOD baseline to hold the card.
#   bash scripts/queue_ours_b.sh <gpu>            # inside tmux; e.g. bash scripts/queue_ours_b.sh 1
#
# Trains, one after another:
#   abl_aitod_ours_vfl   fine level + dynamic query + repsep stride-4 fusion, encoder on VFL (no
#                        enc_losses override); 500 queries, 160 epochs (~12 h). Identical to
#                        abl_aitod_ours except obj=1, so the pair isolates obj=1 alone.
#   dfine_l_aitod seed1  the L AI-TOD baseline, seed 1: the main table's second L seed, holds the card.
# Card A runs abl_aitod_ours (obj=1 on) and the L baseline's seed 0 (queue_ours_a.sh).
# A failed run does not stop the queue.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${1:?usage: bash scripts/queue_ours_b.sh <gpu>}
export REPORT=1
export NOTIFY_URL=${NOTIFY_URL:-https://ntfy.sh/dfine-saturn}

bash scripts/experiments.sh abl_aitod_ours_vfl
SEED=1 bash scripts/experiments.sh dfine_l_aitod
echo "ours-b queue done"
