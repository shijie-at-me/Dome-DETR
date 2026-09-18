#!/usr/bin/env bash
# The queue of card B (2026-09-15, ladder v2). Start it on a free card:
#   bash scripts/queue_card_b.sh <gpu>            # inside tmux; e.g. bash scripts/queue_card_b.sh 1
#
# Trains, one after another:
#   abl_aitod_1b_fusion_light   ladder row 1b: the baseline with the stride-4 fusion block light,
#                               500 queries, 160 epochs; against row 1 it prices the ELAN (~1 day)
#   dfine_l_aitod (seed 1)      the L AI-TOD baseline's second seed, 500 queries: holds the card
#                               (~36 h) once the ablation row is done
# Card A runs row 1 and the L baseline's seed 0 (queue_card_a.sh). The recipe is 160 epochs; a run
# that is still climbing is continued to 200 on its own afterwards (-r last.pth -u epoches=200).
# A failed run does not stop the queue. NOTIFY_URL defaults to the dfine-saturn topic.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${1:?usage: bash scripts/queue_card_b.sh <gpu>}
export REPORT=1
export NOTIFY_URL=${NOTIFY_URL:-https://ntfy.sh/dfine-saturn}

bash scripts/experiments.sh abl_aitod_1b_fusion_light
SEED=1 bash scripts/experiments.sh dfine_l_aitod
echo "card B queue done"
