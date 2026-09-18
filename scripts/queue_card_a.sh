#!/usr/bin/env bash
# The queue of card A (2026-09-15, ladder v2). Start it on a free card:
#   bash scripts/queue_card_a.sh <gpu>            # inside tmux; e.g. bash scripts/queue_card_a.sh 0
#
# Trains, one after another:
#   dfine_s_aitod        ladder row 1: the S AI-TOD baseline, 500 queries, 160 epochs (~1 day)
#   dfine_l_aitod        the L AI-TOD baseline, seed 0, 500 queries: the main table's L row, and it
#                        holds the card (~36 h) once the ablation row is done
# Card B runs row 1b and the L baseline's seed 1 (queue_card_b.sh). The recipe is 160 epochs; a run
# that is still climbing is continued to 200 on its own afterwards (-r last.pth -u epoches=200).
# A failed run does not stop the queue. NOTIFY_URL defaults to the dfine-saturn topic.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${1:?usage: bash scripts/queue_card_a.sh <gpu>}
export REPORT=1
export NOTIFY_URL=${NOTIFY_URL:-https://ntfy.sh/dfine-saturn}

bash scripts/experiments.sh dfine_s_aitod
SEED=0 bash scripts/experiments.sh dfine_l_aitod
echo "card A queue done"
