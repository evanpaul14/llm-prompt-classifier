#!/usr/bin/env bash
# Runs Model 2 then Model 3 training in a tmux session.
# After each run, uses `claude -p` to update binary_results.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="bert_training"
LOG2="$ROOT/logs/model2_train.log"
LOG3="$ROOT/logs/model3_train.log"

mkdir -p "$ROOT/logs"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -x 220 -y 50

tmux send-keys -t "$SESSION" "
set -euo pipefail
cd '$ROOT'
source venv/bin/activate

echo ''
echo '========================================'
echo ' Model 2: Frozen RoBERTa + Head'
echo '========================================'
python model_2_frozen_bert/train.py --max-safe 50000 2>&1 | tee '$LOG2'

echo ''
echo '--- Updating binary_results.md for Model 2 ---'
claude -p \"
The file '$LOG2' contains the full training output for Model 2 (Frozen RoBERTa + classification head, --max-safe 50000).
The file '$ROOT/binary_results.md' tracks results for all models.
Please:
1. Read both files.
2. Fill in the Model 2 section in binary_results.md with the actual results from the log (CV metrics per fold, held-out test metrics, config details), matching the format used for Model 1 and Model 4.
3. Update the Summary table row for Model 2.
\"

echo ''
echo '========================================'
echo ' Model 3: Full RoBERTa Fine-Tune'
echo '========================================'
python model_3_roberta_finetune/train.py --max-safe 50000 2>&1 | tee '$LOG3'

echo ''
echo '--- Updating binary_results.md for Model 3 ---'
claude -p \"
The file '$LOG3' contains the full training output for Model 3 (Full RoBERTa fine-tune, --max-safe 50000).
The file '$ROOT/binary_results.md' tracks results for all models.
Please:
1. Read both files.
2. Fill in the Model 3 section in binary_results.md with the actual results from the log (CV metrics per fold, held-out test metrics, config details), matching the format used for the other models.
3. Update the Summary table row for Model 3.
\"

echo ''
echo '========================================'
echo ' All done. Check binary_results.md.'
echo '========================================'
" Enter

echo "Tmux session '$SESSION' started."
echo "Attach with:  tmux attach -t $SESSION"
echo "Logs:         $LOG2"
echo "              $LOG3"
