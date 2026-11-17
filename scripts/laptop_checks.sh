#!/usr/bin/env bash
# Every pending check that needs the laptop (CUDA GPU + its CPU), in one go. Run from the repository root after
# `git pull`, with the virtualenv active:   bash scripts/laptop_checks.sh
# It refuses to start on a dirty tracked tree (lab results would be recorded as "dirty"), never overwrites a result
# (lab run names are fixed: if one exists it is skipped), logs everything (appended to
# research/l2l/experiments/laptop_checks.log at the end),
# and at the end commits only the new result folders and the log. It does not push.
set -u
cd "$(dirname "$0")/.."
FINAL_LOG=research/l2l/experiments/laptop_checks.log
EXP=research/l2l/experiments
LOG=$(mktemp)  # written during the run; appended to FINAL_LOG only at the end, so the tracked tree stays clean while
               # the lab cases record their provenance
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files are modified: commit or stash them first"; git status --short --untracked-files=no; exit 1
fi
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA device: run this on the GPU laptop')" || exit 1
COMMIT=$(git rev-parse --short HEAD)
echo "== laptop checks at $COMMIT, $(date -u +%FT%TZ)" | tee -a "$LOG"
new=()

lab() {  # lab cpu|gpu <name> <args...>: run a lab case unless its result exists (cpu: the GPU is hidden)
  local where=$1 name=$2; shift 2
  if [ -e "$EXP/$name" ]; then echo "skip $name (exists)" | tee -a "$LOG"; return; fi
  local hide=(); [ "$where" = cpu ] && hide=(env CUDA_VISIBLE_DEVICES=)
  echo "\$ ${hide[*]} python scripts/l2l_lab.py run --name $name $*" | tee -a "$LOG"
  "${hide[@]}" python scripts/l2l_lab.py run --name "$name" "$@" 2>&1 | grep -E "^\*\*Status|Error|error" | tee -a "$LOG"
  [ -e "$EXP/$name" ] && new+=("$EXP/$name")
}

echo "== 1. unit tests: CPU + CUDA variants" | tee -a "$LOG"
python -m pytest tests/test_prefetch.py tests/test_backward.py tests/test_generator.py tests/test_layer_stream.py \
  tests/test_backend.py tests/test_output_head.py -q -p no:cacheprovider 2>&1 | tail -5 | tee -a "$LOG"

echo "== 2. Ticket 05 (CPU only)" | tee -a "$LOG"
lab cpu t05_logprob_equivalence_cpu_v2 --case logprob_equivalence \
  --config research/l2l/configs/store_cpu.yaml --device cpu --hypothesis H03
lab cpu t05_logprob_equivalence_raw_logit_v2 --case logprob_equivalence \
  --config research/l2l/configs/store_cpu.yaml --device cpu --perturb raw_logit

echo "== 3. Ticket 08 correctness: the equivalence cases under l2l_async on the GPU" | tee -a "$LOG"
A=research/l2l/configs/async.yaml
lab gpu t08_forward_equivalence_async --case forward_equivalence --config $A --device cuda --hypothesis H01
lab gpu t08_gradient_equivalence_async --case gradient_equivalence --config $A --device cuda --hypothesis H02
lab gpu t08_generation_equivalence_async --case generation_equivalence --config $A --device cuda --hypothesis H04
lab gpu t08_update_equivalence_async --case update_equivalence --config $A --device cuda --hypothesis H05
lab gpu t08_gradient_equivalence_async_unsafe_reuse --case gradient_equivalence --config $A --device cuda --perturb unsafe_reuse=2
lab gpu t08_forward_equivalence_async_unsafe_reuse --case forward_equivalence --config $A --device cuda --perturb unsafe_reuse=2

echo "== 4. Ticket 08 performance: real Qwen3-0.6B, sync vs async" | tee -a "$LOG"
lab gpu t08_prefetch_timing --case prefetch_timing --config research/l2l/configs/async_qwen3_0.6b.yaml \
  --device cuda --trials 9 --warmup 2 --max-new 16 --hypothesis H06

echo "== done; new results: ${new[*]:-none}" | tee -a "$LOG"
cat "$LOG" >> "$FINAL_LOG"
if [ ${#new[@]} -gt 0 ]; then
  git add "${new[@]}" "$FINAL_LOG" && git commit -q -m "laptop checks at $COMMIT: ${new[*]##*/}" && git log --oneline -1
  echo "committed; now: git push"
fi
