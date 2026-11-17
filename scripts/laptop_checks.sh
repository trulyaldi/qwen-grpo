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

echo "== 4b. Ticket 10: streaming LoRA merge (tiny fp32/bf16 on CPU; real Qwen3-0.6B + runs/qwen step90 on the GPU)" | tee -a "$LOG"
lab cpu t10_merge_equivalence_tiny --case merge_equivalence --config research/l2l/configs/store_cpu.yaml --device cpu \
  --hypothesis H09
lab gpu t10_merge_equivalence_qwen3_0.6b --case merge_equivalence --config research/l2l/configs/store_qwen3_0.6b.yaml \
  --device cuda --real --hypothesis H09

echo "== 4c. Ticket 11: depth residency (H07) and profiled real-model runs (H11 needs Qwen3-4B; last)" | tee -a "$LOG"
lab gpu t11_depth_residency --case depth_residency --config research/l2l/configs/store.yaml --device cuda --hypothesis H07
prof() {  # prof <out name> <args...>: scripts/profile_run.py unless its result exists
  local name=$1; shift
  if [ -e "$EXP/$name.json" ]; then echo "skip $name (exists)" | tee -a "$LOG"; return; fi
  echo "\$ python scripts/profile_run.py --out $name $*" | tee -a "$LOG"
  python scripts/profile_run.py --out "$name" "$@" 2>&1 | grep -vE "Warning|warn" | tail -12 | tee -a "$LOG"
  [ -e "$EXP/$name.json" ] && new+=("$EXP/$name.json")
}
P06=(--config configs/experiments/qwen_l2l.yaml --prediction results/budget/predictions/qwen3-0.6b_4gb.json --budget-gib 4)
prof t11_profile_qwen3_0.6b_sync "${P06[@]}" --backend l2l_sync
prof t11_profile_qwen3_0.6b_async "${P06[@]}" --backend l2l_async
prof t11_profile_qwen3_0.6b_async_mb5 "${P06[@]}" --backend l2l_async --micro-batch-size 5
prof t11_profile_qwen3_0.6b_async_kvcpu "${P06[@]}" --backend l2l_async --kv-cache cpu
prof t11_profile_qwen3_4b_async --config configs/experiments/qwen3_4b_l2l.yaml \
  --prediction results/budget/predictions/qwen3-4b_8gb.json --budget-gib 8 --steps 1 --eval-n 16

echo "== 4d. Ticket 12: DFlash vs plain streamed generation on Qwen3-4B (downloads the 4B model and the drafter)" | tee -a "$LOG"
if [ -e "$EXP/t12_dflash_bench.json" ]; then echo "skip t12_dflash_bench (exists)" | tee -a "$LOG"
else
  python scripts/dflash_bench.py --config configs/experiments/qwen3_4b_l2l_dflash.yaml --max-new 4 32 2>&1 \
    | grep -vE "Warning|warn" | tail -12 | tee -a "$LOG"
  [ -e "$EXP/t12_dflash_bench.json" ] && new+=("$EXP/t12_dflash_bench.json")
fi

echo "== 5. Ticket 09: crash/resume experiment on the real Qwen3-0.6B (H08)" | tee -a "$LOG"
if [ -e "$EXP/t09_resume_qwen_stream.json" ]; then echo "skip t09_resume_qwen_stream (exists)" | tee -a "$LOG"
else
  python scripts/resume_experiment.py --config configs/experiments/qwen_stream_short.yaml --device cuda 2>&1 \
    | grep -E "PASS|FAIL|H08|Error" | tee -a "$LOG"
  [ -e "$EXP/t09_resume_qwen_stream.json" ] && new+=("$EXP/t09_resume_qwen_stream.json")
fi

echo "== done; new results: ${new[*]:-none}" | tee -a "$LOG"
cat "$LOG" >> "$FINAL_LOG"
if [ ${#new[@]} -gt 0 ]; then
  git add "${new[@]}" "$FINAL_LOG" && git commit -q -m "laptop checks at $COMMIT: ${new[*]##*/}" && git log --oneline -1
  echo "committed; now: git push"
fi
