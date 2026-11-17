# Ticket 12: DFlash on the L2L generator

## What was inspected first

The drafter `z-lab/Qwen3-4B-DFlash-b16`: its `config.json`, the official `dflash.py` / `utils.py` (the model and
`spec_generate`), and the README pointing to arXiv 2602.06036 and github.com/z-lab/dflash. These were downloaded to a
scratch directory and read; none of that code is executed.

DFlash differs from ordinary speculative decoding in several ways:

| aspect | DFlash |
|---|---|
| drafter input | target **hidden features**: outputs of target layers 1, 9, 17, 25, 33, concatenated (5 × 2560), passed through `fc` and an RMSNorm |
| drafter | 5 Qwen3-style layers with **non-causal** attention over [new context, noise block]; its cache keeps context only |
| block | [last token, mask_token_id × 15], embedded with the **target's** embedding |
| draft logits | the **target's** LM head; drafts are argmax |
| verification | one target pass over the 16-token block gives logits *and* the next round's features |
| acceptance | the target draws its own token at each block position; accept the matching prefix, then emit the target's draw |

## Implementation (`qwen_grpo/execution/dflash.py`)

**`DFlashDrafter`** re-implements the drafter and loads only its safetensors weights. Any key mismatch is an error.

**`dflash_generate`** uses the L2L target engine: `_target_step` runs the layer executor, and collects the feature
layers' outputs alongside the logits for every block column. It keeps the plain generator's conventions exactly:
prompt expansion, left-padding positions, masks, EOS → pad, and stopping.

**Acceptance, batched.** Each row accepts the longest draft prefix that equals the target's own draws, which are argmax
when greedy and `multinomial(softmax)` when sampling. All rows then advance by **m = the minimum over unfinished rows**,
which keeps the KV caches aligned; a row that matched further emits `y_m`, which equals its draft.

**Why this is exact.** Every emitted token is the target's own draw given the emitted prefix. For an argmax
(point-mass) drafter this is the same accept/resample law as standard speculative sampling.

**KV state after a round.** The target cache (`KVCache`) is cropped to the committed length minus the last token, which
is the next block's first input. The drafter cache holds only consumed context.

**The verifier is always the current LoRA policy.** It's the PEFT model run through the executor; the drafter only
proposes. **Residency:** the drafter (about 1.08 GB in bf16) is GPU-resident, and the target streams as in plain
generation.

**Configuration.** `dflash.enabled` with an L2L backend makes `make_backend` load the drafter and use
`DFlashGenerator` for rollouts. Evaluation stays plain greedy. `config.unimplemented` refuses DFlash only with the
resident backend. Config: `configs/experiments/qwen3_4b_l2l_dflash.yaml`.

## Correctness evidence (CPU; `tests/test_dflash.py`)

Correctness doesn't depend on drafter quality, so these tests use a tiny target and tiny drafters:
- **Greedy, random drafter:** every draft is rejected, target passes = tokens, and the output equals plain streamed
  generation. Run on the L2L executor, with `model.generate` patched to fail.
- **Greedy, oracle drafter** (the true continuation with one wrong token every other round): full and partial
  acceptance and rejections all occur, target passes are fewer than tokens, and the output equals plain generation
  exactly. This exercises the KV crop and the drafter-context bookkeeping.
- **Sampling, argmax oracle:** 2,000 single-row continuations with more than 50 accepted drafts. The joint law of the
  first two tokens matches the model's **exact** law (chi-square, predefined rejection at p < 10⁻³). The plain sampler
  passes the same test as a control.

## Known limitation, to be measured

**Rows advance by the minimum acceptance.** With large batches the minimum is often 0. The DFlash unit test needed
single rows to exercise acceptance, because a 6,000-row batch accepted nothing. For GRPO rollouts (B × G = 30 rows, 4
new tokens in `configs/qwen.yaml`), the speedup may therefore be small or negative. `scripts/dflash_bench.py` measures
acceptance per pass and passes per token, with greedy and sampled decoding, at 4 and 32 new tokens. Per-row ragged
acceptance would need per-row cache lengths, which aren't implemented.

## Real-model experiment (`scripts/dflash_bench.py`, laptop, `laptop_checks.sh` step 4d)

**Setup:** Qwen3-4B with L2L async, the DFlash drafter, the 8 GiB guard, and policies at B scales 0, 0.01 and 0.03.
These are synthetic policy shifts away from the base the drafter was trained on; `--adapters` takes real checkpoints.

**Measured:**
- greedy token equality, which must hold;
- passes per token and accepted drafts per pass;
- greedy and sampled generation time, plain vs DFlash;
- one full GRPO step with each rollout generator;
- peak GPU memory per part;
- the budget verdict, under H11's evidence rule.

## Not established here

- **Everything on the real model:** acceptance, timings and memory. It needs a GPU and the 4B weights plus the drafter
  (about 9 GB of downloads).
- **Per-row ragged acceptance** (see above).
- **Training-time effects:** DFlash changes only how tokens are produced, not their distribution.
