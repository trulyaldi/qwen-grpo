"""Two-example overfitting check: can GRPO + LoRA raise s(x+) and lower s(x-) at the same checkpoint?

s(x) = log P(POSITIVE<|im_end|> | x) - log P(NEGATIVE<|im_end|> | x), scored with train_multistep.label_scores.
Run from the repo root: .venv/bin/python -m diagnostics.two_example_grpo
"""
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from baseline import EOS_IDS, MODEL, build_prompts
from policy import compute_logprobs, generate_rollouts
from sentiment import LABELS, load_splits
from train_grpo import DTYPE, decode
from train_lora import ALPHA, RANK, digest, lora_model
from train_multistep import (CLIP_EPS, EXPECTED_MODULES, G, GPU_BUDGET_GIB, LEARNING_RATE, MAX_NEW_TOKENS, SEED, TARGETS,
                             TRAIN_SAMPLES, balanced_subset, check_grouping, evaluate, group_rewards, grpo_step,
                             label_scores, lora_modules)

ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS, MAX_ATTEMPTS = 100, 200
SCHEDULE = (1, 5, 10, 20, 30, 50, 75, 100)
MIN_UPDATES = 30  # fewer successful updates than the previous 30-step runs -> INCONCLUSIVE


def select_pair(examples, scores):
    """Per class, the example with the smallest |s0|: least confident, so both rewards are likely in its groups.
    Fixed rule over the existing 32-example training subset; returns [x+, x-]."""
    pick = {}
    for label in ("POSITIVE", "NEGATIVE"):
        idx = [i for i, e in enumerate(examples) if e["label"] == label]
        pick[label] = min(idx, key=lambda i: abs(scores[i]))
    return [examples[pick["POSITIVE"]], examples[pick["NEGATIVE"]]]


def decompose(s0, st):
    """s0, st = (s(x+), s(x-)) initially and now -> deltas, margin M, common shift C, differential D."""
    dp, dn = st[0] - s0[0], st[1] - s0[1]
    return {"s_pos": st[0], "s_neg": st[1], "d_pos": dp, "d_neg": dn, "margin": st[0] - st[1],
            "C": (dp + dn) / 2, "D": (dp - dn) / 2}


def separated(row, tol):
    return row["d_pos"] > tol and row["d_neg"] < -tol  # equivalent to D > |C| + tol


def classify(rows, tol, n_updates, unstable=False):
    """rows: scheduled checkpoints after step 0, in order, each with decompose() fields and 'confirmed'
    (criterion also holds when re-scored one row at a time). PASS needs two consecutive separated checkpoints,
    or a separated final checkpoint confirmed by the re-scoring."""
    if unstable or n_updates < MIN_UPDATES or not rows:
        return "INCONCLUSIVE"
    ok = [separated(r, tol) and r["confirmed"] for r in rows]
    if any(a and b for a, b in zip(ok, ok[1:])) or ok[-1]:
        return "PASS"
    if rows[-1]["D"] > tol:  # M_t - M_0 = 2 D_t, and each score carries up to tol: margin gain beyond 2 tol
        return "PARTIAL"
    return "FAIL"


def score_with_grad(model, tok, example):
    """s(x) with autograd, same inputs as label_scores (one row)."""
    enc = tok(build_prompts(tok, [example["text"]]), return_tensors="pt", add_special_tokens=False).to(model.device)
    logp = {}
    for label in LABELS:
        c = torch.tensor([tok(label, add_special_tokens=False)["input_ids"] + [EOS_IDS[0]]], device=model.device)
        r = {"input_ids": torch.cat([enc["input_ids"], c], 1), "prompt_len": enc["input_ids"].shape[1],
             "attention_mask": torch.cat([enc["attention_mask"], torch.ones_like(c)], 1)}
        logp[label] = compute_logprobs(model, r).sum()
    return logp["POSITIVE"] - logp["NEGATIVE"]


def score_grads(model, tok, pair, params):
    """[g(x+), g(x-)], each the flattened gradient of s w.r.t. the LoRA parameters (float64, CPU)."""
    out = []
    for ex in pair:
        model.zero_grad(set_to_none=True)
        score_with_grad(model, tok, ex).backward()
        out.append(torch.cat([p.grad.double().flatten().cpu() for p in params.values()]))
    model.zero_grad(set_to_none=True)
    return out


def flat(params):
    return torch.cat([p.detach().double().flatten().cpu() for p in params.values()])


def first_order(g, dtheta):
    gp, gn = g
    return {"pred_d_pos": float(gp @ dtheta), "pred_d_neg": float(gn @ dtheta),
            "cos_g_pos_g_neg": float(gp @ gn / (gp.norm() * gn.norm())), "g_pos_norm": float(gp.norm()),
            "g_neg_norm": float(gn.norm()), "update_norm": float(dtheta.norm())}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    torch.manual_seed(SEED)
    model = lora_model(DTYPE, "cuda", TARGETS["last2-attention"])
    torch.cuda.reset_peak_memory_stats()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    modules = lora_modules(model)
    assert [n for n, _ in modules] == EXPECTED_MODULES["last2-attention"]
    assert all(p.dtype == torch.float32 for p in params.values()) and all(p.dtype == DTYPE for p in frozen.values())
    frozen_digest = {n: digest(p) for n, p in frozen.items()}
    pairs = [(m.lora_A["default"].weight, m.lora_B["default"].weight) for _, m in modules]

    subset = balanced_subset(load_splits()["train"], TRAIN_SAMPLES, SEED)  # the train subset of earlier runs
    s_subset = label_scores(model, tok, subset)
    pair = select_pair(subset, s_subset)
    greedy0 = evaluate(model, tok, pair)

    def score():
        both = label_scores(model, tok, pair, batch=2)
        single = label_scores(model, tok, pair, batch=1)  # different bf16 batch shape: numerical spread
        return both, max(abs(a - b) for a, b in zip(both, single)), single

    s0, spread0, single0 = score()
    with torch.no_grad():
        assert abs(score_with_grad(model, tok, pair[0]).item() - single0[0]) < 1e-3
    g0 = score_grads(model, tok, pair, params)  # gradients of s at init, for the first-order check
    print(f"x+ id {pair[0]['id']} s0 {s0[0]:+.4f} greedy {greedy0[0]['prediction']}: {pair[0]['text']}")
    print(f"x- id {pair[1]['id']} s0 {s0[1]:+.4f} greedy {greedy0[1]['prediction']}: {pair[1]['text']}")
    print(f"M0 {s0[0] - s0[1]:+.4f}   cos(g+, g-) at init {float(g0[0] @ g0[1] / (g0[0].norm() * g0[1].norm())):+.4f}")

    optimizer = torch.optim.AdamW(params.values(), lr=LEARNING_RATE, weight_decay=0.0)
    theta0 = flat(params)
    checkpoints = [{"step": 0, **decompose(s0, s0), "spread": spread0, "confirmed": False}]
    attempts, updates, first_update = [], [], None
    torch.manual_seed(SEED)
    step, t0, unstable = 0, time.time(), None

    def record():
        st, spread, single = score()
        row = {"step": step, "attempt": len(attempts), **decompose(s0, st), "spread": spread, "single_row": decompose(single0, single)}
        checkpoints.append(row)
        print(f"step {step:3d} (attempt {len(attempts):3d})  s+ {st[0]:+.4f}  s- {st[1]:+.4f}  d+ {row['d_pos']:+.4f}  "
              f"d- {row['d_neg']:+.4f}  M {row['margin']:+.4f}  C {row['C']:+.4f}  D {row['D']:+.4f}  "
              f"|g| {updates[-1]['grad_norm']:.3e}  peak res {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    while step < MAX_STEPS and len(attempts) < MAX_ATTEMPTS:
        prompts = build_prompts(tok, [e["text"] for e in pair])
        rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=MAX_NEW_TOKENS)
        check_grouping(rollouts, tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].cuda(), G)
        responses = decode(tok, rollouts)
        rewards = group_rewards(responses, [e["label"] for e in pair], G)  # [2, 6]: row 0 x+, row 1 x-
        mixed = [bool((r != r[0]).any()) for r in rewards]
        before = flat(params) if step == 0 else None
        try:
            out = grpo_step(model, params, optimizer, rollouts, rewards, clip_eps=CLIP_EPS)
        except RuntimeError as e:  # non-finite loss/gradients: grpo_step takes no step
            unstable = str(e)
            break
        del rollouts
        attempts.append({"attempt": len(attempts) + 1, "responses": responses, "rewards": rewards.tolist(),
                         "mean_reward_pos": rewards[0].mean().item(), "mean_reward_neg": rewards[1].mean().item(),
                         "mixed_pos": mixed[0], "mixed_neg": mixed[1], "updated": out is not None})
        if out is None:
            continue
        step += 1
        if step == 1:
            first_update = first_order(g0, flat(params) - before)
        with torch.no_grad():
            eff = math.sqrt(sum(((ALPHA / RANK) * Bm @ A).norm().item() ** 2 for A, Bm in pairs))
        updates.append({"step": step, "attempt": len(attempts), "signal": "both" if all(mixed) else "POSITIVE only" if mixed[0] else "NEGATIVE only",
                        "loss": out["loss"], "grad_norm": out["grad_norm"], "update_norm": out["update_norm"],
                        "effective_adapter_norm": eff})
        if step in SCHEDULE:
            record()

    if step and checkpoints[-1]["step"] != step:
        record()
    train_seconds = time.time() - t0
    # tolerance: largest batch-shape spread of s seen at any checkpoint (same weights, different bf16 rounding)
    tol = max(c["spread"] for c in checkpoints)
    for c in checkpoints[1:]:
        c["confirmed"] = separated(c["single_row"], tol)
    rows = checkpoints[1:]
    verdict = classify(rows, tol, step, unstable=bool(unstable))
    first_sep = next((c["step"] for c in rows if separated(c, tol)), None)

    final_g = score_grads(model, tok, pair, params)
    final_fo = first_order(final_g, flat(params) - theta0)  # gradient at the end vs total displacement (not first-order valid)
    greedy_final = evaluate(model, tok, pair)
    model.save_pretrained(ROOT / "checkpoints" / "two_example_grpo")
    frozen_changed = [n for n, p in frozen.items() if digest(p) != frozen_digest[n]]

    n = len(attempts)
    summary = {
        "attempts": n, "updates": step,
        "mixed_fraction_pos": sum(a["mixed_pos"] for a in attempts) / n, "mixed_fraction_neg": sum(a["mixed_neg"] for a in attempts) / n,
        "updates_both": sum(u["signal"] == "both" for u in updates),
        "updates_pos_only": sum(u["signal"] == "POSITIVE only" for u in updates),
        "updates_neg_only": sum(u["signal"] == "NEGATIVE only" for u in updates),
        "no_signal_batches": sum(not a["updated"] for a in attempts),
        "mean_reward_pos": sum(a["mean_reward_pos"] for a in attempts) / n, "mean_reward_neg": sum(a["mean_reward_neg"] for a in attempts) / n,
    }
    mem = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    mem["within_budget"] = mem["peak_reserved_gib"] < GPU_BUDGET_GIB
    result = {
        "model": MODEL,
        "config": {"lora_targets": TARGETS["last2-attention"], "r": RANK, "alpha": ALPHA, "dropout": 0.0, "base_dtype": str(DTYPE),
                   "lora_dtype": "torch.float32", "optimizer": "AdamW(weight_decay=0)", "lr": LEARNING_RATE, "B": 2, "G": G,
                   "clip_eps": CLIP_EPS, "max_new_tokens": MAX_NEW_TOKENS, "sampling": "T=1, top_p=1, top_k=0", "seed": SEED,
                   "max_steps": MAX_STEPS, "max_attempts": MAX_ATTEMPTS, "schedule": SCHEDULE,
                   "selection": "train subset of earlier runs (balanced_subset 32, seed 42); per class min |s0|"},
        "selection_pool": [{"id": e["id"], "label": e["label"], "s0": s} for e, s in zip(subset, s_subset)],
        "examples": [{"role": role, "id": e["id"], "text": e["text"], "label": e["label"], "s0": s,
                      "greedy0": g["prediction"], "greedy_final": gf["prediction"]}
                     for role, e, s, g, gf in zip(("x+", "x-"), pair, s0, greedy0, greedy_final)],
        "M0": s0[0] - s0[1], "tolerance": tol, "tolerance_rule": "max |s(batch 2) - s(batch 1)| over all checkpoints",
        "checkpoints": checkpoints, "first_separated_step": first_sep, "verdict": verdict,
        "reward_summary": summary, "updates": updates, "attempts": attempts,
        "first_order": {"first_update_at_init": first_update, "final_gradient_vs_total_displacement": final_fo},
        "frozen_changed": frozen_changed, "unstable": unstable, "train_seconds": train_seconds, "memory": mem,
        "limitations": ["single seed; rollouts stochastic", "bf16 scores: tolerance from batch-shape spread only",
                        "final first-order entry uses the final gradient with the total displacement: a direction check, not a prediction"],
    }
    (ROOT / "results" / "two_example_grpo.json").write_text(json.dumps(result, indent=1))

    print("\n| Step | s(x+) | s(x-) | Δ+ | Δ- | M | C | D |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in checkpoints:
        print(f"| {c['step']} | {c['s_pos']:+.4f} | {c['s_neg']:+.4f} | {c['d_pos']:+.4f} | {c['d_neg']:+.4f} | {c['margin']:+.4f} | {c['C']:+.4f} | {c['D']:+.4f} |")
    print(f"\ntolerance {tol:.2e}; first separated step {first_sep}; verdict {verdict}")
    print(f"rewards: {summary}")
    print(f"first update: {first_update}\nfinal gradient vs displacement: {final_fo}")
    print(f"greedy x+: {greedy0[0]['prediction']} -> {greedy_final[0]['prediction']}, x-: {greedy0[1]['prediction']} -> {greedy_final[1]['prediction']}")
    print(f"frozen changed {len(frozen_changed)}/{len(frozen)}; train {train_seconds:.0f} s; memory {mem}")
    if frozen_changed:
        raise SystemExit("FAILED: frozen weights changed")


if __name__ == "__main__":
    main()
