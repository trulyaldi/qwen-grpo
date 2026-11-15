"""Multi-step GRPO + LoRA on Qwen3-0.6B: fp32 adapters on selected projections of a frozen bf16 model.

Rollouts: T=1 sampling, MAX_NEW_TOKENS. Evaluation: baseline.predict (greedy, baseline.MAX_NEW tokens, BATCH chunks).
Checkpoints hold adapter weights only; optimizer and RNG state are not saved, so they cannot resume training exactly.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer

from baseline import BATCH, EOS_IDS, MAX_NEW, MODEL, build_prompts, predict
from grpo import group_advantages, grpo_loss
from policy import compute_logprobs, generate_rollouts
from sentiment import LABELS, load_splits, parse_label, reward
from train_grpo import DTYPE, INIT_TOL, decode, masked_max
from train_lora import ALPHA, RANK, TARGET, digest, lora_model

ROOT = Path(__file__).parent
BASE_FILE = ROOT / "results" / "baseline_validation.json"
# PEFT re.fullmatch-es a string target against module names, so these regexes select exact modules.
TARGETS = {
    "last-o-proj": TARGET,
    "last2-attention": r"model\.layers\.(26|27)\.self_attn\.(q|v|o)_proj",
}
EXPECTED_MODULES = {
    "last-o-proj": ["model.layers.27.self_attn.o_proj"],
    "last2-attention": [f"model.layers.{i}.self_attn.{p}_proj" for i in (26, 27) for p in "qvo"],
}
SUFFIX = {"last-o-proj": "", "last2-attention": "_last2"}
BATCH_MODES = ("cycle", "balanced", "unconstrained")
TRAIN_SAMPLES, VAL_SUBSET, G = 32, 160, 6
SCORE_BATCH = 8  # label scoring keeps full [N, T, V] logits; 8 rows keeps that well inside the GPU budget
MAX_NEW_TOKENS, MAX_STEPS, EVAL_EVERY = 8, 30, 10
LEARNING_RATE, CLIP_EPS, SEED = 1e-4, 0.2, 42
GPU_BUDGET_GIB, BOOTSTRAP = 4.0, 10_000


def output_name(targets, tag="", B=1, mode="cycle"):
    return f"grpo_lora{SUFFIX[targets]}" + (f"_b{B}_{mode}" if B > 1 else "") + tag


def _ref(targets, B=1, mode="cycle"):
    name = output_name(targets, B=B, mode=mode)
    return ROOT / "results" / f"{name}_step{MAX_STEPS}_validation.json", ROOT / "checkpoints" / f"{name}_step{MAX_STEPS}", targets


# Saved step-30 predictions (and adapters, for label scoring) of earlier runs: (predictions, checkpoint, targets).
REFERENCES = {
    "base": (BASE_FILE, None, None),
    "last-o-proj B=1": _ref("last-o-proj"),
    "last2 B=1": _ref("last2-attention"),
    "last2 B=2 balanced": _ref("last2-attention", 2, "balanced"),
    "last2 B=2 unconstrained": _ref("last2-attention", 2, "unconstrained"),
}


def batches(train, B, mode, seed=SEED):
    """Deterministic, infinite stream of training batches. A zero-signal batch is consumed, never retried.

    cycle:         B=1; train[0], train[1], ... in subset order (the original policy)
    balanced:      B=2; [next POSITIVE, next NEGATIVE], each class reshuffled (own seeded RNG) after each pass
    unconstrained: B prompts from one stream over the whole subset, reshuffled after each pass
    """
    def stream(pool, rng):
        while True:
            order = list(pool)
            rng.shuffle(order)
            yield from order

    if mode == "cycle":
        assert B == 1
        return ([train[i % len(train)]] for i in itertools.count())
    if mode == "balanced":
        assert B == 2
        pos, neg = (stream([e for e in train if e["label"] == label], random.Random(seed + k))
                    for k, label in enumerate(("POSITIVE", "NEGATIVE")))
        return ([next(pos), next(neg)] for _ in itertools.count())
    assert mode == "unconstrained"
    s = stream(train, random.Random(seed))
    return ([next(s) for _ in range(B)] for _ in itertools.count())


def check_grouping(rollouts, prompt_ids, G):
    """Rows b*G .. b*G+G-1 must start with prompt b's left-padded tokens, the order group_rewards assumes."""
    P = rollouts["prompt_len"]
    assert torch.equal(rollouts["input_ids"][:, :P], prompt_ids.repeat_interleave(G, dim=0)), "rollout rows not grouped by prompt"


def lora_modules(model):
    """(name without PEFT prefix, module) for every module carrying a LoRA adapter."""
    return [(n.removeprefix("base_model.model."), m) for n, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)]


def expected_lora_params(modules):
    """sum over adapted Linear layers of r * (in + out): A is [r, in], B is [out, r]."""
    return sum(RANK * (m.base_layer.in_features + m.base_layer.out_features) for _, m in modules)


def l2(norms):
    """Aggregate L2 norm sqrt(sum_k ||theta_k||^2) from per-tensor norms."""
    return math.sqrt(sum(x * x for x in norms))


def balanced_subset(rows, n, seed):
    """n/2 rows per label, drawn and shuffled with a fixed seed; id = index in `rows`."""
    assert n % 2 == 0
    rng = random.Random(seed)
    picked = [i for label in LABELS for i in rng.sample([i for i, r in enumerate(rows) if r["label"] == label], n // 2)]
    rng.shuffle(picked)
    return [{"id": i, **rows[i]} for i in picked]


def group_rewards(responses, labels, G):
    """responses [B*G], rows b*G .. b*G+G-1 answer prompt b -> rewards [B, G]."""
    assert len(responses) == len(labels) * G, (len(responses), len(labels), G)
    return torch.tensor([reward(r, labels[i // G]) for i, r in enumerate(responses)]).reshape(len(labels), G)


def grpo_step(model, params, optimizer, rollouts, rewards, clip_eps=CLIP_EPS):
    """One AdamW update from a fresh rollout group -> scalar logs, or None (no step) if no group has mixed rewards."""
    if (rewards == rewards[:, :1]).all():
        return None
    adv = group_advantages(rewards).flatten().to(rollouts["input_ids"].device)
    mask = rollouts["completion_mask"]
    with torch.no_grad():
        old_logp = compute_logprobs(model, rollouts)  # sampling policy of this group only
    new_logp = compute_logprobs(model, rollouts)
    init_diff = masked_max(new_logp.detach() - old_logp, mask)
    if init_diff > INIT_TOL:
        raise RuntimeError(f"old/current log-prob mismatch before the step: {init_diff:.3e}")
    loss = grpo_loss(new_logp, old_logp, adv, mask, clip_eps=clip_eps)
    if not (torch.isfinite(adv).all() and torch.isfinite(loss)):
        raise RuntimeError(f"non-finite advantages {adv.tolist()} or loss {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in params.items()}
    if not all(math.isfinite(g) for g in grad_norms.values()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"non-finite gradients {grad_norms}; no optimizer step taken")
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(), "init_diff": init_diff, "grad_norms": grad_norms,
        "grad_norm": math.sqrt(sum(g * g for g in grad_norms.values())),
        "update_norm": math.sqrt(sum((p.detach() - before[n]).pow(2).sum().item() for n, p in params.items())),
    }


def metrics(preds):
    n = len(preds)
    m = {"n": n, "accuracy": sum(p["correct"] for p in preds) / n,
         "invalid_rate": sum(p["prediction"] is None for p in preds) / n,
         "predicted_positive_rate": sum(p["prediction"] == "POSITIVE" for p in preds) / n,
         "predicted_negative_rate": sum(p["prediction"] == "NEGATIVE" for p in preds) / n}
    f1 = []
    for label in LABELS:
        tp = sum(p["prediction"] == label == p["label"] for p in preds)
        gold, pred = sum(p["label"] == label for p in preds), sum(p["prediction"] == label for p in preds)
        m[f"recall_{label}"] = tp / gold
        f1.append(2 * tp / (gold + pred))  # 2TP / (2TP + FP + FN); invalid outputs count as FN
    m["macro_f1"] = sum(f1) / len(f1)
    m["confusion"] = {str(k): {g: sum(p["prediction"] == k and p["label"] == g for p in preds) for g in LABELS}
                      for k in (*LABELS, None)}  # rows: predicted (None = invalid), columns: gold
    return m


def paired_bootstrap(base_correct, new_correct, n=BOOTSTRAP, seed=SEED, chunk=1000):
    """Mean accuracy difference and 95% percentile CI, resampling paired examples with replacement."""
    d = torch.tensor(new_correct, dtype=torch.float64) - torch.tensor(base_correct, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    means = torch.cat([d[torch.randint(len(d), (chunk, len(d)), generator=gen)].mean(1) for _ in range(n // chunk)])
    return d.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item()


def paired_changes(base, new):
    assert [(p["id"], p["text"], p["label"]) for p in base] == [(p["id"], p["text"], p["label"]) for p in new]
    out = {label: {"base_wrong_now_right": sum(not b["correct"] and t["correct"] for b, t in zip(base, new) if b["label"] == label),
                   "base_right_now_wrong": sum(b["correct"] and not t["correct"] for b, t in zip(base, new) if b["label"] == label)}
           for label in LABELS}
    out["flips"] = dict(Counter(f"{b['prediction']}->{t['prediction']}" for b, t in zip(base, new) if b["prediction"] != t["prediction"]))
    out["invalid_to_valid"] = sum(b["prediction"] is None and t["prediction"] is not None for b, t in zip(base, new))
    out["valid_to_invalid"] = sum(b["prediction"] is not None and t["prediction"] is None for b, t in zip(base, new))
    return out


def paired_compare(ref, new):
    """new vs ref on the same examples (asserted aligned): accuracy difference, paired bootstrap CI, flips."""
    changes = paired_changes(ref, new)
    diff, lo, hi = paired_bootstrap([p["correct"] for p in ref], [p["correct"] for p in new])
    return {"ref_metrics": metrics(ref), "accuracy_diff": diff, "ci95": [lo, hi], "paired_changes": changes}


@torch.no_grad()
def label_scores(model, tok, examples, batch=SCORE_BATCH):
    """s(x) = log P(POSITIVE<|im_end|> | x) - log P(NEGATIVE<|im_end|> | x): teacher-forced sums over every token
    of the label's full tokenization plus the end-of-turn token, on the evaluation prompt. Not used in training."""
    conts = {label: tok(label, add_special_tokens=False)["input_ids"] + [EOS_IDS[0]] for label in LABELS}
    out = []
    for i in range(0, len(examples), batch):
        chunk = examples[i : i + batch]
        enc = tok(build_prompts(tok, [e["text"] for e in chunk]), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logp = {}
        for label, c in conts.items():
            c = torch.tensor(c, device=model.device).expand(len(chunk), -1)
            r = {"input_ids": torch.cat([enc["input_ids"], c], 1), "prompt_len": enc["input_ids"].shape[1],
                 "attention_mask": torch.cat([enc["attention_mask"], torch.ones_like(c)], 1)}
            logp[label] = compute_logprobs(model, r).sum(1)
        out += (logp["POSITIVE"] - logp["NEGATIVE"]).tolist()
    return out


def win_matrix(scores, labels):
    """W[i, j] = 1[s_pos_i > s_neg_j] + 0.5 * 1[tie], positives and negatives in example order; AUC = W.mean()."""
    s = torch.tensor(scores, dtype=torch.float64)
    is_pos = torch.tensor([label == "POSITIVE" for label in labels])
    p, n = s[is_pos][:, None], s[~is_pos][None, :]
    return (p > n).double() + 0.5 * (p == n).double()


def paired_auc_bootstrap(W_new, W_ref, n=BOOTSTRAP, seed=SEED, chunk=500):
    """AUC difference with a class-stratified paired bootstrap: the same resampled examples score both models.

    With multiplicity vectors c1, c0 of resampled positives/negatives, the resampled AUC is c1^T W c0 / (n1 n0).
    """
    n1, n0 = W_new.shape
    gen = torch.Generator().manual_seed(seed)
    counts = lambda m: torch.zeros(chunk, m, dtype=torch.float64).scatter_add_(
        1, torch.randint(m, (chunk, m), generator=gen), torch.ones(chunk, m, dtype=torch.float64))
    diffs = []
    for _ in range(n // chunk):
        c1, c0 = counts(n1), counts(n0)
        diffs.append((((c1 @ W_new) * c0).sum(1) - ((c1 @ W_ref) * c0).sum(1)) / (n1 * n0))
    d = torch.cat(diffs)
    return {"auc_diff": (W_new.mean() - W_ref.mean()).item(), "ci95": [d.quantile(0.025).item(), d.quantile(0.975).item()]}


def evaluate(model, tok, examples):
    return predict(model, tok, examples, [e["id"] for e in examples])[0]


def summary(m):
    return (f"acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}  rec POS {m['recall_POSITIVE']:.4f} NEG {m['recall_NEGATIVE']:.4f}"
            f"  invalid {m['invalid_rate']:.4f}  pred POS {m['predicted_positive_rate']:.4f}  (n={m['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    ap.add_argument("--val-subset", type=int, default=VAL_SUBSET)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--max-attempts", type=int, default=None, help="rollout-group budget (default 10 * steps)")
    ap.add_argument("--full-val", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora-targets", choices=TARGETS, default="last-o-proj")
    ap.add_argument("--batch-size", type=int, default=1, help="prompts per optimizer step (B)")
    ap.add_argument("--batch-mode", choices=BATCH_MODES, default=None, help="default: cycle for B=1")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _smoke or _lr0")
    args = ap.parse_args()
    B, mode = args.batch_size, args.batch_mode or ("cycle" if args.batch_size == 1 else None)
    if (mode == "cycle") != (B == 1) or (mode == "balanced" and B != 2):
        ap.error("B=1 uses --batch-mode cycle; balanced needs --batch-size 2; B>1 needs an explicit mode")
    max_attempts = args.max_attempts or 10 * args.steps
    name = output_name(args.lora_targets, args.tag, B, mode)

    splits = load_splits()
    train = balanced_subset(splits["train"], args.train_samples, SEED)
    val_all = [{"id": i, **r} for i, r in enumerate(splits["validation"])]
    val_sub = sorted(balanced_subset(splits["validation"], args.val_subset, SEED), key=lambda e: e["id"])
    base_file = json.loads(BASE_FILE.read_text())["predictions"]
    base_file_m = metrics(base_file)
    assert len(base_file) == 1066 and round(base_file_m["accuracy"], 4) == 0.7486 and sum(p["prediction"] is None for p in base_file) == 3
    assert [(p["id"], p["text"], p["label"]) for p in base_file] == [(e["id"], e["text"], e["label"]) for e in val_all]

    torch.manual_seed(SEED)  # LoRA A init
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    assert tok.pad_token_id == EOS_IDS[1]  # baseline.py pads with tok.pad_token_id, lora_model with EOS_IDS[1]
    model = lora_model(DTYPE, "cuda", TARGETS[args.lora_targets])
    torch.cuda.reset_peak_memory_stats()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    modules = lora_modules(model)
    module_names = [n for n, _ in modules]
    assert module_names == EXPECTED_MODULES[args.lora_targets], module_names
    n_trainable = sum(p.numel() for p in params.values())
    assert n_trainable == expected_lora_params(modules) and len(params) == 2 * len(modules)
    assert all(".lora_A." in n or ".lora_B." in n for n in params)
    assert all(p.dtype == torch.float32 for p in params.values())
    assert all(p.dtype == DTYPE for p in frozen.values())
    names_A, names_B = [n for n in params if ".lora_A." in n], [n for n in params if ".lora_B." in n]
    pairs = [(m.lora_A["default"].weight, m.lora_B["default"].weight) for _, m in modules]
    assert all(params[n].abs().max() == 0 for n in names_B)  # adapter contributes exactly zero: initial model = base
    n_total = n_trainable + sum(p.numel() for p in frozen.values())
    print(f"LoRA modules ({len(modules)}): {module_names}")
    print(f"trainable parameters: {n_trainable:,} of {n_total:,} ({100 * n_trainable / n_total:.4f}%)")
    init = {n: p.detach().clone() for n, p in params.items()}
    frozen_digest = {n: digest(p) for n, p in frozen.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=0.0)

    result = {
        "model": MODEL, "method": "GRPO+LoRA", "seed": SEED,
        "config": {"lr": args.lr, "steps": args.steps, "max_attempts": max_attempts, "train_samples": args.train_samples,
                   "train_ids": [e["id"] for e in train], "val_subset_ids": [e["id"] for e in val_sub],
                   "B": B, "G": G, "batch_mode": mode,
                   "batch_policy": f"{mode} (see batches()); a zero-signal batch is consumed, not retried", "rollout_max_new_tokens": MAX_NEW_TOKENS, "eval_max_new_tokens": MAX_NEW,
                   "eval_batch": BATCH, "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0},
                   "clip_eps": CLIP_EPS, "kl_beta": 0.0, "optimizer": "AdamW(weight_decay=0)", "grad_clipping": None,
                   "lora": {"targets": args.lora_targets, "target_regex": TARGETS[args.lora_targets], "modules": module_names,
                            "trainable_params": n_trainable, "r": RANK, "alpha": ALPHA, "dropout": 0.0},
                   "dtypes": {"base": str(DTYPE), "lora": "torch.float32"}, "gpu_budget_gib": GPU_BUDGET_GIB,
                   "checkpoint_contents": "adapter weights + adapter_config only (no optimizer/RNG state)"},
        "training": [], "zero_signal": [], "evaluations": [],
    }

    base_full = None
    if args.full_val:
        base_full = evaluate(model, tok, val_all)
        mism = [p["id"] for p, q in zip(base_full, base_file) if p["response"] != q["response"]]
        result["base_reproduction"] = {"n": len(base_full), "mismatched_ids": mism, "metrics": metrics(base_full)}
        print(f"base full validation (in-process): {summary(metrics(base_full))}\n  mismatches vs {BASE_FILE.name}: {len(mism)}")

    def eval_subset(step, ckpt=None):
        preds = evaluate(model, tok, val_sub)
        result["evaluations"].append({"step": step, "checkpoint": ckpt, "split": "validation", "metrics": metrics(preds), "predictions": preds})
        print(f"  [val subset, step {step}] {summary(metrics(preds))}")

    eval_subset(0)
    torch.manual_seed(SEED)  # rollout sampling
    step = attempts = 0
    ckpt = None
    stream, mixed_counts, t0 = batches(train, B, mode), Counter(), time.time()
    while step < args.steps and attempts < max_attempts:
        batch = next(stream)
        attempts += 1
        prompts = build_prompts(tok, [e["text"] for e in batch])
        rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=MAX_NEW_TOKENS)
        check_grouping(rollouts, tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].cuda(), G)
        responses = decode(tok, rollouts)
        rewards = group_rewards(responses, [e["label"] for e in batch], G)  # [B, G]
        mixed = int((rewards != rewards[:, :1]).any(dim=1).sum())
        mixed_counts[mixed] += 1
        group = {"attempt": attempts, "prompt_ids": [e["id"] for e in batch], "labels": [e["label"] for e in batch],
                 "mixed_groups": mixed, "responses": responses, "rewards": rewards.tolist(), "mean_reward": rewards.mean().item(),
                 "invalid_responses": sum(parse_label(r) is None for r in responses),
                 "rollout_sha256": hashlib.sha256(rollouts["response_ids"].cpu().numpy().tobytes()).hexdigest()[:16]}
        out = grpo_step(model, params, optimizer, rollouts, rewards)
        del rollouts
        if out is None:
            result["zero_signal"].append(group)
            continue
        step += 1
        with torch.no_grad():  # norms over several modules aggregate as sqrt(sum of squares)
            row = {"step": step, **group, "loss": out["loss"], "init_diff": out["init_diff"], "grad_norm": out["grad_norm"],
                   "grad_norm_A": l2(out["grad_norms"][n] for n in names_A), "grad_norm_B": l2(out["grad_norms"][n] for n in names_B),
                   "update_norm": out["update_norm"],
                   "lora_A_norm": l2(params[n].norm().item() for n in names_A),
                   "lora_B_norm": l2(params[n].norm().item() for n in names_B),
                   "effective_adapter_norm": l2(((ALPHA / RANK) * Bm @ A).norm().item() for A, Bm in pairs),
                   "change_from_init": math.sqrt(sum((p - init[n]).pow(2).sum().item() for n, p in params.items())),
                   "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        result["training"].append(row)
        ids = ",".join(f"{e['id']}{e['label'][0]}" for e in batch)
        print(f"step {step:2d} (attempt {attempts:3d}) {ids:12} mixed {mixed}  R {row['mean_reward']:.2f}  "
              f"loss {row['loss']:+.4f}  |g| {row['grad_norm']:.3e} (A {row['grad_norm_A']:.2e}, B {row['grad_norm_B']:.2e})  "
              f"|dtheta| {row['update_norm']:.2e}  |(a/r)BA| {row['effective_adapter_norm']:.3e}  "
              f"alloc {row['cuda_allocated_gib']:.2f}  peak res {row['peak_reserved_gib']:.2f} GiB")
        if step % args.eval_every == 0 or step == args.steps:
            ckpt = f"checkpoints/{name}_step{step}"
            model.save_pretrained(ROOT / ckpt)
            eval_subset(step, ckpt)

    result["train_seconds"] = time.time() - t0
    result["groups_generated"], result["successful_steps"] = attempts, step  # groups_generated counts batches
    result["prompt_groups"], result["responses_generated"] = attempts * B, attempts * B * G
    result["batches_by_mixed_groups"] = {str(k): mixed_counts[k] for k in range(B + 1)}
    result["zero_signal_groups"] = len(result["zero_signal"])  # batches skipped: no group had mixed rewards
    if step < args.steps:
        print(f"Attempt budget exhausted: {step} of {args.steps} steps after {attempts} groups")

    result["frozen_changed"] = [n for n, p in frozen.items() if digest(p) != frozen_digest[n]]
    result["params_equal_init"] = all(torch.equal(p, init[n]) for n, p in params.items())
    if args.lr == 0:
        subset = [e["predictions"] for e in result["evaluations"]]
        result["control"] = {"params_equal_init": result["params_equal_init"],
                             "subset_predictions_unchanged": all(s == subset[0] for s in subset)}

    if ckpt is not None:  # evaluate the saved artifact, not just the in-memory adapter
        live = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        loaded = load_peft_weights(str(ROOT / ckpt))
        assert live.keys() == loaded.keys() and all(torch.equal(live[k], loaded[k].to(live[k].device)) for k in live), ckpt
        set_peft_model_state_dict(model, loaded)
        assert all(torch.equal(p, live[k]) for k, p in get_peft_model_state_dict(model).items())
        result["final_checkpoint"] = {"path": ckpt, "step": step, "reloaded_bitwise_equal": True}

    if args.full_val and ckpt is not None:
        trained_full = evaluate(model, tok, val_all)
        tm, bm = metrics(trained_full), metrics(base_full)
        diff, lo, hi = paired_bootstrap([p["correct"] for p in base_full], [p["correct"] for p in trained_full])
        out_path = ROOT / "results" / f"{name}_step{step}_validation.json"
        result["final"] = {"base": bm, "trained": tm, "accuracy_diff": diff, "bootstrap_ci95": [lo, hi],
                           "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "paired_by": "id"},
                           "paired_changes": paired_changes(base_full, trained_full),
                           "vs_saved": {ref: paired_compare(json.loads(p.read_text())["predictions"], trained_full)
                                        for ref, (p, _, _) in REFERENCES.items() if p.exists() and p != out_path}}
        scores = {"trained": label_scores(model, tok, val_all)}
        with model.disable_adapter():
            scores["base"] = label_scores(model, tok, val_all)
        for ref, (_, ref_ckpt, targets) in REFERENCES.items():  # same targets -> load into this model
            if ref_ckpt is not None and targets == args.lora_targets and ref_ckpt.exists() and ref_ckpt != ROOT / ckpt:
                set_peft_model_state_dict(model, load_peft_weights(str(ref_ckpt)))
                scores[ref] = label_scores(model, tok, val_all)
        set_peft_model_state_dict(model, loaded)
        W = {k: win_matrix(v, [e["label"] for e in val_all]) for k, v in scores.items()}
        result["final"]["auc"] = {k: w.mean().item() for k, w in W.items()}
        result["final"]["auc_vs"] = {k: paired_auc_bootstrap(W["trained"], W[k]) for k in W if k != "trained"}
        result["final"]["label_scores"] = {"ids": [e["id"] for e in val_all], **scores}
        out_path.write_text(json.dumps({
            "model": MODEL, "checkpoint": f"{name.replace('_', '-')}-step{step}", "split": "validation",
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW}, "metrics": tm, "predictions": trained_full,
        }, indent=1))

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    mem = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "device_gib": total / 2**30}
    mem["within_budget"] = mem["peak_reserved_gib"] < GPU_BUDGET_GIB
    result["memory"] = mem
    (ROOT / "results" / f"{name}_training.json").write_text(json.dumps(result, indent=1))

    rows = result["training"]
    print(f"\nbatches by number of mixed groups: {result['batches_by_mixed_groups']}, prompt groups {attempts * B}, "
          f"responses {attempts * B * G}, train time {result['train_seconds']:.0f} s")
    print(f"steps {step}/{args.steps}, batches {attempts}, zero-signal batches {result['zero_signal_groups']}, "
          f"frozen changed {len(result['frozen_changed'])}/{len(frozen)}, params equal init: {result['params_equal_init']}")
    if rows:
        print(f"grad norm range [{min(r['grad_norm'] for r in rows):.3e}, {max(r['grad_norm'] for r in rows):.3e}]  "
              f"change from init {rows[-1]['change_from_init']:.3e}  |(a/r)BA| {rows[-1]['effective_adapter_norm']:.3e}  "
              f"allocated after step: {min(r['cuda_allocated_gib'] for r in rows):.3f}-{max(r['cuda_allocated_gib'] for r in rows):.3f} GiB")
    if "control" in result:
        print(f"LR=0 control: {result['control']}")
    if "final" in result:
        f = result["final"]
        print(f"\nfull validation  base:    {summary(f['base'])}\nfull validation  trained: {summary(f['trained'])}")
        print(f"accuracy diff {f['accuracy_diff']:+.4f}  paired bootstrap 95% CI [{f['bootstrap_ci95'][0]:+.4f}, {f['bootstrap_ci95'][1]:+.4f}]")
        print(f"paired changes: {f['paired_changes']}")
        for ref, c in f["vs_saved"].items():
            print(f"\nvs saved {ref} predictions: {summary(c['ref_metrics'])}")
            print(f"  accuracy diff {c['accuracy_diff']:+.4f}  paired bootstrap 95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
            print(f"  paired changes: {c['paired_changes']}")
        for k, a in f["auc"].items():
            vs = f["auc_vs"].get(k)
            print(f"ROC AUC {k:24} {a:.4f}" + (f"   trained - this: {vs['auc_diff']:+.4f}  CI [{vs['ci95'][0]:+.4f}, {vs['ci95'][1]:+.4f}]" if vs else ""))
    print(f"peak allocated {mem['peak_allocated_gib']:.2f} GiB, peak reserved {mem['peak_reserved_gib']:.2f} GiB, "
          f"device {mem['device_gib']:.2f} GiB, budget {GPU_BUDGET_GIB} GiB: {'PASS' if mem['within_budget'] else 'FAIL'}"
          "  (allocator stats exclude the CUDA context)")
    if result["frozen_changed"] or ("control" in result and not all(result["control"].values())):
        raise SystemExit("FAILED: frozen weights changed or LR=0 control changed")


if __name__ == "__main__":
    main()
