"""Fixed-rollout gradient comparison: manual GRPO (grpo.py, policy.py) vs TRL GRPOTrainer on identical inputs.

Run from the repo root:  .venv/bin/python -m research.trl_reference.compare_gradients
Same model, same LoRA init, same prompts, same completion token ids, same rewards and masks. TRL receives the
fixed completions through its `rollout_func` hook, so its own reward, advantage, mask, log-prob and loss code runs.
"""
import json
import math

import torch
from transformers import AutoTokenizer

from baseline import EOS_IDS, MODEL, build_prompts
from grpo import group_advantages, grpo_loss, response_mask
from policy import compute_logprobs, generate_rollouts
from train_grpo import DTYPE, decode
from train_lora import lora_model
from train_multistep import TARGETS, group_rewards

from research.trl_reference.train import CONFIG, G, HERE, MANUAL_TRAINING, ROOT, BalancedGRPOTrainer, check_lora, grpo_args, make_dataset, make_reward, stream, train_examples

ADV_EPS = 1e-4


def trl_advantages(rewards):
    """rewards [B, G] -> TRL's scale_rewards="group" advantages, restated: (R - mean) / (Bessel-corrected std + 1e-4)."""
    r = rewards.double()
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, unbiased=True, keepdim=True) + ADV_EPS)).float()


def completion_lists(rollouts):
    """Manual rollouts -> TRL-style completion id lists: each row up to and including its first EOS."""
    return [row[m].tolist() for row, m in zip(rollouts["response_ids"].cpu(), rollouts["completion_mask"].cpu())]


def manual_rollouts(tok, prompts, completions, device):
    """TRL-style completion lists for B prompts (rows b*G .. b*G+G-1) -> the manual rollouts dict (policy.py layout)."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    R = max(len(c) for c in completions)
    assert all(c[-1] in EOS_IDS or len(c) == R for c in completions), "an unterminated completion must span all R columns"
    resp = torch.full((len(completions), R), tok.pad_token_id)
    for i, c in enumerate(completions):
        resp[i, : len(c)] = torch.tensor(c)
    mask = response_mask(resp, EOS_IDS)
    prompt_mask = enc["attention_mask"].repeat_interleave(G, 0)
    return {"input_ids": torch.cat([enc["input_ids"].repeat_interleave(G, 0), resp], 1).to(device),
            "attention_mask": torch.cat([prompt_mask, mask.long()], 1).to(device),
            "response_ids": resp.to(device), "completion_mask": mask.to(device), "prompt_len": enc["input_ids"].shape[1]}


def manual_gradient(model, params, rollouts, advantages, micro=None):
    """Gradient of grpo.grpo_loss (old = current, as in train_multistep.grpo_step) w.r.t. the LoRA params.

    micro: process rows in chunks of this size and weight each chunk loss by its share of rows. Same math, different
    bf16 batch shapes: the numerical noise floor.
    """
    N = rollouts["input_ids"].shape[0]
    adv = advantages.flatten().to(rollouts["input_ids"].device)
    model.zero_grad(set_to_none=True)
    total = 0.0
    for s in range(0, N, micro or N):
        rows = slice(s, min(s + (micro or N), N))
        sub = {k: v[rows] if torch.is_tensor(v) else v for k, v in rollouts.items()}
        with torch.no_grad():
            old = compute_logprobs(model, sub)
        loss = grpo_loss(compute_logprobs(model, sub), old, adv[rows], sub["completion_mask"]) * (rows.stop - rows.start) / N
        loss.backward()
        total += loss.item()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    model.zero_grad(set_to_none=True)
    return grads, total


def make_trainer(model, tok, examples, completions, reward_log, args):
    """A trainer whose generation returns `completions` (prompts still tokenized by TRL itself)."""
    def fixed(prompts, trainer):
        return {"prompt_ids": trainer._tokenize_prompts(prompts)[0], "completion_ids": completions, "logprobs": None}

    return BalancedGRPOTrainer(model=model, reward_funcs=[make_reward(tok, examples, log=reward_log)], args=args,
                               train_dataset=make_dataset(tok, examples), processing_class=tok, rollout_func=fixed,
                               sampler_order=list(range(len(examples))))


def trl_gradient(trainer, model, params, rows):
    """TRL's own pipeline on one generation batch: rewards -> advantages -> masks -> log-probs -> loss -> backward."""
    model.train()
    trainer.current_gradient_accumulation_steps = 1  # set by Trainer.train(); 1 in this configuration
    out = trainer._generate_and_score_completions(rows)
    model.zero_grad(set_to_none=True)
    loss = trainer.compute_loss(model, out)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    model.zero_grad(set_to_none=True)
    return grads, loss.item(), out


def compare(a, b):
    """a = reference. Relative L2 difference, cosine, norm ratio |b|/|a| over all LoRA tensors jointly."""
    va = torch.cat([g.double().flatten() for g in a.values()])
    vb = torch.cat([b[n].double().flatten() for n in a])
    return {"rel_diff": ((va - vb).norm() / (va.norm() + 1e-12)).item(),
            "cosine": torch.nn.functional.cosine_similarity(va, vb, dim=0).item(),
            "norm_ratio": (vb.norm() / va.norm()).item(),
            "per_tensor_rel_diff": {n: ((a[n] - b[n]).double().norm() / (a[n].double().norm() + 1e-12)).item() for n in a}}


def main():
    torch.manual_seed(CONFIG["seed"])
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = lora_model(DTYPE, "cuda", TARGETS[CONFIG["lora"]["targets"]])
    params, _ = check_lora(model)
    examples = train_examples()
    pair = stream(examples, 1)[0][0]  # the first balanced batch: [POSITIVE, NEGATIVE]
    prompts = build_prompts(tok, [e["text"] for e in pair])

    torch.manual_seed(CONFIG["seed"])  # as train_multistep.py before its first rollout
    rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=CONFIG["grpo_config"]["max_completion_length"])
    responses = decode(tok, rollouts)
    rewards = group_rewards(responses, [e["label"] for e in pair], G)
    assert (rewards != rewards[:, :1]).any(), "first batch has no mixed group"
    manual_first = json.loads(MANUAL_TRAINING.read_text())["training"][0]
    completions = completion_lists(rollouts)
    rebuilt = manual_rollouts(tok, prompts, completions, "cuda")
    assert all(torch.equal(rebuilt[k], rollouts[k]) for k in ("input_ids", "attention_mask", "response_ids", "completion_mask"))

    adv_manual, adv_trl = group_advantages(rewards), trl_advantages(rewards)
    g_manual, loss_manual = manual_gradient(model, params, rollouts, adv_manual)
    g_manual_trl_adv, loss_manual_trl_adv = manual_gradient(model, params, rollouts, adv_trl)
    g_manual_micro, _ = manual_gradient(model, params, rollouts, adv_trl, micro=G)
    with torch.no_grad():
        logp_manual = compute_logprobs(model, rollouts)

    reward_log = []
    trainer = make_trainer(model, tok, examples, completions, reward_log, grpo_args(HERE / "checkpoints" / "grad_check_output"))
    by_id = {e["id"]: i for i, e in enumerate(examples)}
    dataset = trainer.train_dataset
    rows = [dataset[by_id[e["id"]]] for e in pair for _ in range(G)]
    g_trl, loss_trl, out = trl_gradient(trainer, model, params, rows)
    with torch.no_grad():
        logp_trl, _, _ = trainer._get_per_token_logps_and_entropies(
            model, torch.cat([out["prompt_ids"], out["completion_ids"]], 1),
            torch.cat([out["prompt_mask"], out["completion_mask"]], 1), out["completion_ids"].shape[1])

    mask = rollouts["completion_mask"]
    assert torch.equal(out["completion_ids"], rollouts["response_ids"]) and torch.equal(out["completion_mask"].bool(), mask)
    assert torch.equal(out["prompt_ids"], rollouts["input_ids"][:, : rollouts["prompt_len"]])
    trl_rewards = torch.tensor(reward_log[0]["rewards"]).reshape(2, G)
    assert torch.equal(trl_rewards, rewards), (trl_rewards, rewards)

    # bf16 noise floor: the same manual math in other row-chunkings, i.e. other bf16 batch shapes.
    g_shapes = {m: manual_gradient(model, params, rollouts, adv_trl, micro=m)[0] for m in (1, 2, 3, 4)} | {G: g_manual_micro}
    bf16_shapes = {f"micro_{m}": {"manual_full_vs_manual_micro": compare(g_manual_trl_adv, g),
                                  "manual_micro_vs_trl": compare(g, g_trl)} for m, g in g_shapes.items()}

    # fp32: the same base weights upcast exactly (bf16 -> fp32), same adapters, same rollouts. Isolates the math.
    del g_shapes
    model.float()
    check_lora(model, dtype=torch.float32)
    g32_manual, _ = manual_gradient(model, params, rollouts, adv_trl, micro=2)  # micro-batched: full fp32 logits do not fit
    g32_manual_m3, _ = manual_gradient(model, params, rollouts, adv_trl, micro=3)
    g32_manual_raw, _ = manual_gradient(model, params, rollouts, adv_manual, micro=2)
    g32_trl, loss32_trl, _ = trl_gradient(trainer, model, params, rows)
    fp32 = {"trl_vs_manual_with_trl_advantages": compare(g32_manual, g32_trl), "trl_vs_manual": compare(g32_manual_raw, g32_trl),
            "noise_floor_manual_micro2_vs_micro3": compare(g32_manual, g32_manual_m3), "trl_loss": loss32_trl}

    result = {
        "model": MODEL, "dtype": str(DTYPE), "prompt_ids": [e["id"] for e in pair], "labels": [e["label"] for e in pair],
        "responses": responses, "rewards": rewards.tolist(),
        "rollouts_equal_manual_attempt_1": responses == manual_first["responses"],
        "advantages": {"manual_population_std": adv_manual.tolist(), "trl_formula_restated": adv_trl.tolist(),
                       "trl_internal": out["advantages"].tolist(),
                       "trl_internal_vs_restated_max_abs": (out["advantages"].cpu() - adv_trl.flatten()).abs().max().item(),
                       "trl_over_manual_per_group": [(adv_trl[b][rewards[b] != rewards[b].mean()] / adv_manual[b][rewards[b] != rewards[b].mean()]).tolist()
                                                     for b in range(2)]},
        "logp_trl_vs_manual_max_abs": (logp_trl - logp_manual)[mask].abs().max().item(),
        "loss": {"manual": loss_manual, "manual_with_trl_advantages": loss_manual_trl_adv, "trl": loss_trl},
        "grad_norms": {k: math.sqrt(sum(g.double().pow(2).sum().item() for g in v.values()))
                       for k, v in {"manual": g_manual, "manual_with_trl_advantages": g_manual_trl_adv, "trl": g_trl}.items()},
        "trl_vs_manual": compare(g_manual, g_trl),
        "trl_vs_manual_with_trl_advantages": compare(g_manual_trl_adv, g_trl),
        "noise_floor_manual_vs_manual_microbatched": compare(g_manual_trl_adv, g_manual_micro),
        "bf16_batch_shapes": bf16_shapes,
        "fp32": fp32,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    out_path = HERE / "results" / "gradient_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f"prompts {result['prompt_ids']} rewards {rewards.tolist()}  rollouts = manual attempt 1: {result['rollouts_equal_manual_attempt_1']}")
    print(f"TRL advantages vs restated formula: max abs {result['advantages']['trl_internal_vs_restated_max_abs']:.2e}; "
          f"TRL/manual advantage ratio per group {result['advantages']['trl_over_manual_per_group']}")
    print(f"per-token log-prob TRL vs manual: max abs {result['logp_trl_vs_manual_max_abs']:.3e}")
    print(f"loss manual {loss_manual:+.6f}  manual(TRL adv) {loss_manual_trl_adv:+.6f}  TRL {loss_trl:+.6f}")
    for k in ("trl_vs_manual", "trl_vs_manual_with_trl_advantages", "noise_floor_manual_vs_manual_microbatched"):
        c = result[k]
        print(f"{k:45} rel diff {c['rel_diff']:.3e}  cos {c['cosine']:.6f}  norm ratio {c['norm_ratio']:.5f}")
    for m, d in bf16_shapes.items():
        a, b = d["manual_full_vs_manual_micro"], d["manual_micro_vs_trl"]
        print(f"bf16 {m:8} manual full vs micro: rel {a['rel_diff']:.3e} cos {a['cosine']:.5f}   micro vs TRL: rel {b['rel_diff']:.3e} cos {b['cosine']:.5f}")
    for k in ("trl_vs_manual_with_trl_advantages", "trl_vs_manual", "noise_floor_manual_micro2_vs_micro3"):
        c = fp32[k]
        print(f"fp32 {k:40} rel diff {c['rel_diff']:.3e}  cos {c['cosine']:.9f}  norm ratio {c['norm_ratio']:.6f}")
    print(f"peak reserved {result['peak_reserved_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
