"""One diagnostic GRPO step on Qwen3-0.6B (one fp32 trainable matrix in a bf16 model), plus an LR=0 control."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from qwen_grpo.grpo import group_advantages, grpo_loss
from qwen_grpo.model import DTYPE, EOS_IDS, MODEL
from qwen_grpo.policy import compute_logprobs, generate_rollouts
from qwen_grpo.sentiment import build_prompts, load_splits, reward
from qwen_grpo.training import INIT_TOL, decode, masked_max

B, G = 1, 6
MAX_NEW_TOKENS, LEARNING_RATE, CLIP_EPS = 8, 1e-5, 0.2
SEED, MAX_ATTEMPTS = 0, 5
TRAINABLE = "model.layers.27.self_attn.o_proj.weight"


def make_fp32_trainable(model, name):
    """Freeze everything; keep `name`'s Linear in fp32, computing it in fp32 between bf16 neighbours.

    A plain cast fails (F.linear needs equal dtypes), and autocast would round the weight back to bf16.
    """
    for n, p in model.named_parameters():
        p.requires_grad_(n == name)
    linear = model.get_submodule(name.removesuffix(".weight"))
    linear.float()
    linear.register_forward_pre_hook(lambda m, args: (args[0].float(),))
    linear.register_forward_hook(lambda m, args, out: out.to(DTYPE))
    return linear.weight


def grpo_step(model, param, lr, rollouts, old_logp, adv):
    """One AdamW step on `param`; returns the measurements, leaves the step applied."""
    optimizer = torch.optim.AdamW([param], lr=lr, weight_decay=0.0)
    assert [p for group in optimizer.param_groups for p in group["params"]] == [param]
    before = param.detach().clone()
    mask = rollouts["completion_mask"]

    new_logp = compute_logprobs(model, rollouts)
    init_diff = masked_max(new_logp.detach() - old_logp, mask)
    if init_diff > INIT_TOL:
        raise RuntimeError(f"old/current log-prob mismatch before the step: {init_diff:.3e}")
    ratio = (new_logp.detach() - old_logp)[mask].exp()

    optimizer.zero_grad(set_to_none=True)
    loss = grpo_loss(new_logp, old_logp, adv, mask, clip_eps=CLIP_EPS)
    loss.backward()
    grad = param.grad.detach().clone()
    grad_norm = grad.norm().item()
    if not (torch.isfinite(grad).all() and grad_norm > 0):
        raise RuntimeError(f"gradient is zero or non-finite (norm {grad_norm}); no update performed")
    optimizer.step()

    with torch.no_grad():
        after_logp = compute_logprobs(model, rollouts)
        loss_after = grpo_loss(after_logp, old_logp, adv, mask, clip_eps=CLIP_EPS).item()
    dlogp = torch.where(mask, after_logp - old_logp, 0.0)
    delta_i = dlogp.sum(1) / mask.sum(1)  # mean valid-token log-prob change per completion

    # Independent reference: first AdamW step from zero state, bias-corrected: -lr * g / (|g| + eps).
    eps = optimizer.param_groups[0]["eps"]
    expected = -lr * grad / (grad.abs() + eps)
    actual = param.detach() - before
    return {
        "init_diff": init_diff, "ratio": (ratio.min().item(), ratio.max().item()),
        "grad": grad, "grad_norm": grad_norm, "grad_dtype": param.grad.dtype,
        "loss": loss.item(), "loss_after": loss_after,
        "delta_i": delta_i, "S": (adv * delta_i).mean().item(), "logp_delta": masked_max(dlogp, mask),
        "param_delta": actual.norm().item(), "changed_frac": (actual != 0).float().mean().item(),
        "upd_max_err": (actual - expected).abs().max().item(),
        "upd_rel_err": ((actual - expected).norm() / expected.norm()).item() if lr else float("nan"),
        "upd_cos": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item() if lr else float("nan"),
        "upd_tol": 2 * torch.finfo(torch.float32).eps * before.abs().max().item(),
        "expected_nonzero": (expected != 0).float().mean().item(), "before": before,
    }


def main():
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).cuda().eval()
    model.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)

    param = make_fp32_trainable(model, TRAINABLE)
    frozen = {n: p for n, p in model.named_parameters() if n != TRAINABLE}
    assert [n for n, p in model.named_parameters() if p.requires_grad] == [TRAINABLE]
    assert param.dtype == torch.float32 and all(p.dtype == DTYPE for p in frozen.values())
    frozen_before = {n: p.detach().cpu().clone() for n, p in frozen.items()}

    example = load_splits()["train"][0]
    prompts = build_prompts(tok, [example["text"]])
    print(f"Model: {MODEL}\nFrozen precision: {DTYPE} ({len(frozen)} tensors)\nPrompts: {B}\nGroup size: {G}")
    print(f"Trainable: {TRAINABLE} {tuple(param.shape)} {param.dtype} = {param.numel():,} "
          f"of {sum(p.numel() for p in model.parameters()):,}")
    print(f"Example (train[0]): {example['text']!r}  label={example['label']}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=MAX_NEW_TOKENS)
        responses = decode(tok, rollouts)
        rewards = torch.tensor([reward(r, example["label"]) for r in responses]).reshape(B, G)
        if (rewards != rewards[:, :1]).any(dim=1).all():
            break
    else:
        print(f"\nNo mixed-reward group in {MAX_ATTEMPTS} attempts; no optimizer step performed.")
        return
    advantages = group_advantages(rewards).reshape(B * G).cuda()

    with torch.no_grad():
        old_logp = compute_logprobs(model, rollouts)

    real = grpo_step(model, param, LEARNING_RATE, rollouts, old_logp, advantages)
    frozen_changed = [n for n, p in frozen.items() if not torch.equal(p.detach().cpu(), frozen_before[n])]

    with torch.no_grad():
        param.copy_(real["before"])  # identical initial weights for the control
    control = grpo_step(model, param, 0.0, rollouts, old_logp, advantages)
    frozen_changed += [n for n, p in frozen.items() if not torch.equal(p.detach().cpu(), frozen_before[n])]

    torch.cuda.synchronize()
    alloc, reserved = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30

    print(f"\nRollouts (attempt {attempt} of at most {MAX_ATTEMPTS}):")
    for r, rew, a, d in zip(responses, rewards.flatten(), advantages, real["delta_i"]):
        print(f"  {r!r:12} reward={rew.item():.0f}  A={a.item():+.4f}  mean dlogp={d.item():+.4e}")
    print(f"\nInitial max logprob difference: {real['init_diff']:.3e}")
    print(f"Initial ratio range: [{real['ratio'][0]:.6f}, {real['ratio'][1]:.6f}]")
    print(f"Gradient norm: {real['grad_norm']:.6e}  (dtype {real['grad_dtype']})")
    print(f"\nReal step (lr={LEARNING_RATE}):")
    print(f"  Loss before: {real['loss']:+.6e}\n  Loss after:  {real['loss_after']:+.6e}")
    print(f"  Loss difference: {real['loss_after'] - real['loss']:+.6e}")
    print(f"  Advantage-weighted score S: {real['S']:+.6e}")
    print(f"  Parameter delta: {real['param_delta']:.6e}")
    print(f"  Elements changed: {real['changed_frac']:.2%}  (nonzero in expected update: {real['expected_nonzero']:.2%})")
    print(f"  Update vs expected AdamW: max abs err {real['upd_max_err']:.3e} (tol {real['upd_tol']:.3e}), "
          f"rel L2 err {real['upd_rel_err']:.3e}, cosine {real['upd_cos']:.8f}")
    print(f"  Policy logprob delta (max |.|): {real['logp_delta']:.6e}")
    print(f"  Frozen parameters changed: {len(set(frozen_changed))} of {len(frozen)} tensors (all compared, after both runs)")
    print("\nLR=0 control (same initial weights, rollouts, advantages, old log-probs):")
    print(f"  Initial max logprob difference: {control['init_diff']:.3e}")
    print(f"  Gradient norm: {control['grad_norm']:.6e}  (bitwise equal to real step: {torch.equal(control['grad'], real['grad'])})")
    print(f"  Parameter delta: {control['param_delta']:.6e}")
    print(f"  Policy logprob delta: {control['logp_delta']:.6e}   loss after - before: {control['loss_after'] - control['loss']:+.3e}")
    print(f"\nPeak GPU allocated: {alloc:.2f} GiB\nPeak GPU reserved: {reserved:.2f} GiB  (device total {total:.2f} GiB)")

    checks = {
        "trainable param and grad fp32": param.dtype == real["grad_dtype"] == torch.float32,
        "frozen params bf16 and unchanged": not frozen_changed and all(p.dtype == DTYPE for p in frozen.values()),
        "initial old/current agree": real["init_diff"] <= INIT_TOL,
        "gradient finite and nonzero": real["grad_norm"] > 0,
        "update matches expected AdamW": real["upd_max_err"] <= real["upd_tol"],
        "policy log-probs changed": real["logp_delta"] > 0,
        "GRPO objective improved (loss decreased)": real["loss_after"] < real["loss"],
        "advantage-weighted score S > 0": real["S"] > 0,
        "control: gradient equals real step": torch.equal(control["grad"], real["grad"]),
        "control: parameter unchanged": control["param_delta"] == 0,
        "control: policy unchanged": control["logp_delta"] == 0,
        "peak reserved within 4 GiB device": reserved < total,
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print(f"\nResult: {'PASS' if all(checks.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
