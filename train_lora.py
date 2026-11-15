"""One diagnostic GRPO step with LoRA (fp32) on the last o_proj of a frozen bf16 Qwen3-0.6B, plus controls.

Same setup as train_grpo.py (the full-rank reference).
"""
import hashlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from baseline import EOS_IDS, MODEL, build_prompts
from grpo import group_advantages, grpo_loss
from policy import compute_logprobs, generate_rollouts
from sentiment import load_splits, reward
from train_grpo import B, CLIP_EPS, DTYPE, G, INIT_TOL, LEARNING_RATE, MAX_ATTEMPTS, MAX_NEW_TOKENS, SEED, decode, masked_max

GPU_BUDGET_GIB = 4.0  # local stand-in for the assignment's 8 GB cap
TARGET = r"model\.layers\.27\.self_attn\.o_proj"  # PEFT re.fullmatch-es a string against full module names
RANK, ALPHA = 8, 16


def lora_config(target=TARGET):
    return LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=target)


def lora_model(dtype, device, target=TARGET):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
    return get_peft_model(base, lora_config(target)).eval()


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def seq_mean(x, mask):
    return torch.where(mask, x, 0.0).sum(1) / mask.sum(1)


def verdict(value, uncertainty, good_if_negative):
    """PASS / FAIL only if |value| exceeds its uncertainty; a zero uncertainty means none was available."""
    if uncertainty <= 0 or abs(value) <= uncertainty:
        return "INCONCLUSIVE"
    return "PASS" if (value < 0) == good_if_negative else "FAIL"


def rowwise_spread(model, rollouts, logp):
    """max |batched - one-row-at-a-time| log-prob: identical math, different bf16 rounding path."""
    with torch.no_grad():
        rows = [compute_logprobs(model, {k: v[i : i + 1] if torch.is_tensor(v) else v for k, v in rollouts.items()})
                for i in range(len(logp))]
    return masked_max(torch.cat(rows) - logp, rollouts["completion_mask"])


def adapter_contribution(model, module, rollouts):
    """Trained adapter's effect on the target layer's output for one fixed captured input x [N, T, 2048]."""
    xs = []
    hook = module.register_forward_pre_hook(lambda m, args: xs.append(args[0].detach()))
    with torch.no_grad():
        compute_logprobs(model, rollouts)
    hook.remove()
    x = xs[0]
    with torch.no_grad():
        h_on, h_off = module(x), module.base_layer(x)  # [N, T, 1024] bf16
        A, Bm = module.lora_A["default"].weight, module.lora_B["default"].weight  # [r, 2048], [1024, r]
        d64 = (ALPHA / RANK) * (x.double() @ A.double().T) @ Bm.double().T  # theory: (a/r) B A x, row-vector form
        d32 = module.lora_B["default"](module.lora_A["default"](x.float())) * module.scaling["default"]  # PEFT's fp32 path
        # Higham: |fl(xA^T B^T) - xA^T B^T| <= gamma_{K+r} |x||A|^T|B|^T, gamma_n ~ n u, u = 2^-24
        bound = (2048 + RANK) * 2.0**-24 * (ALPHA / RANK) * (x.double().abs() @ A.double().abs().T) @ Bm.double().abs().T
        emulated = (h_off.float() + d32).to(h_off.dtype)  # PEFT: bf16 base + fp32 lora, cast back to bf16
    obs = h_on.double() - h_off.double()
    pred = emulated.double() - h_off.double()
    nz = pred != 0
    return {
        "theory_norm": d64.norm().item(), "theory_max": d64.abs().max().item(),
        "fp32_err": (d32.double() - d64).abs().max().item(), "fp32_within_bound": bool(((d32.double() - d64).abs() <= bound).all()),
        "obs_norm": obs.norm().item(), "emulation_exact": torch.equal(h_on, emulated),
        "pred_changed": int(nz.sum()), "obs_changed": int((obs != 0).sum()), "numel": obs.numel(),
        "signs_agree": bool((obs[nz].sign() == d64[nz].sign()).all()),
    }


def lora_step(model, params, lr, rollouts, old_logp, adv):
    """One AdamW step on the LoRA params; returns measurements, leaves the step applied."""
    optimizer = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.0)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in params.values()}
    before = {n: p.detach().clone() for n, p in params.items()}
    mask = rollouts["completion_mask"]

    new_logp = compute_logprobs(model, rollouts)
    init_diff = masked_max(new_logp.detach() - old_logp, mask)
    if init_diff > INIT_TOL:
        raise RuntimeError(f"old/current log-prob mismatch before the step: {init_diff:.3e}")
    ratio = (new_logp.detach() - old_logp)[mask].exp()

    # Per-completion gradients of mean log-prob, for a first-order fp32 prediction of Delta_i.
    seq = seq_mean(new_logp, mask)
    g_seq = [torch.autograd.grad(seq[i], list(params.values()), retain_graph=True) for i in range(len(seq))]

    optimizer.zero_grad(set_to_none=True)
    loss = grpo_loss(new_logp, old_logp, adv, mask, clip_eps=CLIP_EPS)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    if not all(torch.isfinite(g).all() for g in grads.values()) or all(g.norm() == 0 for g in grads.values()):
        raise RuntimeError("LoRA gradients are non-finite or all zero; no update performed")
    optimizer.step()

    with torch.no_grad():
        after_logp = compute_logprobs(model, rollouts)
        loss_after = grpo_loss(after_logp, old_logp, adv, mask, clip_eps=CLIP_EPS).item()
    delta = {n: p.detach() - before[n] for n, p in params.items()}  # actual applied update, fp32
    gd = [(grads[n] * delta[n]) for n in params]
    pred_i = torch.stack([sum((g * delta[n]).sum() for g, n in zip(gs, params)) for gs in g_seq])
    delta_i = seq_mean(after_logp - old_logp, mask)
    return {
        "init_diff": init_diff, "ratio": (ratio.min().item(), ratio.max().item()),
        "grads": grads, "loss": loss.item(), "loss_after": loss_after,
        "D": sum(x.sum() for x in gd).item(),  # grad L . Delta theta
        "D_err": sum(x.numel() for x in gd) * 2.0**-24 * sum(x.abs().sum() for x in gd).item(),
        "grad_norm": sum(g.pow(2).sum() for g in grads.values()).sqrt().item(),
        "step_norm": sum(d.pow(2).sum() for d in delta.values()).sqrt().item(),
        "delta_i": delta_i, "S": (adv * delta_i).mean().item(),
        "pred_i": pred_i, "S_pred": (adv * pred_i).mean().item(),
        "param_delta": {n: d.norm().item() for n, d in delta.items()},
        "logp_delta": masked_max(after_logp - old_logp, mask), "after_logp": after_logp, "before": before,
    }


def main():
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = lora_model(DTYPE, "cuda")

    lora_modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)]
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    assert lora_modules == ["base_model.model.model.layers.27.self_attn.o_proj"], lora_modules
    assert sorted(tuple(p.shape) for p in params.values()) == [(RANK, 2048), (1024, RANK)]
    assert sum(p.numel() for p in params.values()) == RANK * (2048 + 1024) == 24_576
    assert all(p.dtype == torch.float32 for p in params.values()), {n: p.dtype for n, p in params.items()}
    assert all(p.dtype == DTYPE for p in frozen.values())
    name_A, name_B = (next(n for n in params if f"lora_{x}." in n) for x in "AB")
    assert params[name_B].abs().max() == 0 and params[name_A].abs().max() > 0  # PEFT default init
    frozen_digest = {n: digest(p) for n, p in frozen.items()}  # SHA-256 of raw bytes, every frozen tensor
    W0 = model.get_submodule(lora_modules[0]).base_layer.weight
    W0_before = W0.detach().cpu().clone()  # the one 1024x2048 bf16 matrix the merge modifies

    example = load_splits()["train"][0]
    prompts = build_prompts(tok, [example["text"]])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=MAX_NEW_TOKENS)
        responses = decode(tok, rollouts)
        rewards = torch.tensor([reward(r, example["label"]) for r in responses]).reshape(B, G)
        if (rewards != rewards[:, :1]).any(dim=1).all():
            break
    else:
        print(f"No mixed-reward group in {MAX_ATTEMPTS} attempts; no optimizer step performed.")
        return
    adv = group_advantages(rewards).reshape(B * G).cuda()
    mask = rollouts["completion_mask"]

    with torch.no_grad():
        old_logp = compute_logprobs(model, rollouts)
        with model.disable_adapter():
            base_logp = compute_logprobs(model, rollouts)

    spread = rowwise_spread(model, rollouts, old_logp)

    real = lora_step(model, params, LEARNING_RATE, rollouts, old_logp, adv)
    with torch.no_grad():
        with model.disable_adapter():
            off_logp = compute_logprobs(model, rollouts)
    layer = adapter_contribution(model, model.get_submodule(lora_modules[0]), rollouts)  # trained adapter, unmerged
    trained = {n: p.detach().clone() for n, p in params.items()}
    dW_eff = (ALPHA / RANK) * trained[name_B] @ trained[name_A]  # fp32 effective weight update
    changed = lambda: sorted(n for n, p in frozen.items() if digest(p) != frozen_digest[n])
    frozen_changed = changed()

    with torch.no_grad():
        for n, p in params.items():
            p.copy_(real["before"][n])  # identical initial LoRA weights for the control
    control = lora_step(model, params, 0.0, rollouts, old_logp, adv)
    frozen_changed += changed()

    with torch.no_grad():
        for n, p in params.items():
            p.copy_(trained[n])
        model.merge_adapter()  # in place into the bf16 base weight
        merged_logp = compute_logprobs(model, rollouts)
    torch.cuda.synchronize()
    alloc, reserved = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
    free, total = (b / 2**30 for b in torch.cuda.mem_get_info())
    merged_dW = W0.detach().cpu().float() - W0_before.float()
    intended = dW_eff.cpu()

    print(f"Model: {MODEL}\nBase dtype: {DTYPE}\nLoRA dtype: {params[name_A].dtype}")
    print(f"LoRA target: {lora_modules[0].removeprefix('base_model.model.')}\nRank: {RANK}\nAlpha: {ALPHA}")
    print(f"Trainable parameters: {sum(p.numel() for p in params.values()):,}  "
          + ", ".join(f"{n.split('.')[-3]} {tuple(p.shape)}" for n, p in params.items()))
    print(f"\nExample (train[0]): {example['text']!r}  label={example['label']}  (mixed group on attempt {attempt})")
    for r, rew, a, d, p in zip(responses, rewards.flatten(), adv, real["delta_i"], real["pred_i"]):
        print(f"  {r!r:12} reward={rew.item():.0f}  A={a.item():+.4f}  Delta_i bf16={d.item():+.3e}  first-order fp32={p.item():+.3e}")
    print(f"\nInitial max logprob difference: {real['init_diff']:.3e}  (adapter-on vs disabled at init: {masked_max(old_logp - base_logp, mask):.3e})")
    print(f"Initial ratio range: [{real['ratio'][0]:.6f}, {real['ratio'][1]:.6f}]")
    print(f"Gradient norm A: {real['grads'][name_A].norm().item():.6e}\nGradient norm B: {real['grads'][name_B].norm().item():.6e}  (dtype {real['grads'][name_B].dtype})")
    print(f"\nLoRA A parameter delta: {real['param_delta'][name_A]:.6e}\nLoRA B parameter delta: {real['param_delta'][name_B]:.6e}")
    print(f"Effective update ||(a/r) B A||: {dW_eff.norm().item():.6e}  max |.|: {dW_eff.abs().max().item():.3e}")
    print(f"Frozen parameters changed: {len(set(frozen_changed))} of {len(frozen)} (SHA-256 digest of every frozen tensor, before merge)")

    print("\nFirst-order descent check (fp32 grads from the bf16 forward, actual applied update):")
    cos = real["D"] / (real["grad_norm"] * real["step_norm"])
    print(f"  D = grad L . Delta theta: {real['D']:+.6e}  (fp32 rounding bound {real['D_err']:.1e})")
    print(f"  ||grad L||: {real['grad_norm']:.6e}   ||Delta theta||: {real['step_norm']:.6e}   cosine: {cos:+.4f}")
    print(f"  note: D = -S_first-order at ratio 1 (same quantity), and AdamW step 1 gives D <= 0 for any lr > 0")

    U = 2 * adv.abs().mean().item() * spread  # |dL|, |dS| <= mean|A| * eps per evaluation, two evaluations
    dL = real["loss_after"] - real["loss"]
    print("\nBF16 objective (same rollouts, re-evaluated after the step):")
    print(f"  Loss before: {real['loss']:+.6e}\n  Loss after:  {real['loss_after']:+.6e}\n  Loss difference: {dL:+.6e}")
    print(f"  Advantage-weighted score S: {real['S']:+.6e}  (first-order fp32: {real['S_pred']:+.3e})")
    print(f"  bf16 path spread eps (batched vs row-wise log-prob, max): {spread:.3e}")
    print(f"  uncertainty estimate U = 2 mean|A| eps = {U:.3e}  (an estimate of bf16 evaluation error, not a bound)")

    print("\nLayer-level adapter contribution (trained adapter, fixed captured input to layer-27 o_proj):")
    print(f"  theory (a/r) B A x in fp64: ||.|| = {layer['theory_norm']:.3e}, max |.| = {layer['theory_max']:.3e}")
    print(f"  PEFT fp32 LoRA path vs fp64 theory: max err {layer['fp32_err']:.2e}, within Higham bound: {layer['fp32_within_bound']}")
    print(f"  observed bf16 output change: ||.|| = {layer['obs_norm']:.3e}, {layer['obs_changed']} of {layer['numel']} elements changed")
    print(f"  emulated bf16 rounding of (h_off + fp32 delta) predicts {layer['pred_changed']} changed; "
          f"bitwise equal to observed: {layer['emulation_exact']}; signs agree with theory: {layer['signs_agree']}")

    print("\nSub-resolution bf16 log-prob measurements (reported, not counted in verdicts):")
    on_off = masked_max(real["after_logp"] - off_logp, mask)
    print(f"  Policy logprob delta: {real['logp_delta']:.6e}\n  Adapter on/off difference: {on_off:.6e}")
    print(f"  Disabled vs original base: {masked_max(off_logp - base_logp, mask):.6e}")

    print("\nMerge (bf16, in place):")
    print(f"  max |logp_adapter - logp_merged|: {masked_max(real['after_logp'] - merged_logp, mask):.6e}")
    print(f"  ||intended fp32 dW||: {intended.norm().item():.3e}   ||realized bf16 merged dW||: {merged_dW.norm().item():.3e}")
    print(f"  ||realized - intended||: {(merged_dW - intended).norm().item():.3e}  (relative {(merged_dW - intended).norm().item() / intended.norm().item():.3f}); "
          f"{(merged_dW != 0).float().mean().item():.4%} of elements changed")
    print("  Limitation: identical bf16 log-probs do not imply weight-level parity; the bf16 merge rounds away the part of the update below half an ulp of W0.")

    print("\nLR=0 control (same initial LoRA weights, rollouts, advantages, old log-probs):")
    print(f"  Gradient agreement: {all(torch.equal(control['grads'][n], real['grads'][n]) for n in params)} (bitwise)")
    print(f"  Parameter delta: {max(control['param_delta'].values()):.6e}\n  Policy delta: {control['logp_delta']:.6e}\n  D: {control['D']:+.6e}")

    within = reserved <= GPU_BUDGET_GIB
    print(f"\nPeak allocated: {alloc:.2f} GiB\nPeak reserved: {reserved:.2f} GiB\nPhysical device VRAM: {total:.2f} GiB")
    print(f"Configured GPU budget: {GPU_BUDGET_GIB:.2f} GiB\nWithin configured budget: {'PASS' if within else 'FAIL'}")
    print(f"  note: allocator stats exclude the CUDA context and non-PyTorch allocations; device-wide in use now "
          f"(all processes): {total - free:.2f} GiB")

    mechanical = {
        "exactly one LoRA projection, LoRA fp32, base bf16": len(lora_modules) == 1,  # dtypes asserted above
        "finite gradients; grad A = 0 (B = 0 at init), grad B nonzero":
            real["grads"][name_A].norm() == 0 and real["grads"][name_B].norm() > 0,
        "B updated, A unchanged": real["param_delta"][name_B] > 0 and real["param_delta"][name_A] == 0,
        "frozen weights unchanged (SHA-256, all tensors)": not frozen_changed,
        "disabling the adapter restores base log-probs": masked_max(off_logp - base_logp, mask) == 0,
        "layer: PEFT fp32 contribution = (a/r) B A x within bound": layer["fp32_within_bound"],
        "layer: bf16 output = rounding of (h_off + delta), signs agree": layer["emulation_exact"] and layer["signs_agree"],
        "LR=0: same grads, no param/policy change, D = 0":
            all(torch.equal(control["grads"][n], real["grads"][n]) for n in params)
            and max(control["param_delta"].values()) == 0 and control["logp_delta"] == 0 and control["D"] == 0,
    }
    print()
    for name, ok in mechanical.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if layer["pred_changed"] == 0:
        print("  [INCONCLUSIVE] layer check non-discriminating: no output element predicted to change in bf16")
    A_v = "FAIL" if not all(mechanical.values()) else "INCONCLUSIVE" if layer["pred_changed"] == 0 else "PASS"
    B_v = verdict(real["D"], real["D_err"], good_if_negative=True)
    C_v = verdict(dL, U, good_if_negative=True)
    if C_v != "INCONCLUSIVE" and verdict(real["S"], U, good_if_negative=False) != C_v:
        C_v = "INCONCLUSIVE"  # loss and S must agree
    print(f"\nA. Mechanical correctness: {A_v}")
    print(f"B. Optimizer descent direction (D < 0): {B_v}")
    print(f"C. BF16 objective improvement: {C_v}")
    print(f"GPU budget: {'PASS' if within else 'FAIL'}")


if __name__ == "__main__":
    main()
