"""One real GRPO + LoRA step on Qwen3 with the L2L backend, timed, and compared with the resident backend.

    python scripts/l2l_real_step.py --config configs/qwen.yaml --device cuda            # the laptop GPU
    python scripts/l2l_real_step.py --config configs/qwen.yaml --device cpu --batch-size 1 --group-size 2  # smoke

What it does (real weights, the config's dtype / LoRA / batch, attention as the model ships: SDPA, head_dim 128):
  1. L2L: load the model on the CPU, build backend.L2LBackend (WeightStore, pool of GPU buffers, checkpointed layers),
     sample B*G rollouts with the streamed generator (timed per token), score them with the real reward, run
     prepare() and `--iterations` complete training.grpo_update calls (timed), recording log-probs, loss, clipping
     statistics and every LoRA gradient and update, and the peak GPU memory.
  2. Resident (unless --no-compare): load the model on the device, copy the same initial adapters, sample with the same
     seed through model.generate() (the rollouts are compared), then run the same updates on the L2L rollouts and
     compare everything bitwise / by max abs difference.
LoRA B starts at zero in training, which makes every LoRA-A gradient exactly zero; --b-std (default 1e-2) sets B to a
seeded N(0, b_std) on both sides so the gradient comparison is nontrivial (--b-std 0 keeps the training init).
Writes research/l2l/experiments/real_step_<timestamp>.json (or --out) and prints a summary. Nothing under runs/.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from qwen_grpo import config, lab  # noqa: E402
from qwen_grpo.config import DTYPES  # noqa: E402
from qwen_grpo.execution.backend import L2LBackend, ResidentBackend  # noqa: E402
from qwen_grpo.model import EOS_IDS, lora_model, lora_modules  # noqa: E402
from qwen_grpo.sentiment import build_prompts, load_splits  # noqa: E402
from qwen_grpo.training import balanced_subset, decode, group_rewards, grpo_update, prepare  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak(device):
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def init_adapters(model, b_std, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in lora_modules(model):
            w = m.lora_B["default"].weight
            if b_std > 0:
                w.copy_((torch.randn(w.shape, generator=g) * b_std).to(w.dtype))
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapters(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[n].to(p.device))


def run_updates(backend, model, rollouts, rewards, args, device, lr):
    """prepare + `iterations` grpo_update calls; returns per-iteration logs, log-probs, grads, updates; timed."""
    params = backend.trainable()
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.0)
    cap = {"grads": None, "before": None}
    opt.register_step_pre_hook(lambda o, a, k: cap.update(
        grads={n: p.grad.detach().cpu().clone() for n, p in params.items()},
        before={n: p.detach().cpu().clone() for n, p in params.items()}))
    logps = []

    def logprobs(m, r):
        lp = backend.logprobs(m, r)
        logps.append(lp.detach().cpu())
        return lp

    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    targets = prepare(model, rollouts, rewards, args.micro_batch_size, logprobs=backend.logprobs)
    sync(device)
    out = {"prepare_s": time.perf_counter() - t0, "iterations": []}
    if targets is None:
        raise SystemExit("zero-signal batch: every group has equal rewards (use --fixed-rewards)")
    adv, old = targets
    out["old_logp"] = old.detach().cpu()
    for k in range(args.iterations):
        logps.clear()
        sync(device)
        t0 = time.perf_counter()
        log = grpo_update(model, params, opt, rollouts, adv, old, args.clip_eps, k, args.micro_batch_size,
                          logprobs=logprobs)
        sync(device)
        out["iterations"].append({
            "seconds": time.perf_counter() - t0, "log": {k2: v for k2, v in log.items() if k2 != "grad_norms"},
            "logp": torch.cat(logps), "grads": cap["grads"],
            "updates": {n: p.detach().cpu() - cap["before"][n] for n, p in params.items()}})
    out["peak_mib"] = peak(device)
    return out


def diff(a, b):
    a, b = a.double(), b.double()
    return {"max_abs": (a - b).abs().max().item(), "ref_max": a.abs().max().item(), "equal": bool(torch.equal(a, b))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/qwen.yaml", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, help="prompts (default: training.batch_size)")
    ap.add_argument("--group-size", type=int, help="completions per prompt (default: training.group_size)")
    ap.add_argument("--max-new", type=int, default=32, help="new tokens per rollout (timing; training uses its own)")
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--weight-buffers", type=int, default=2)
    ap.add_argument("--storage", choices=["pinned", "pageable"], help="default: pinned on CUDA, pageable on CPU")
    ap.add_argument("--kv-device", choices=["compute", "cpu"], default="compute")
    ap.add_argument("--b-std", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-rewards", action="store_true", help="alternate 1/0 rewards instead of the real reward")
    ap.add_argument("--no-compare", action="store_true", help="skip the resident side")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    cfg = config.load(args.config)
    device = torch.empty(0, device=args.device).device
    dtype = DTYPES[cfg.model.dtype]
    t, lo = cfg.training, cfg.lora
    B, G = args.batch_size or t.batch_size, args.group_size or t.group_size
    args.clip_eps = t.clip_eps
    storage = args.storage or ("pinned" if device.type == "cuda" else "pageable")
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    examples = balanced_subset(load_splits()["train"], max(2, B + B % 2), cfg.data.subset_seed)[:B]
    prompts = build_prompts(tok, [e["text"] for e in examples])
    labels = [e["label"] for e in examples]
    report = {"created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "provenance": lab.provenance(),
              "config": str(args.config), "model": cfg.model.name, "dtype": cfg.model.dtype, "device": str(device),
              "B": B, "G": G, "max_new": args.max_new, "micro_batch_size": args.micro_batch_size,
              "iterations": args.iterations, "storage": storage, "weight_buffers": args.weight_buffers,
              "kv_device": args.kv_device, "lora": {"targets": lo.targets, "rank": lo.rank, "alpha": lo.alpha},
              "b_std": args.b_std, "seed": args.seed}

    # --- L2L ------------------------------------------------------------------------------------------------------
    torch.manual_seed(t.seed)
    model = lora_model(dtype, "cpu", lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
    attn = model.config._attn_implementation
    report["attention"], report["head_dim"] = attn, model.config.head_dim
    adapters = init_adapters(model, args.b_std, args.seed)
    reset_peak(device)
    t0 = time.perf_counter()
    kv = device if args.kv_device == "compute" else torch.device("cpu")
    l2l = L2LBackend(model, device, storage, args.weight_buffers, kv_device=kv)
    report["l2l_setup_s"] = time.perf_counter() - t0
    store = l2l.store
    report["streamed_bytes_per_pass"] = sum(s.nbytes for s in store.slabs)
    report["layers"] = len(store.slabs)

    torch.manual_seed(args.seed)
    l2l.executor.trace.clear()
    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    rollouts = l2l.generate_rollouts(tok, prompts, G, args.max_new)
    sync(device)
    gen_s = time.perf_counter() - t0
    steps = l2l.sampler.last["steps"]
    report["l2l_generation"] = {"seconds": gen_s, "steps": steps, "seconds_per_token_step": gen_s / max(steps, 1),
                                "rows": B * G, "weight_loads": len(l2l.executor.trace),
                                "weight_gb_moved": sum(r["slab_bytes"] for r in l2l.executor.trace) / 1e9,
                                "effective_weight_gb_per_s": sum(r["slab_bytes"] for r in l2l.executor.trace) / 1e9 / gen_s,
                                "kv_cache_mib": l2l.sampler.last["kv_cache_bytes"] / 2**20, "peak_mib": peak(device)}
    responses = decode(tok, rollouts)
    if args.fixed_rewards:
        rewards = torch.tensor([[float((b + g) % 2) for g in range(G)] for b in range(B)])
    else:
        rewards = group_rewards(responses, labels, G)
    if (rewards == rewards[:, :1]).all():
        print("note: the real rewards are zero-signal for this batch; using alternating fixed rewards instead")
        rewards = torch.tensor([[float((b + g) % 2) for g in range(G)] for b in range(B)])
        report["rewards_fixed"] = True
    report["rewards"] = rewards.tolist()
    report["responses"] = responses
    l2l.executor.trace.clear()
    ours = run_updates(l2l, model, rollouts, rewards, args, device, t.learning_rate)
    loads = l2l.executor.trace
    report["l2l_step"] = {"prepare_s": ours["prepare_s"], "update_s": [it["seconds"] for it in ours["iterations"]],
                          "peak_mib": ours["peak_mib"],
                          "forward_loads": sum(r["phase"] == "forward" for r in loads),
                          "backward_loads": sum(r["phase"] == "backward" for r in loads),
                          "logs": [it["log"] for it in ours["iterations"]]}
    cpu_rollouts = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in rollouts.items()}
    l2l.close()
    del model, l2l, rollouts
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- resident ---------------------------------------------------------------------------------------------------
    if not args.no_compare:
        torch.manual_seed(t.seed)
        res_model = lora_model(dtype, device, lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
        load_adapters(res_model, adapters)
        res = ResidentBackend(res_model)
        torch.manual_seed(args.seed)
        reset_peak(device)
        sync(device)
        t0 = time.perf_counter()
        res_rollouts = res.generate_rollouts(tok, prompts, G, args.max_new)
        sync(device)
        report["resident_generation"] = {"seconds": time.perf_counter() - t0, "peak_mib": peak(device)}
        report["rollouts_equal"] = all(torch.equal(res_rollouts[k].cpu(), cpu_rollouts[k])
                                       for k in ("input_ids", "attention_mask", "completion_mask"))
        rollouts = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu_rollouts.items()}
        theirs = run_updates(res, res_model, rollouts, rewards, args, device, t.learning_rate)
        report["resident_step"] = {"prepare_s": theirs["prepare_s"],
                                   "update_s": [it["seconds"] for it in theirs["iterations"]],
                                   "peak_mib": theirs["peak_mib"], "logs": [it["log"] for it in theirs["iterations"]]}
        cmp = {"old_logp": diff(theirs["old_logp"], ours["old_logp"]), "iterations": []}
        for a, b in zip(theirs["iterations"], ours["iterations"]):
            row = {"logp": diff(a["logp"], b["logp"]),
                   "loss": {"resident": a["log"]["loss"], "l2l": b["log"]["loss"], "equal": a["log"]["loss"] == b["log"]["loss"]},
                   "clip_frac_equal": a["log"]["clip_frac"] == b["log"]["clip_frac"],
                   "grads": {n: diff(a["grads"][n], b["grads"][n]) for n in a["grads"]},
                   "updates": {n: diff(a["updates"][n], b["updates"][n]) for n in a["updates"]}}
            row["all_grads_equal"] = all(v["equal"] for v in row["grads"].values())
            row["all_updates_equal"] = all(v["equal"] for v in row["updates"].values())
            row["worst_grad_rel"] = max(v["max_abs"] / v["ref_max"] if v["ref_max"] else 0.0 for v in row["grads"].values())
            row["zero_grads"] = [n for n, v in row["grads"].items() if v["ref_max"] == 0]
            cmp["iterations"].append(row)
        report["comparison"] = cmp

    out = args.out or ROOT / "research" / "l2l" / "experiments" / f"real_step_{report['created_utc']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    g = report["l2l_generation"]
    print(f"model {cfg.model.name} {cfg.model.dtype} attn={attn} head_dim={report['head_dim']} device={device} "
          f"rows={B * G} layers={report['layers']} streamed/pass={report['streamed_bytes_per_pass'] / 2**20:.0f} MiB")
    print(f"L2L generation: {g['steps']} steps, {g['seconds_per_token_step'] * 1000:.1f} ms/step for {B * G} rows, "
          f"{g['effective_weight_gb_per_s']:.2f} GB/s of weights, peak {g['peak_mib']} MiB")
    s = report["l2l_step"]
    print(f"L2L step: prepare {s['prepare_s']:.2f} s, updates {[round(x, 2) for x in s['update_s']]} s, "
          f"loads fwd {s['forward_loads']} / bwd {s['backward_loads']}, peak {s['peak_mib']} MiB")
    if "comparison" in report:
        r = report
        print(f"resident: generation {r['resident_generation']['seconds']:.2f} s, step prepare {r['resident_step']['prepare_s']:.2f} s, "
              f"updates {[round(x, 2) for x in r['resident_step']['update_s']]} s, peak {r['resident_step']['peak_mib']} MiB")
        c = r["comparison"]
        print(f"rollouts equal (same seed): {r['rollouts_equal']}; old logp equal: {c['old_logp']['equal']} "
              f"(max abs {c['old_logp']['max_abs']:.2e})")
        for k, it in enumerate(c["iterations"]):
            print(f"iteration {k}: loss equal {it['loss']['equal']} ({it['loss']['resident']:.6g} vs {it['loss']['l2l']:.6g}), "
                  f"logp max abs {it['logp']['max_abs']:.2e}, all grads equal {it['all_grads_equal']} "
                  f"(worst rel {it['worst_grad_rel']:.2e}), all updates equal {it['all_updates_equal']}, "
                  f"clip_frac equal {it['clip_frac_equal']}, zero-gradient tensors {len(it['zero_grads'])}")
    print(f"written: {out}")
    return report


if __name__ == "__main__":
    main()
