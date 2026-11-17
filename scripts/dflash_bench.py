"""Ticket 12 / H10: plain streamed generation (A) vs DFlash-assisted streamed generation (B) on Qwen3-4B with the L2L
backend, at several LoRA policies, under the profiler and the 8 GiB memory guard.

    python scripts/dflash_bench.py --config configs/experiments/qwen3_4b_l2l_dflash.yaml --max-new 4 32

Policies: LoRA B scaled to --b-scales (0 = the base policy the drafter was trained on, then increasingly shifted
policies; seeded) or real adapters via --adapters; the drafter never changes, so acceptance shows how it degrades as
the policy moves away from the base. Per policy and max_new:
  greedy    A and B on the same prompts: token-by-token equality (must hold exactly; B's verifier is the LoRA policy),
            wall time, target passes, accepted drafts per pass, target passes per generated token
  sampling  A and B (temperature 1): wall time, acceptance (distribution equality is tested exactly on tiny models in
            tests/test_dflash.py; here only statistics)
  GRPO step rollout (sampling) + prepare + one grpo_update, with A and with B as the rollout generator
Peak GPU memory per part (allocator) and the profiler's nvidia-smi samples; verdict on the budget as in
scripts/profile_run.py (never PASS without assignment-grade evidence).
Writes research/l2l/experiments/<--out or t12_dflash_bench>.json.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qwen_grpo import config, lab  # noqa: E402
from qwen_grpo.config import DTYPES  # noqa: E402
from qwen_grpo.execution.backend import make_backend  # noqa: E402
from qwen_grpo.execution.dflash import DFlashGenerator  # noqa: E402
from qwen_grpo.execution.generator import StreamedGenerator  # noqa: E402
from qwen_grpo.execution.profiler import Profiler  # noqa: E402
from qwen_grpo.model import EOS_IDS, lora_model, lora_modules  # noqa: E402
from qwen_grpo.sentiment import build_prompts, load_splits  # noqa: E402
from qwen_grpo.training import balanced_subset, decode, group_rewards, grpo_update, prepare  # noqa: E402


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--b-scales", type=float, nargs="*", default=[0.0, 0.01, 0.03])
    ap.add_argument("--adapters", type=Path, nargs="*", default=[])
    ap.add_argument("--max-new", type=int, nargs="*", default=[4, 32])
    ap.add_argument("--budget-gib", type=float, default=8.0)
    ap.add_argument("--out", default="t12_dflash_bench")
    ap.add_argument("--out-dir", type=Path, help="directory of the JSON result (default research/l2l/experiments; "
                    "qwen-grpo preflight --gpu writes under runs/_preflight/)")
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    if not cfg.dflash.enabled:
        raise SystemExit("the config must enable dflash")
    t = cfg.training
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    rep = {"config": str(a.config), "model": cfg.model.name, "drafter": cfg.dflash.draft_model,
           "execution": dataclasses.asdict(cfg.execution), "B": t.batch_size, "G": t.group_size,
           "provenance": lab.provenance(), "policies": []}
    device = torch.device("cuda")
    with Profiler(int(a.budget_gib * 2**30)) as prof:
        try:
            with prof.phase("setup"):
                torch.manual_seed(t.seed)
                model = lora_model(DTYPES[cfg.model.dtype], "cpu", cfg.lora.spec(), cfg.lora.rank, cfg.lora.alpha,
                                   name=cfg.model.name)
                be = make_backend(cfg, model, device)
            dfl = be.sampler
            assert isinstance(dfl, DFlashGenerator), "the backend did not attach the DFlash generator"
            rep["drafter_bytes"] = dfl.drafter.nbytes()
            plain_s = StreamedGenerator(be.executor, dfl.kv_device, do_sample=True)
            plain_g = StreamedGenerator(be.executor, dfl.kv_device, do_sample=False)
            dfl_g = DFlashGenerator(dfl.drafter, be.executor, dfl.kv_device, do_sample=False)
            examples = balanced_subset(load_splits()["train"], max(2, t.batch_size + t.batch_size % 2),
                                       cfg.data.subset_seed)[: t.batch_size]
            prompts = build_prompts(tok, [e["text"] for e in examples])
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            params = be.trainable()
            init = {n: p.detach().clone() for n, p in params.items()}
            policies = [("b_scale", s) for s in a.b_scales] + [("adapter", str(p)) for p in a.adapters]
            for kind, value in policies:
                with torch.no_grad():
                    for n, p in params.items():
                        p.copy_(init[n])
                    if kind == "b_scale":
                        g = torch.Generator().manual_seed(0)
                        for _, m in lora_modules(model):
                            w = m.lora_B["default"].weight
                            w.copy_((torch.randn(w.shape, generator=g) * value).to(w.dtype).to(w.device))
                    else:
                        from qwen_grpo.checkpointing import load_adapter
                        load_adapter(model, value)
                row = {"policy": {kind: value}, "runs": []}
                for max_new in a.max_new:
                    r = {"max_new": max_new}
                    call = lambda gen: gen(model, enc["input_ids"], enc["attention_mask"], t.group_size, max_new,
                                           EOS_IDS, tok.pad_token_id)
                    with prof.phase(f"greedy_plain_{max_new}"):
                        sa, r["greedy_plain_s"], r["greedy_plain_peak"] = timed(lambda: call(plain_g))
                    with prof.phase(f"greedy_dflash_{max_new}"):
                        sb, r["greedy_dflash_s"], r["greedy_dflash_peak"] = timed(lambda: call(dfl_g))
                    st = dfl_g.last
                    r["greedy_equal"] = bool(torch.equal(sa, sb))
                    r["greedy_target_passes"], r["greedy_steps"] = st["target_passes"], st["steps"]
                    r["greedy_accepted_per_pass"] = st["accepted_per_pass"]
                    r["greedy_passes_per_token"] = st["target_passes"] / max(st["steps"], 1)
                    with prof.phase(f"sample_plain_{max_new}"):
                        _, r["sample_plain_s"], r["sample_plain_peak"] = timed(lambda: call(plain_s))
                    with prof.phase(f"sample_dflash_{max_new}"):
                        _, r["sample_dflash_s"], r["sample_dflash_peak"] = timed(lambda: call(dfl))
                    sd = dfl.last
                    r["sample_accepted_per_pass"], r["sample_passes_per_token"] = (
                        sd["accepted_per_pass"], sd["target_passes"] / max(sd["steps"], 1))
                    row["runs"].append(r)
                    print(f"{kind}={value} max_new={max_new}: greedy equal {r['greedy_equal']}, passes/token "
                          f"{r['greedy_passes_per_token']:.2f} (greedy) {r['sample_passes_per_token']:.2f} (sampling), "
                          f"greedy {r['greedy_plain_s']:.1f}s -> {r['greedy_dflash_s']:.1f}s, sampling "
                          f"{r['sample_plain_s']:.1f}s -> {r['sample_dflash_s']:.1f}s")
                # one full GRPO step with each rollout generator (rollout + prepare + update), same policy start
                for name, gen in (("plain", plain_s), ("dflash", dfl)):
                    with torch.no_grad():
                        snapshot = {n: p.detach().clone() for n, p in params.items()}
                    opt = torch.optim.AdamW(params.values(), lr=t.learning_rate, weight_decay=0.0)
                    be.sampler = gen

                    def step():
                        r_ = be.generate_rollouts(tok, prompts, t.group_size, t.max_new_tokens)
                        rw = group_rewards(decode(tok, r_), [e["label"] for e in examples], t.group_size)
                        if (rw == rw[:, :1]).all():
                            rw = torch.tensor([[float((b + k) % 2) for k in range(t.group_size)]
                                               for b in range(t.batch_size)])
                        adv, old = prepare(model, r_, rw, t.micro_batch_size, logprobs=be.logprobs)
                        return grpo_update(model, params, opt, r_, adv, old, t.clip_eps, 0, t.micro_batch_size,
                                           logprobs=be.logprobs)
                    with prof.phase(f"grpo_step_{name}"):
                        _, row[f"grpo_step_{name}_s"], row[f"grpo_step_{name}_peak"] = timed(step)
                    with torch.no_grad():
                        for n, p in params.items():
                            p.copy_(snapshot[n])
                be.sampler = dfl
                rep["policies"].append(row)
        except KeyboardInterrupt:
            rep["stopped_by_guard"] = bool(prof.violations)
            if not prof.violations:
                raise
    rep["profile"] = prof.report()
    ev, viol = rep["profile"]["evidence"], rep["profile"]["violations"]
    rep["budget_verdict"] = "FAIL" if viol else ("PASS" if ev == "assignment" else "INCOMPLETE_EVIDENCE")
    rep["all_greedy_equal"] = all(r["greedy_equal"] for p in rep["policies"] for r in p["runs"])
    (a.out_dir or ROOT).mkdir(parents=True, exist_ok=True)
    out = (a.out_dir or ROOT / "research" / "l2l" / "experiments") / f"{a.out}.json"
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"greedy equal everywhere: {rep['all_greedy_equal']}; budget verdict {rep['budget_verdict']} (evidence {ev}); "
          f"written {out}")
    return 0 if rep["all_greedy_equal"] and not viol else 1


if __name__ == "__main__":
    sys.exit(main())
