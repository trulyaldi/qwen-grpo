"""Ticket 11: profile every phase of the L2L workflow on a real model, with a GPU-memory guard, and compare with the
budget calculator's predictions.

    python scripts/profile_run.py --config configs/experiments/qwen_l2l.yaml --backend l2l_async \
        --prediction results/budget/predictions/qwen3-0.6b_4gb.json --budget-gib 4
    python scripts/profile_run.py --config configs/experiments/qwen3_4b_l2l.yaml \
        --prediction results/budget/predictions/qwen3-4b_8gb.json --budget-gib 8

Phases (execution.profiler.Profiler windows; nvidia-smi sampled throughout, raw output kept):
  setup         model loaded on the CPU, WeightStore built (pinned), non-streamed tensors placed on the GPU
  rollout       streamed sampling of batch_size x group_size completions (execution.generator)
  old_logp      no-grad log-probs in micro-batches (training.prepare's pass)
  forward       differentiable forward + GRPO loss (checkpointed layers)
  backward      backward with layer recomputation
  optimizer     AdamW step
  merge         streaming LoRA merge of the current adapter into the base checkpoint (CPU, tensor by tensor)
  evaluation    greedy validation predictions through L2L (never model.generate)
repeated --steps times for rollout..optimizer. Also: the effective pinned H2D bandwidth (measured) and, per phase,
the weight bytes the executor copied and the transfer time that bandwidth predicts, against the phase's wall time.
Verdict: FAIL if the guard fired; PASS only if no violation and the evidence is assignment-grade (numeric per-process
nvidia-smi on every sample, not WSL2); otherwise INCOMPLETE_EVIDENCE (a budget is never claimed on weaker evidence).
Writes research/l2l/experiments/<--out or t11_profile_<run>_<backend>>.json.
"""
import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qwen_grpo import config, lab  # noqa: E402
from qwen_grpo.config import DTYPES  # noqa: E402
from qwen_grpo.evaluation import evaluate, metrics  # noqa: E402
from qwen_grpo.execution.backend import make_backend  # noqa: E402
from qwen_grpo.execution.prefetch import executor_bytes  # noqa: E402
from qwen_grpo.execution.profiler import Profiler, h2d_bandwidth  # noqa: E402
from qwen_grpo.grpo import grpo_loss  # noqa: E402
from qwen_grpo.model import lora_model, lora_modules  # noqa: E402
from qwen_grpo.sentiment import build_prompts, load_splits  # noqa: E402
from qwen_grpo.training import balanced_subset, decode, group_rewards, micro_batches, prepare  # noqa: E402

PREDICTED_PHASE = {"setup": "initialization", "rollout": "rollout", "old_logp": "training", "forward": "training",
                   "backward": "training", "optimizer": "training", "merge": "merge", "evaluation": "evaluation"}


def copied_bytes(ex, since):
    """Weight bytes copied since the counter reading `since` (executor_bytes: prefetches included). Before
    2026-11-19 this summed only the trace entries whose use was not a hit, so every layer an async prefetch had
    already delivered counted 0 bytes (t11 transfers under-counted; tests/test_h2d_accounting.py)."""
    return executor_bytes(ex) - since


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--backend", choices=["l2l_sync", "l2l_async"])
    ap.add_argument("--prefetch-depth", type=int, default=1)
    ap.add_argument("--micro-batch-size", type=int)
    ap.add_argument("--kv-cache", choices=["gpu", "cpu"])
    ap.add_argument("--budget-gib", type=float)
    ap.add_argument("--prediction", type=Path, help="results/budget/predictions/<name>.json to compare with")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    ap.add_argument("--out-dir", type=Path, help="directory of the JSON result (default research/l2l/experiments; "
                    "e.g. an evidence/<id>/ directory)")
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    x = cfg.execution
    backend_name = a.backend or (x.backend if x.backend != "resident" else "l2l_sync")
    x = dataclasses.replace(x, backend=backend_name, weight_storage=x.weight_storage or "pinned",
                            resident_layers=0, weight_buffers=max(x.weight_buffers or 2, a.prefetch_depth + 1),
                            prefetch_depth=a.prefetch_depth if backend_name == "l2l_async" else 0,
                            activation_checkpointing=True, kv_cache=a.kv_cache or x.kv_cache)
    t = cfg.training
    if a.micro_batch_size:
        t = dataclasses.replace(t, micro_batch_size=a.micro_batch_size)
    cfg = dataclasses.replace(cfg, execution=x, training=t)
    # an explicit --budget-gib wins; else the config's budget (memory.budget_bytes exact bytes, or run.gpu_budget_gib)
    budget = int(a.budget_gib * 2**30) if a.budget_gib else config.budget_bytes(cfg)
    device = torch.device(a.device)
    rep = {"config": str(a.config), "model": cfg.model.name, "dtype": cfg.model.dtype, "execution": dataclasses.asdict(x),
           "micro_batch_size": t.micro_batch_size, "B": t.batch_size, "G": t.group_size, "max_new": t.max_new_tokens,
           "steps": a.steps, "provenance": lab.provenance()}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    with Profiler(budget, device=a.device) as prof:
        try:
            with prof.phase("setup"):
                torch.manual_seed(t.seed)
                model = lora_model(DTYPES[cfg.model.dtype], "cpu", cfg.lora.spec(), cfg.lora.rank, cfg.lora.alpha,
                                   name=cfg.model.name)
                g = torch.Generator().manual_seed(0)
                with torch.no_grad():  # nonzero B: every gradient and the merge are nontrivial
                    for _, m in lora_modules(model):
                        w = m.lora_B["default"].weight
                        w.copy_((torch.randn(w.shape, generator=g) * 1e-2).to(w.dtype))
                be = make_backend(cfg, model, device)
            ex, store = be.executor, be.store
            rep["streamed_bytes_per_pass"] = sum(s.nbytes for s in store.slabs)
            rep["pinned_bytes"] = rep["streamed_bytes_per_pass"] if x.weight_storage == "pinned" else 0
            rep["h2d_bandwidth_bytes_per_s"] = h2d_bandwidth(store.max_slab_bytes, a.device) if prof.cuda else None
            params = be.trainable()
            opt = torch.optim.AdamW(params.values(), lr=t.learning_rate, weight_decay=0.0)
            train = balanced_subset(load_splits()["train"], cfg.data.train_samples, cfg.data.subset_seed)
            transfers = []
            for step in range(a.steps):
                batch = train[step * t.batch_size:(step + 1) * t.batch_size]
                prompts = build_prompts(tok, [e["text"] for e in batch])
                n0 = executor_bytes(ex)
                with prof.phase("rollout"):
                    r = be.generate_rollouts(tok, prompts, t.group_size, t.max_new_tokens)
                transfers.append(("rollout", copied_bytes(ex, n0)))
                rewards = group_rewards(decode(tok, r), [e["label"] for e in batch], t.group_size)
                if (rewards == rewards[:, :1]).all():
                    rewards = torch.tensor([[float((b + k) % 2) for k in range(t.group_size)] for b in range(t.batch_size)])
                n0 = executor_bytes(ex)
                with prof.phase("old_logp"):
                    adv, old = prepare(model, r, rewards, t.micro_batch_size, logprobs=be.logprobs)
                transfers.append(("old_logp", copied_bytes(ex, n0)))
                opt.zero_grad(set_to_none=True)
                N, losses = r["input_ids"].shape[0], []
                fwd_bytes = bwd_bytes = 0
                for rows, c in micro_batches(r, t.micro_batch_size):
                    n0 = executor_bytes(ex)
                    with prof.phase("forward"):
                        lp = be.logprobs(model, c)
                        loss = grpo_loss(lp, old[rows], adv[rows], c["completion_mask"], t.clip_eps) * (lp.shape[0] / N)
                    fwd_bytes += copied_bytes(ex, n0)
                    n0 = executor_bytes(ex)
                    with prof.phase("backward"):
                        loss.backward()
                    bwd_bytes += copied_bytes(ex, n0)
                    losses.append(loss.item())
                    del lp, loss
                transfers += [("forward", fwd_bytes), ("backward", bwd_bytes)]
                with prof.phase("optimizer"):
                    opt.step()
            if not a.no_merge:
                from qwen_grpo.export.merge import stream_merge
                with tempfile.TemporaryDirectory(prefix="t11_merge_") as tmp:
                    model.save_pretrained(Path(tmp) / "adapter")
                    with prof.phase("merge"):
                        merge_rep = stream_merge(cfg.model.name, Path(tmp) / "adapter", Path(tmp) / "merged")
                    rep["merge"] = {k: merge_rep[k] for k in ("scale", "copied", "shards", "total_bytes")}
            val = [{"id": i, **e} for i, e in enumerate(load_splits()["validation"][: a.eval_n])]
            n0 = executor_bytes(ex)
            with prof.phase("evaluation"):
                preds = evaluate(model, tok, val, generator=be.greedy)
            transfers.append(("evaluation", copied_bytes(ex, n0)))
            rep["evaluation_metrics"] = metrics(preds)
            rep["store_intact"] = store.verify() == []
        except KeyboardInterrupt:
            rep["stopped_by_guard"] = bool(prof.violations)
            if not prof.violations:
                raise
    rep["profile"] = prof.report()
    bw = rep.get("h2d_bandwidth_bytes_per_s")
    agg = {}
    for name, nbytes in transfers:
        agg.setdefault(name, 0)
        agg[name] += nbytes
    phase_s = {}
    for p in rep["profile"]["phases"]:
        phase_s[p["phase"]] = phase_s.get(p["phase"], 0) + p["seconds"]
    rep["transfers"] = {n: {"bytes": b, "predicted_transfer_s": b / bw if bw else None, "phase_wall_s": phase_s.get(n),
                            "transfer_share_of_wall": (b / bw) / phase_s[n] if bw and phase_s.get(n) else None}
                        for n, b in agg.items()}
    if a.prediction:
        pred = json.loads(a.prediction.read_text())
        by = {p["phase"]: p for p in pred["phases"]}
        rows = []
        for p in rep["profile"]["phases"]:
            q = by.get(PREDICTED_PHASE.get(p["phase"]))
            if q is None:
                continue
            gp = q["gpu_peak"]
            rows.append({"phase": p["phase"], "predicted_phase": q["phase"],
                         "predicted_allocated": gp.get("gpu_allocated"), "observed_peak_allocated": p.get("peak_allocated"),
                         "predicted_reserved": gp.get("gpu_reserved"), "observed_peak_reserved": p.get("peak_reserved"),
                         "predicted_device": gp.get("gpu_device"), "observed_device_peak": p.get("device_used_peak"),
                         "observed_process_peak": p.get("process_used_peak"),
                         "predicted_cpu": q.get("cpu_peak_bytes"), "observed_cpu_rss_peak": p.get("cpu_rss_peak")})
        rep["prediction"] = {"file": str(a.prediction), "prediction_config": pred.get("config"), "rows": rows,
                             "note": "the prediction's batch / token settings may differ from this run's; see both configs"}
    ev, viol = rep["profile"]["evidence"], rep["profile"]["violations"]
    rep["verdict"] = "FAIL" if viol else ("PASS" if ev == "assignment" else "INCOMPLETE_EVIDENCE")
    (a.out_dir or ROOT).mkdir(parents=True, exist_ok=True)
    out = (a.out_dir or ROOT / "research" / "l2l" / "experiments") / f"{a.out or f't11_profile_{cfg.run.name}_{backend_name}'}.json"
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"{cfg.model.name} {backend_name} mb={t.micro_batch_size}: verdict {rep['verdict']} (evidence {ev}, "
          f"{len(viol)} violations, budget {budget} B = {budget / 2**30:.3f} GiB)")
    for p in rep["profile"]["phases"]:
        print(f"  {p['phase']:10s} {p['seconds']:7.2f} s  alloc {(p.get('peak_allocated') or 0) / 2**20:7.0f} MiB  "
              f"reserved {(p.get('peak_reserved') or 0) / 2**20:7.0f}  device {(p.get('device_used_peak') or 0) / 2**20:7.0f}"
              f"  process {(p.get('process_used_peak') or 0) / 2**20:7.0f}  rss {(p.get('cpu_rss_peak') or 0) / 2**20:7.0f}")
    print(f"written: {out}")
    return 0 if rep["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
