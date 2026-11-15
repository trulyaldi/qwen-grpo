"""Reports rebuilt from saved files, never by loading a model.

Verification of one checkpoint: save(evidence, data) writes the JSON evidence; render(out, data) reads it (and the
training log it names) and writes out/plots/ and out/report.md. A run: update_run(name) writes runs/<name>/plots/,
summary.json (machine-readable) and report.md (the one report to read) from everything saved under runs/<name>/.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from qwen_grpo import runs  # noqa: E402
from qwen_grpo.checkpointing import STATE, lock_owner  # noqa: E402

MODELS = ("base", "adapter", "merged", "lr_zero")
LABEL = {"base": "Base", "adapter": "Adapter", "merged": "Merged", "lr_zero": "LR=0"}
SECTIONS = ("parameter_updates", "policy_change", "merge_equivalence", "negative_control", "confidence_intervals",
            "reward_integrity", "training_diagnostics")


def save(evidence, out):
    """Into directory `out`: evaluation/<model>.json (metrics, AUC, raw predictions, label scores),
    verification/<section>.json, config.yaml, manifest.json."""
    out = Path(out)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    (out / "verification").mkdir(exist_ok=True)
    for name, ev in evidence["evaluations"].items():
        (out / "evaluation" / f"{name}.json").write_text(json.dumps({"model": name, "note": evidence["notes"].get(name), **ev}, indent=1))
    for k in SECTIONS:
        (out / "verification" / f"{k}.json").write_text(json.dumps(evidence[k], indent=1))
    (out / "config.yaml").write_text(yaml.safe_dump(evidence["config"], sort_keys=False))
    manifest = {k: evidence[k] for k in ("checkpoint", "split", "n", "label_balance", "notes", "training_log", "gpu_memory")}
    manifest["models"] = list(evidence["evaluations"])
    if "config_sha256" in evidence:
        manifest["config_sha256"] = evidence["config_sha256"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load(out):
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    ev = {"evaluations": {k: json.loads((out / "evaluation" / f"{k}.json").read_text()) for k in m["models"]}}
    for k in SECTIONS:
        ev[k] = json.loads((out / "verification" / f"{k}.json").read_text())
    ev["config"] = yaml.safe_load((out / "config.yaml").read_text())
    return {**m, **ev}


# --- plots -------------------------------------------------------------------------------------------------------

def _figure(path, xlabel, ylabel, title, draw):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    draw(ax)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path.name


def training_plots(log, plots):
    """Plots 1-4 and the loss diagnostic from a training log; every point is a recorded value."""
    rows, evals = log["training"], log["evaluations"]
    attempts = sorted(rows + log["zero_signal"], key=lambda g: g["attempt"])
    B = log["config"]["B"]
    steps = [r["step"] for r in rows]
    made = [
        _figure(plots / "reward.png", "optimizer step", "mean sampled reward", "Training reward (rollouts at each step)",
                lambda ax: ax.plot(steps, [r["mean_reward"] for r in rows], "o-", ms=3)),
    ]
    base = next((e for e in evals if e["step"] == 0), None)

    def acc(ax):
        ax.plot([e["step"] for e in evals], [e["metrics"]["accuracy"] for e in evals], "o-", label=f"adapter (n={evals[0]['metrics']['n']})")
        if base:  # step 0 = zero adapter = base model, on exactly the same validation subset
            ax.axhline(base["metrics"]["accuracy"], color="gray", ls="--", label="base, same subset (step 0)")
    made.append(_figure(plots / "validation_accuracy.png", "optimizer step", "accuracy", "Validation-subset accuracy", acc))
    made.append(_figure(plots / "mixed_reward_fraction.png", "rollout attempt (incl. skipped zero-signal batches)",
                        "fraction of groups with mixed rewards", f"Mixed-reward groups per batch (B={B})",
                        lambda ax: ax.plot([g["attempt"] for g in attempts], [g["mixed_groups"] / B for g in attempts], "o-", ms=3)))
    made.append(_figure(plots / "gradient_norm.png", "optimizer step", "LoRA gradient L2 norm",
                        "LoRA gradient norm (no gradient clipping is applied)",
                        lambda ax: ax.plot(steps, [r["grad_norm"] for r in rows], "o-", ms=3)))
    made.append(_figure(plots / "loss.png", "optimizer step", "GRPO loss", "GRPO loss: training diagnostic, not evidence of learning",
                        lambda ax: ax.plot(steps, [r["loss"] for r in rows], "o-", ms=3)))

    if not all("predicted_positive_rate" in e["metrics"] and e.get("predictions") for e in evals):
        return made  # not recorded by this log: no plot rather than an invented one

    def bias(ax):
        gold = evals[0]["predictions"]
        ax.plot([e["step"] for e in evals], [e["metrics"]["predicted_positive_rate"] for e in evals], "o-", label="predicted POSITIVE")
        ax.axhline(sum(p["label"] == "POSITIVE" for p in gold) / len(gold), color="gray", ls="--", label="gold POSITIVE rate")
        ax.set_ylim(0.0, 1.0)
    made.append(_figure(plots / "prediction_bias.png", "optimizer step", "fraction of the validation subset",
                        "POSITIVE prediction rate (step 0 = base model)", bias))
    return made


def model_plot(evals, plots):
    names = [m for m in MODELS if m in evals]
    metrics = ("accuracy", "macro_f1")

    def draw(ax):
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            xs = [j + (i - (len(names) - 1) / 2) * w for j in range(len(metrics))]
            bars = ax.bar(xs, [evals[n]["metrics"][k] for k in metrics], w, label=LABEL[n])
            ax.bar_label(bars, fmt="%.3f", fontsize=6)
        ax.set_xticks(range(len(metrics)), ["accuracy", "macro-F1"])
        ax.set_ylim(0.0, 1.05)  # bars start at zero; a truncated axis would exaggerate small differences
        ax.legend(fontsize=8, loc="lower center", ncol=len(names))
    n = evals[names[0]]["metrics"]["n"]
    return _figure(plots / "model_comparison.png", "metric", "value", f"Held-out comparison (n={n}; differences and CIs in report.md)", draw)


# --- report ------------------------------------------------------------------------------------------------------

def _f(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _ci(c):
    return f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]" + (" *" if c["ci_excludes_zero"] else "")


def _bf16_cause(b16):
    """Measured cause of bf16 merge differences: the share of the LoRA update lost when W0 + dW is stored in bf16."""
    r = [w.get("error_relative_to_delta") for w in b16.get("weights", {}).values()]
    if not r or None in r:
        return " (weight-level error relative to the update not recorded; rerun verify)."
    return (f" Storing W₀ + (α/r)BA in bf16 distorts the update itself: ‖(W_merged − W₀) − (α/r)BA‖_F / ‖(α/r)BA‖_F = "
            f"{min(r):.2f}–{max(r):.2f} across modules: ‖ΔW‖ is a small fraction of ‖W₀‖ (section 1), not far above bf16's "
            "relative precision (2⁻⁸). The attached adapter keeps the update in fp32. So the merge is algebraically correct (fp32 check), "
            "but a bf16 merge does not exactly preserve this adapter's behavior: deploy with the adapter attached or merge in fp32.")


def report(ev, plots_made, missing_plots):
    c, evals = ev["config"], ev["evaluations"]
    t, lo = c["training"], c["lora"]
    st = c["verification"]["statistics"]
    rows = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Negative recall", "recall_NEGATIVE"),
            ("Positive recall", "recall_POSITIVE"), ("Predicted-positive rate", "predicted_positive_rate"), ("Invalid rate", "invalid_rate")]
    cols = [m for m in MODELS]
    cell = lambda m, k: _f(evals[m]["metrics"][k]) if m in evals else "not run"
    md = [f"# Verification: `{ev['checkpoint']}`", "",
          "## Experiment configuration", "",
          *([] if ev["training_log"]["source"] else [
              f"**No training log for this checkpoint** ({ev['training_log']['missing_reason']}). The GRPO settings below are those of "
              "the YAML; only the LoRA settings are confirmed (against the checkpoint's adapter_config.json).", ""]),
          f"- Model {c['model']['name']} ({c['model']['dtype']} base, fp32 LoRA), non-thinking; dataset Rotten Tomatoes (`data/sentiment.json`).",
          f"- LoRA r={lo['rank']}, α={lo['alpha']}, dropout {lo['dropout']}, targets `{lo['targets']}`.",
          f"- GRPO: B={t['batch_size']} ({t['batch_mode']}), G={t['group_size']}, lr {t['learning_rate']}, {t['max_steps']} steps, "
          f"clip ε {t['clip_eps']}, rollout max_new_tokens {t['max_new_tokens']}, seed {t['seed']}.",
          f"- Held-out: `{ev['split']}`, n={ev['n']} {ev['label_balance']}; greedy, max_new_tokens 4 (baseline settings).",
          f"- Paired bootstrap: {st['bootstrap_samples']} resamples, {int(st['confidence_level'] * 100)}% percentile CIs, seed {st['seed']}.", "",
          "## Model comparison", "",
          "| Metric | " + " | ".join(LABEL[m] for m in cols) + " |", "|---|" + "---:|" * len(cols)]
    md += [f"| {name} | " + " | ".join(cell(m, k) for m in cols) + " |" for name, k in rows]
    md += ["| ROC AUC (label scores) | " + " | ".join(_f(evals[m]["auc"]) if m in evals else "not run" for m in cols) + " |", ""]
    md += [f"- {LABEL[m]}: {ev['notes'][m]}" for m in cols if m in ev["notes"]]
    md += ["", "## Paired differences (new − reference)", "",
           "`*` marks a CI that excludes zero. One training seed: the CIs cover evaluation sampling only.", "",
           "| Comparison | Δ accuracy [CI] | Δ macro-F1 [CI] | Δ AUC [CI] | improved / degraded |", "|---|---|---|---|---|"]
    for k, cmp in ev["confidence_intervals"].items():
        pc = cmp["paired_changes"]
        imp = sum(v["base_wrong_now_right"] for lab, v in pc.items() if lab in ("POSITIVE", "NEGATIVE"))
        deg = sum(v["base_right_now_wrong"] for lab, v in pc.items() if lab in ("POSITIVE", "NEGATIVE"))
        md.append(f"| {k.replace('_vs_', ' vs ')} | {_ci(cmp['accuracy'])} | {_ci(cmp['macro_f1'])} | {_ci(cmp['auc'])} | {imp} / {deg} |")

    pu, pol = ev["parameter_updates"], ev["policy_change"]
    eff = pu["effective"]
    md += ["", "## Verification evidence", "",
           f"### 1. Adapter parameters changed — {pu['status']}", "",
           f"- Initialization: {pu['init_source']}; validated by the LR=0 control: {_f(pu['init_validated_by_control'])}"
           + ("." if ev["training_log"]["source"] or pu["init_source"].startswith("saved") else
              " (the control validates the seeded procedure; that this checkpoint's run used the same seed is unconfirmed "
              "without its training log)."),
           f"- ‖θ − θ₀‖₂ = {pu['total_delta']:.4e}; {pu['tensors_changed']}/{pu['tensors_total']} LoRA tensors changed; "
           f"checkpoint holds only LoRA tensors: {pu['only_lora_tensors_in_checkpoint']}.",
           f"- Effective ‖(α/r)(BA − B₀A₀)‖_F per module: " + ", ".join(f"{k.split('layers.')[-1]} {v['delta_frobenius']:.3e} "
                                                                       f"({v['relative_to_W0']:.2e} of ‖W₀‖)" for k, v in eff.items()) + "."]
    td = ev["training_diagnostics"]
    if td:
        md += [f"- Training log: frozen tensors changed {len(td['frozen_changed'])}; {td['optimizer_updates']} updates from "
               f"{td['rollout_attempts']} rollout attempts ({td['zero_signal_batches']} zero-signal batches skipped); "
               f"mixed-reward group fraction {_f(td['mixed_reward_group_fraction'])}; grad norm range {td['grad_norm_range']} "
               f"(no clipping); max old/current log-prob gap {td['max_init_diff']} (so ρ ≡ 1 and clipping is inactive)."]
    else:
        md += [f"- Training log: not available ({ev['training_log']['missing_reason']}). Frozen-weight digests, gradient norms and "
               "reward curves of this run cannot be checked here."]
    p = pol.get("adapter")
    md += ["", "### 2. Policy outputs changed", "",
           f"- Adapter vs base on {p['n']} held-out examples: mean |Δs| {p['mean_abs_score_change']:.4f}, max |Δs| "
           f"{p['max_abs_score_change']:.4f}, mean Δs {p['mean_score_change']:+.4f} (s = log P(POSITIVE⟨im_end⟩) − log P(NEGATIVE⟨im_end⟩)); "
           f"greedy predictions differ on {p['prediction_disagreement_rate']:.2%}; correctness changed on {p['correctness_changed']} "
           f"({p['improved']} improved, {p['degraded']} degraded).",
           "- A nonzero change shows the adapter is active, not that it helps; a mean Δs far from 0 indicates a label-prior shift."]
    me = ev["merge_equivalence"]
    f32, b16 = me["fp32_cpu"], me["bf16_gpu"]
    md += ["", f"### 3. Merge equivalence — fp32 (merge correctness) {f32['status']}, bf16 (behavior preserved at deployment "
           f"precision) {b16['status']}", "",
           f"- fp32 on CPU, {f32['n_examples']} examples, all positions of prompt + POSITIVE⟨im_end⟩: max |Z_adapter − Z_merged| "
           f"{f32['adapter_vs_merged']['max_abs']:.3e} = {f32['adapter_vs_merged']['relative']:.2e} of max |Z| "
           f"(tolerance {f32['tolerance_relative']:.0e}); the adapter itself moves logits by {f32['adapter_vs_base']['max_abs']:.3e} "
           f"(power ratio {f32['power_ratio']:.1e}). Merged weights match W₀ + (α/r)BA to max "
           f"{max(w['max_abs_error'] for w in f32['weights'].values()):.2e}.",
           f"- bf16 on GPU (deployment precision): max |ΔZ| {b16['adapter_vs_merged']['max_abs']:.3e} "
           f"({b16['adapter_vs_merged']['relative']:.2e} relative) vs adapter effect {b16['adapter_vs_base']['max_abs']:.3e}; on the full "
           f"split, greedy predictions differ on {b16['full_split']['prediction_disagreement_rate']:.2%} of examples, max |Δs| "
           f"{b16['full_split']['max_abs_score_change']:.4f}." + _bf16_cause(b16)]
    nc = ev["negative_control"]
    md += ["", f"### 4. LR=0 negative control — {nc['status']}", ""]
    if nc["status"] == "NOT_RUN":
        md += [f"- {nc['reason']}."]
    else:
        pc = nc["policy_change_vs_base"]
        md += [f"- `{nc['checkpoint']}`: adapter equals the seeded initialization bitwise: {nc['equals_reconstructed_init']}; "
               f"B = 0: {nc['lora_B_all_zero']}; greedy disagreement with base {pc['prediction_disagreement_rate']:.2%}, max |Δs| "
               f"{pc['max_abs_score_change']}.",
               "- The control shows that the training procedure itself (sampling, rewards, backward) changes nothing without "
               "optimizer steps. It cannot show that a trained run's improvement is real learning rather than noise or reward exploitation."]
    ri = ev["reward_integrity"]
    acc = [pr["response"] for pr in ri["probes"] if pr["training_reward"] == 1.0]
    md += ["", "### 5. Reward integrity", "",
           f"- Training reward contract: {ri['parser_contract']}. Probe strings rewarded for gold POSITIVE: "
           + ", ".join(f"`{a!r}`" for a in acc) + f" (of {len(ri['probes'])} probes).",
           f"- Independent check: {ri['strict_contract']}. Disagreements (reward 1, not canonical):"]
    for name, r in ri["models"].items():
        md.append(f"  - {LABEL.get(name, name)}: {r['disagreements']}/{r['n']}; logged-correctness mismatches {r['logged_correct_mismatches']}; "
                  f"top responses {r['response_forms']}")
    tr = ri["training_rollouts"]
    md.append(f"  - Training rollouts: {tr['disagreements']}/{tr['n']}; logged rewards that re-score differently: "
              f"{tr['logged_reward_rescore_mismatches']}" if tr else "  - Training rollouts: not available (no training log).")
    md += ["- Zero disagreements show the model did not exploit the parser's leniency on these samples, not that exploitation is impossible."]

    md += ["", "## Interpretation", "",
           "1. **Parameters changed**: section 1 (raw and effective LoRA deltas against the initial adapter).",
           "2. **Policy changed**: section 2 (label-score shifts and prediction flips vs base).",
           "3. **Better held-out performance**: only if the adapter-vs-base CIs exclude zero; otherwise not established.",
           "4. **Merging preserves behavior**: section 3 (fp32 for correctness; bf16 for deployment-precision effects).",
           "Rising training reward is none of these: it is measured on the policy's own samples from 32 training prompts, it can rise "
           "from exploiting the parser or a label prior, and the GRPO loss is ≈ 0 by construction (advantages sum to zero per group at ρ = 1).",
           "", "## Plots", ""]
    md += [f"- `plots/{p}`" for p in plots_made] + [f"- not generated: {m}" for m in missing_plots]
    md += ["", "## Limitations", "",
           "- One training seed: seed-to-seed variation of the trained adapter is not measured.",
           "- Reward exploitation outside the probe strings and the observed responses is not covered.",
           "- One held-out split from the training distribution: distribution shift and rare failures are not measured.",
           "- bf16 results depend on batch shape and hardware; the fp32 merge check is the hardware-robust statement.",
           "- Behavior outside the two-label task (fluency, other prompts) is not measured by classification metrics.",
           f"- Memory during verification: GPU peak reserved {ev['gpu_memory']['peak_reserved_gib']:.2f} GiB (torch allocator, GPU phase "
           "only; excludes the CUDA context); host peak RSS "
           + (f"{ev['gpu_memory']['host_peak_rss_gib']:.2f} GiB." if "host_peak_rss_gib" in ev["gpu_memory"] else "not recorded.")]
    return "\n".join(md) + "\n"


def render(out, data=None):
    """report.md and plots/ in `out` from the evidence in `data` (default: `out` itself, the layout before runs/)."""
    out = Path(out)
    ev = load(data if data is not None else out)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    made, missing = [], []
    src = ev["training_log"]["source"]
    if src:
        made += training_plots(json.loads((runs.ROOT / src).read_text()), plots)
    else:
        missing.append(f"training plots (reward, validation accuracy, mixed-reward fraction, gradient norm, loss): "
                       f"{ev['training_log']['missing_reason']}")
    made.append(model_plot(ev["evaluations"], plots))
    (out / "report.md").write_text(report(ev, made, missing))
    return out / "report.md"


# --- run summary and report ----------------------------------------------------------------------------------------

KEYS = ("n", "accuracy", "macro_f1", "recall_NEGATIVE", "recall_POSITIVE", "predicted_positive_rate", "invalid_rate")


def compact(m, auc=None):
    return {**{k: m[k] for k in KEYS if k in m}, **({"auc": auc} if auc is not None else {})}


def steps_in(directory):
    found = [(int(m.group(1)), p) for p in (directory.iterdir() if directory.is_dir() else [])
             if (m := re.fullmatch(r"step(\d+)", p.name))]
    return sorted(found)


def training_record(name):
    """(log, source): training/training.json once training has ended; while it runs or after an interruption, the log
    saved with the latest resumable checkpoint; else (None, reason)."""
    import torch

    path = runs.training_dir(name) / "training.json"
    if path.is_file():
        return json.loads(path.read_text()), runs.rel(path)
    for step, ckpt in reversed(runs.checkpoints(name)):
        if (ckpt / STATE).is_file():
            return torch.load(ckpt / STATE, weights_only=True)["result"], f"{runs.rel(ckpt / STATE)} (training did not finish)"
    return None, "no training log and no resumable checkpoint"


def verification_summary(ev, report):
    ci = ev["confidence_intervals"].get("adapter_vs_base", {})
    me = ev["merge_equivalence"]
    return {"split": ev["split"], "n": ev["n"], "report": report,
            "models": {k: compact(e["metrics"], e["auc"]) for k, e in ev["evaluations"].items()},
            "parameter_updates": ev["parameter_updates"]["status"], "total_lora_delta": ev["parameter_updates"]["total_delta"],
            "merge_fp32": me["fp32_cpu"]["status"], "merge_bf16": me["bf16_gpu"]["status"],
            "negative_control": ev["negative_control"]["status"],
            "adapter_vs_base": {k: {"diff": ci[k]["diff"], "ci": ci[k]["ci"]} for k in ("accuracy", "macro_f1", "auc") if k in ci},
            "prediction_disagreement_vs_base": ev["policy_change"]["adapter"]["prediction_disagreement_rate"]}


def summarize(name):
    """summary.json of runs/<name>: everything below is read from saved files; what was not measured is None."""
    d = runs.run_dir(name)
    local = lambda path: str(Path(path).relative_to(d))  # paths in the summary are relative to the run directory
    m = runs.manifest(name)
    if m is None:
        raise SystemExit(f"{runs.rel(d)} is not a run directory (no manifest.json)")
    import yaml

    cfg = yaml.safe_load((d / "config.yaml").read_text())
    log, log_source = training_record(name)
    t, rows = cfg["training"], (log or {}).get("training", [])
    ckpts = runs.checkpoints(name)
    final = (log or {}).get("final")
    evaluations = {f"step{s}": {f.stem: {**compact(r["metrics"], r["auc"]["trained"]), "base_auc": r["auc"]["base"],
                                        "accuracy_diff_vs_base": r["vs_baseline"]["accuracy_diff"],
                                        "ci95": r["vs_baseline"]["ci95"], "file": local(f)}
                                for f in sorted(p.glob("*.json")) for r in [json.loads(f.read_text())]}
                   for s, p in steps_in(d / "evaluation")}
    verification = {f"step{s}": verification_summary(load(p / "evidence"), local(p / "report.md"))
                    for s, p in steps_in(d / "verification") if (p / "evidence" / "manifest.json").is_file()}
    return {
        "run": name, "id": m["id"], "status": runs.status(name, lock_owner), "created": m["created"],
        "config_sha256": m["config_sha256"], "git_commit": m["git_commit"], "git_dirty": m["git_dirty"],
        "model": cfg["model"], "lora": cfg["lora"], "data": cfg["data"],
        "grpo": {k: t[k] for k in ("batch_size", "batch_mode", "group_size", "learning_rate", "max_steps", "max_attempts",
                                   "clip_eps", "temperature", "max_new_tokens", "seed")},
        "training_log": log_source if log else None,  # repository-relative
        "progress": {"optimizer_steps": rows[-1]["step"] if rows else 0, "max_steps": t["max_steps"],
                     "rollout_attempts": max([g["attempt"] for g in rows + (log or {}).get("zero_signal", [])], default=0),
                     "zero_signal_batches": len((log or {}).get("zero_signal", [])),
                     "train_seconds": (log or {}).get("train_seconds")},
        "memory": {**((log or {}).get("memory") or {}), "gpu_budget_gib": cfg["run"]["gpu_budget_gib"],
                   "gpu": ((log or {}).get("environment") or {}).get("gpu")},
        "checkpoints": [{"step": s, "path": local(p), "resumable": (p / STATE).is_file()} for s, p in ckpts],
        "latest_checkpoint": local(ckpts[-1][1]) if ckpts else None,
        "validation_subset": [{"step": e["step"], **compact(e["metrics"])} for e in (log or {}).get("evaluations", [])],
        "full_validation": None if final is None else {
            "base": compact(final["base"], final["auc"]["base"]), "trained": compact(final["trained"], final["auc"]["trained"]),
            "accuracy_diff": final["accuracy_diff"], "ci95": final["bootstrap_ci95"],
            "auc_diff": final["auc_vs"]["base"]["auc_diff"], "auc_ci95": final["auc_vs"]["base"]["ci95"]},
        "checks": None if not log or "frozen_changed" not in log else {
            "frozen_weights_changed": len(log["frozen_changed"]), "params_equal_init": log["params_equal_init"],
            **({"lr_zero_control": log["control"]} if "control" in log else {})},
        "evaluations": evaluations, "verification": verification,
        "artifacts": {"config": "config.yaml", "manifest": "manifest.json", "report": "report.md",
                      **{k: local(f) for k, f in [("training_log", d / "training" / "training.json"),
                                                  ("console_log", d / "training" / "training.log"),
                                                  ("final_validation_predictions", d / "training" / "validation.json")]
                         if f.is_file()},
                      "plots": sorted(local(f) for f in (d / "plots").glob("*.png"))},
    }


def _pct(x):
    return "—" if x is None else f"{x:.4f}"


def _table(columns):
    """Markdown table of compact metrics, one column per {name: metrics}."""
    keys = [k for k in (*KEYS, "auc") if any(k in c for c in columns.values())]
    lines = ["| metric | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    return lines + [f"| {k} | " + " | ".join(str(c[k]) if k == "n" and k in c else _pct(c.get(k)) for c in columns.values()) + " |"
                    for k in keys]


def run_report(s, plots_made):
    """report.md of a run from its summary: what was measured, and what was not."""
    g, lo, p = s["grpo"], s["lora"], s["progress"]
    ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
    md = [f"# Run `{s['run']}` — {s['status']}", "",
          f"id `{s['id']}`, created {s['created']}, config sha256 `{s['config_sha256'][:16]}…`, git "
          f"`{(s['git_commit'] or 'unknown')[:12]}`{' (uncommitted changes)' if s['git_dirty'] else ''}.", "",
          "## 1. Configuration", "",
          f"- Model {s['model']['name']} ({s['model']['dtype']} base, fp32 LoRA), thinking {s['model']['thinking']}.",
          f"- LoRA r={lo['rank']}, α={lo['alpha']}, dropout {lo['dropout']}, targets `{lo['targets']}`.",
          f"- GRPO: B={g['batch_size']} ({g['batch_mode']}), G={g['group_size']}, lr {g['learning_rate']}, max {g['max_steps']} steps "
          f"/ {g['max_attempts']} attempts, clip ε {g['clip_eps']}, temperature {g['temperature']}, rollout max_new_tokens "
          f"{g['max_new_tokens']}, seed {g['seed']}.",
          f"- Data: {s['data']['train_samples']} training prompts, validation subset {s['data']['val_subset']} (subset seed "
          f"{s['data']['subset_seed']}). Full settings: `config.yaml`.", "",
          "## 2. Training progress", ""]
    if s["training_log"] is None:
        md += ["- No training log is available (training never reached its first checkpoint)."]
    else:
        secs = p["train_seconds"]
        md += [f"- {p['optimizer_steps']}/{p['max_steps']} optimizer steps from {p['rollout_attempts']} rollout attempts "
               f"({p['zero_signal_batches']} zero-signal batches skipped); training time "
               + ("not recorded (training did not finish)" if secs is None else f"{secs:.0f} s") + f". Source: `{s['training_log']}`.",
               "- Checkpoints: " + ", ".join(f"step {c['step']}{' (resumable)' if c['resumable'] else ''}" for c in s["checkpoints"])
               + (f"; latest `{s['latest_checkpoint']}/`." if s["latest_checkpoint"] else "none.")]
    md += ["", "## 3. Reward and gradient diagnostics", ""]
    if s["checks"]:
        c = s["checks"]
        md += [f"- Frozen base tensors changed: {c['frozen_weights_changed']}; adapter equal to its initialization: "
               f"{c['params_equal_init']}."] + ([f"- LR=0 control (in training): {c['lr_zero_control']}."] if "lr_zero_control" in c else [])
    else:
        md += ["- End-of-training checks (frozen weights, parameter change) not available: training did not finish."]
    md += ["- Rising rollout reward is measured on the policy's own samples of training prompts: it is not held-out evidence.",
           "- Plots: " + (", ".join(f"[{x}](plots/{x})" for x in plots_made) if plots_made else "none (no training log).")]
    md += ["", "## 4. Base vs trained (full validation, at the end of training)", ""]
    fv = s["full_validation"]
    if fv:
        md += _table({"base": fv["base"], "trained": fv["trained"]})
        md += ["", f"- Accuracy trained − base {fv['accuracy_diff']:+.4f}, paired bootstrap 95% CI {ci(fv['ci95'])}; "
               f"AUC diff {fv['auc_diff']:+.4f}, CI {ci(fv['auc_ci95'])}. One training seed: CIs cover evaluation sampling only."]
    else:
        md += ["- Not available: training.full_val is false or training did not finish. See `validate` results (section 5)."]
    md += ["", "## 5. Held-out evaluations (`validate` / `test`)", ""]
    if s["evaluations"]:
        md += ["| checkpoint | split | n | accuracy | macro-F1 | recall NEG | recall POS | pred. POS rate | AUC | Δ acc vs base [95% CI] |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        md += [f"| {st} | [{sp}]({r['file']}) | {r['n']} | {_pct(r['accuracy'])} | {_pct(r['macro_f1'])} | "
               f"{_pct(r['recall_NEGATIVE'])} | {_pct(r['recall_POSITIVE'])} | {_pct(r['predicted_positive_rate'])} | {_pct(r['auc'])} | "
               f"{r['accuracy_diff_vs_base']:+.4f} {ci(r['ci95'])} |" for st, e in s["evaluations"].items() for sp, r in e.items()]
    else:
        md += ["- None run yet (`scripts/run.py validate --run <name>`)."]
    md += ["", "## 6. POSITIVE prediction rate and classification drift", ""]
    sub = s["validation_subset"]
    if sub:
        md += [f"- Validation subset (n={sub[0]['n']}): predicted-POSITIVE rate " + ", ".join(
            f"step {e['step']} {e['predicted_positive_rate']:.3f}" for e in sub) + "; accuracy " + ", ".join(
            f"step {e['step']} {e['accuracy']:.3f}" for e in sub) + ". Step 0 is the base model (zero adapter)."]
    if fv:
        md += [f"- Full validation: predicted-POSITIVE rate base {fv['base']['predicted_positive_rate']:.4f} → trained "
               f"{fv['trained']['predicted_positive_rate']:.4f}; recall NEGATIVE {fv['base']['recall_NEGATIVE']:.4f} → "
               f"{fv['trained']['recall_NEGATIVE']:.4f}, POSITIVE {fv['base']['recall_POSITIVE']:.4f} → {fv['trained']['recall_POSITIVE']:.4f}."]
    if not sub and not fv:
        md += ["- Not available."]
    md += ["", "## 7. Verification", ""]
    if s["verification"]:
        md += ["| checkpoint | split (n) | parameters changed | merge fp32 / bf16 | LR=0 control | adapter − base accuracy [CI] | report |",
               "|---|---|---|---|---|---|---|"]
        for st, v in s["verification"].items():
            a = v["adapter_vs_base"].get("accuracy")
            md.append(f"| {st} | {v['split']} ({v['n']}) | {v['parameter_updates']} | {v['merge_fp32']} / {v['merge_bf16']} | "
                      f"{v['negative_control']} | " + (f"{a['diff']:+.4f} {ci(a['ci'])}" if a else "—")
                      + f" | [report]({v['report']}) |")
        md += ["", "NOT_RUN means the check was not executed (e.g. no LR=0 control configured), not that it passed."]
    else:
        md += ["- Not run yet (`scripts/run.py verify --run <name>`)."]
    mem = s["memory"]
    md += ["", "## 8. Hardware and memory", "",
           f"- GPU: {mem['gpu']['name'] if mem.get('gpu') else 'not recorded'}; peak reserved "
           + (f"{mem['peak_reserved_gib']:.2f} GiB" if mem.get("peak_reserved_gib") is not None else "not recorded")
           + f" (budget {mem['gpu_budget_gib']} GiB" + (f", within: {mem['within_budget']}" if mem.get("within_budget") is not None else "")
           + "); torch allocator statistics, excluding the CUDA context."]
    md += ["", "## 9. Files", "",
           "- `config.yaml` (resolved config), `manifest.json` (identity, status history, artifacts), `summary.json` (this report as data)",
           "- `training/` (training.json, training.log, validation.json), `checkpoints/step<N>/`, `evaluation/step<N>/`, "
           "`verification/step<N>/` (report.md, evidence/)"]
    return "\n".join(md) + "\n"


def update_run(name):
    """Rebuild runs/<name>/{plots/, verification reports, summary.json, report.md} and the manifest's artifact list."""
    d = runs.run_dir(name)
    log, _ = training_record(name)
    made = []
    if log and log.get("training"):
        (d / "plots").mkdir(exist_ok=True)
        made = training_plots(log, d / "plots")
    for _, out in steps_in(d / "verification"):
        if (out / "evidence" / "manifest.json").is_file():
            render(out, out / "evidence")
    s = summarize(name)
    runs.write_json(d / "summary.json", s)
    (d / "report.md").write_text(run_report(s, made))
    runs.record_artifacts(name)
    return d / "report.md"
