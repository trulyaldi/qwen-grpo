"""Verification outputs: JSON evidence files, Matplotlib plots and a Markdown report, all rebuilt from saved files.

save(evidence, out) writes the JSON files; render(out) reads them (and the training log they name) and writes plots/
and report.md without loading any model.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("base", "adapter", "merged", "lr_zero")
LABEL = {"base": "Base", "adapter": "Adapter", "merged": "Merged", "lr_zero": "LR=0"}
SECTIONS = ("parameter_updates", "policy_change", "merge_equivalence", "negative_control", "confidence_intervals",
            "reward_integrity", "training_diagnostics")


def save(evidence, out):
    """evaluation/<model>.json (metrics, AUC, raw predictions, label scores), verification/<section>.json, manifest."""
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
    md += ["", f"### 3. Merge equivalence — fp32 {f32['status']}, bf16 {b16['status']}", "",
           f"- fp32 on CPU, {f32['n_examples']} examples, all positions of prompt + POSITIVE⟨im_end⟩: max |Z_adapter − Z_merged| "
           f"{f32['adapter_vs_merged']['max_abs']:.3e} = {f32['adapter_vs_merged']['relative']:.2e} of max |Z| "
           f"(tolerance {f32['tolerance_relative']:.0e}); the adapter itself moves logits by {f32['adapter_vs_base']['max_abs']:.3e} "
           f"(power ratio {f32['power_ratio']:.1e}). Merged weights match W₀ + (α/r)BA to max "
           f"{max(w['max_abs_error'] for w in f32['weights'].values()):.2e}.",
           f"- bf16 on GPU (deployment precision): max |ΔZ| {b16['adapter_vs_merged']['max_abs']:.3e} "
           f"({b16['adapter_vs_merged']['relative']:.2e} relative) vs adapter effect {b16['adapter_vs_base']['max_abs']:.3e}; on the full "
           f"split, greedy predictions differ on {b16['full_split']['prediction_disagreement_rate']:.2%} of examples, max |Δs| "
           f"{b16['full_split']['max_abs_score_change']:.4f}. Merged weights are rounded to bf16 (W₀ + (α/r)BA), while the attached "
           "adapter adds an fp32 LoRA output to the bf16 base output, so small bf16 differences are expected; the fp32 check is the "
           "test of merge correctness."]
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


def render(out):
    out = Path(out)
    ev = load(out)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    made, missing = [], []
    src = ev["training_log"]["source"]
    if src:
        made += training_plots(json.loads((ROOT / src).read_text()), plots)
    else:
        missing.append(f"training plots (reward, validation accuracy, mixed-reward fraction, gradient norm, loss): "
                       f"{ev['training_log']['missing_reason']}")
    made.append(model_plot(ev["evaluations"], plots))
    (out / "report.md").write_text(report(ev, made, missing))
    return out / "report.md"
