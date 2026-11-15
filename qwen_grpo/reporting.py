"""Training diagnostics plots from a saved training log (results/<run>/training.json): reward, validation-subset
accuracy, mixed-reward fraction, gradient norm and loss. Every point is a recorded value."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
