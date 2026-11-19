"""Builds reports/figures/fig_memory_scores.svg from the saved run artifacts (read-only).

Run from the repository root:  .venv/bin/python -I reports/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Arial (metric-compatible with the Liberation Sans of the assignment PDF), from the Windows host if present
for f in Path("/mnt/c/Windows/Fonts").glob("arial*.ttf"):
    font_manager.fontManager.addfont(str(f))

ROOT = Path.cwd()
MEM = ROOT / "runs/qwen3_4b_final_plain_150_v2/training/memory/20261119T054525Z.json"
T13 = ROOT / "research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json"
DATA = ROOT / "data/sentiment.json"
OUT = ROOT / "reports/figures/fig_memory_scores.svg"

plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "svg.fonttype": "none"})
INK, ACC, MUTED, GRID = "#222222", "#1f5fa8", "#8a8a8a", "#e6e6e6"

m = json.loads(MEM.read_text())
s = m["samples"]
t0 = s[0]["t"]
t = [x["t"] - t0 for x in s]
gb = lambda k: [x[k] / 1e9 for x in s]

e = json.loads(T13.read_text())
gold = [r["label"] for r in json.loads(DATA.read_text())["test"]]
gold = [("NEGATIVE", "POSITIVE")[g] if isinstance(g, int) else g for g in gold]
sb, st = e["label_scores"]["base"], e["label_scores"]["trained"]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 1.78), gridspec_kw={"width_ratios": [1.55, 1]})
a.axhline(8.0, color=INK, lw=0.9, ls=(0, (4, 2)))
a.text(t[-1], 7.75, "assignment cap 8 GB = 8×10⁹ B (nvidia-smi)", ha="right", va="top", color=INK, fontsize=6.8)
a.plot(t, gb("process_used"), color=ACC, lw=1.0, label="nvidia-smi per process (incl. CUDA context)")
a.plot(t, gb("reserved"), color=INK, lw=0.7, label="PyTorch reserved")
a.plot(t, gb("allocated"), color=MUTED, lw=0.5, label="PyTorch allocated")
a.set_xlim(0, t[-1]); a.set_ylim(0, 9.6)
a.set_xlabel("seconds since sampler start (5947 samples, median gap 0.56 s)")
a.set_ylabel("GPU memory (10⁹ B)")
a.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=False, fontsize=6.5, handlelength=1.6)
a.set_title("(a) Training run: 150 GRPO updates, peak 2.865×10⁹ B per process", fontsize=7.5, loc="left")

pos = [g == "POSITIVE" for g in gold]
b.scatter([x for x, p in zip(sb, pos) if p], [y for y, p in zip(st, pos) if p], s=2.2, color=ACC, lw=0, alpha=0.7,
          label="gold POSITIVE")
b.scatter([x for x, p in zip(sb, pos) if not p], [y for y, p in zip(st, pos) if not p], s=2.2, color=INK, lw=0,
          alpha=0.55, label="gold NEGATIVE")
lo, hi = min(sb + st) - 1, max(sb + st) + 1
b.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls="--")
b.axhline(0, color=GRID, lw=0.6, zorder=0); b.axvline(0, color=GRID, lw=0.6, zorder=0)
b.set_xlim(lo, hi); b.set_ylim(lo, hi)
b.set_xlabel("base  s(x) = log P(POS) − log P(NEG)")
b.set_ylabel("trained (step 150)  s(x)")
b.legend(loc="upper left", frameon=False, fontsize=6.5, markerscale=3, handletextpad=0.2)
b.set_title("(b) Test set, n = 1066: mean shift +6.01 nats", fontsize=7.5, loc="left")
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=2)
fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig(OUT)
print("wrote", OUT)
