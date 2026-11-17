"""Evaluation: greedy predictions, metrics, complete-label scores, ROC AUC and paired bootstraps."""
from collections import Counter

import torch

from qwen_grpo.grpo import response_mask
from qwen_grpo.model import EOS_IDS
from qwen_grpo.policy import compute_logprobs
from qwen_grpo.runs import ROOT, baseline_path
from qwen_grpo.sentiment import LABELS, build_prompts, parse_label, reward

# Defaults of the saved baseline (research/baseline.py); runs override them through the YAML evaluation section.
MAX_NEW, BATCH = 4, 32  # each label is 2 tokens + EOS
SCORE_BATCH = 8  # label scoring keeps full [N, T, V] logits; 8 rows keeps that well inside the GPU budget
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 42
RESULTS, BENCHMARKS = ROOT / "results", ROOT / "results" / "benchmarks"  # research/baseline.py's outputs (shared)
predictions_path = baseline_path  # saved greedy predictions and label scores of the base model on a split


@torch.no_grad()
def generate(model, tok, sentences, max_new=MAX_NEW, generator=None):
    """Greedy responses -> (text without EOS, token length incl. EOS, truncated) per sentence.
    generator: None = model.generate(); else a greedy execution.generator.StreamedGenerator (the L2L backend)."""
    batch = tok(build_prompts(tok, sentences), return_tensors="pt", padding=True).to(model.device)
    if generator is None:
        ids = model.generate(**batch, do_sample=False, max_new_tokens=max_new)
    else:
        g = model.generation_config
        ids = generator(model, batch["input_ids"], batch["attention_mask"], num_return_sequences=1,
                        max_new_tokens=max_new, eos_token_ids=EOS_IDS, pad_token_id=g.pad_token_id)
    ids = ids[:, batch["input_ids"].shape[1]:]
    mask = response_mask(ids, EOS_IDS)
    out = []
    for row, m in zip(ids.tolist(), mask):
        n = int(m.sum())
        truncated = row[n - 1] not in EOS_IDS
        out.append((tok.decode(row[: n if truncated else n - 1]), n, truncated))
    return out



def predict(model, tok, examples, ids, batch=BATCH, max_new=MAX_NEW, generator=None):
    """Greedy prediction records in `batch`-sized chunks (order matters: bf16 outputs depend on batch shape)."""
    predictions, lengths, truncated = [], [], []
    for i in range(0, len(examples), batch):
        chunk = examples[i : i + batch]
        for k, ex, (response, n, trunc) in zip(ids[i : i + batch], chunk, generate(model, tok, [ex["text"] for ex in chunk], max_new, **({"generator": generator} if generator else {}))):
            predictions.append({
                "id": k,
                "text": ex["text"],
                "label": ex["label"],
                "response": response,
                "prediction": parse_label(response),
                "correct": reward(response, ex["label"]) == 1.0,
            })
            lengths.append(n)
            truncated.append(trunc)
    return predictions, lengths, truncated



def metrics(preds):
    n = len(preds)
    m = {"n": n, "accuracy": sum(p["correct"] for p in preds) / n,
         "invalid_rate": sum(p["prediction"] is None for p in preds) / n,
         "predicted_positive_rate": sum(p["prediction"] == "POSITIVE" for p in preds) / n,
         "predicted_negative_rate": sum(p["prediction"] == "NEGATIVE" for p in preds) / n}
    f1 = []
    for label in LABELS:
        tp = sum(p["prediction"] == label == p["label"] for p in preds)
        gold, pred = sum(p["label"] == label for p in preds), sum(p["prediction"] == label for p in preds)
        m[f"recall_{label}"] = tp / gold
        f1.append(2 * tp / (gold + pred))  # 2TP / (2TP + FP + FN); invalid outputs count as FN
    m["macro_f1"] = sum(f1) / len(f1)
    m["confusion"] = {str(k): {g: sum(p["prediction"] == k and p["label"] == g for p in preds) for g in LABELS}
                      for k in (*LABELS, None)}  # rows: predicted (None = invalid), columns: gold
    return m



def percentile_ci(x, level=0.95):
    a = (1 - level) / 2
    return x.quantile(a).item(), x.quantile(1 - a).item()


def paired_bootstrap(base_correct, new_correct, n=BOOTSTRAP, seed=42, chunk=1000, level=0.95):
    """Mean accuracy difference and percentile CI, resampling paired examples with replacement."""
    assert n % chunk == 0, (n, chunk)
    d = torch.tensor(new_correct, dtype=torch.float64) - torch.tensor(base_correct, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    means = torch.cat([d[torch.randint(len(d), (chunk, len(d)), generator=gen)].mean(1) for _ in range(n // chunk)])
    return (d.mean().item(), *percentile_ci(means, level))


def macro_f1_counts(preds, weights):
    """Macro-F1 under example multiplicities weights [K, n] (one row per resample) -> [K]; 2TP / (gold + predicted)."""
    f1 = []
    for label in LABELS:
        gold = torch.tensor([p["label"] == label for p in preds], dtype=torch.float64)
        pred = torch.tensor([p["prediction"] == label for p in preds], dtype=torch.float64)
        f1.append(2 * (weights @ (gold * pred)) / (weights @ gold + weights @ pred))
    return sum(f1) / len(f1)


def paired_f1_bootstrap(ref, new, n=BOOTSTRAP, seed=42, chunk=1000, level=0.95):
    """Macro-F1 difference new - ref with a paired bootstrap: each resample (multiplicity vector) scores both models."""
    assert n % chunk == 0, (n, chunk)
    assert [(p["id"], p["label"]) for p in ref] == [(p["id"], p["label"]) for p in new]
    m = len(ref)
    gen = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n // chunk):
        c = torch.zeros(chunk, m, dtype=torch.float64).scatter_add_(
            1, torch.randint(m, (chunk, m), generator=gen), torch.ones(chunk, m, dtype=torch.float64))
        diffs.append(macro_f1_counts(new, c) - macro_f1_counts(ref, c))
    ones = torch.ones(1, m, dtype=torch.float64)
    return {"f1_diff": (macro_f1_counts(new, ones) - macro_f1_counts(ref, ones)).item(),
            "ci": list(percentile_ci(torch.cat(diffs), level))}



def paired_changes(base, new):
    assert [(p["id"], p["text"], p["label"]) for p in base] == [(p["id"], p["text"], p["label"]) for p in new]
    out = {label: {"base_wrong_now_right": sum(not b["correct"] and t["correct"] for b, t in zip(base, new) if b["label"] == label),
                   "base_right_now_wrong": sum(b["correct"] and not t["correct"] for b, t in zip(base, new) if b["label"] == label)}
           for label in LABELS}
    out["flips"] = dict(Counter(f"{b['prediction']}->{t['prediction']}" for b, t in zip(base, new) if b["prediction"] != t["prediction"]))
    out["invalid_to_valid"] = sum(b["prediction"] is None and t["prediction"] is not None for b, t in zip(base, new))
    out["valid_to_invalid"] = sum(b["prediction"] is not None and t["prediction"] is None for b, t in zip(base, new))
    return out



def paired_compare(ref, new, n=BOOTSTRAP, seed=42):
    """new vs ref on the same examples (asserted aligned): accuracy difference, paired bootstrap CI, flips."""
    changes = paired_changes(ref, new)
    diff, lo, hi = paired_bootstrap([p["correct"] for p in ref], [p["correct"] for p in new], n=n, seed=seed)
    return {"ref_metrics": metrics(ref), "accuracy_diff": diff, "ci95": [lo, hi], "paired_changes": changes}



@torch.no_grad()
def label_scores(model, tok, examples, batch=SCORE_BATCH, logprobs=None):
    """s(x) = log P(POSITIVE<|im_end|> | x) - log P(NEGATIVE<|im_end|> | x): teacher-forced sums over every token
    of the label's full tokenization plus the end-of-turn token, on the evaluation prompt. Not used in training.
    logprobs: (model, rollouts) -> [n, R], default compute_logprobs (a backend's logprobs for L2L)."""
    logprobs = logprobs or compute_logprobs
    conts = {label: tok(label, add_special_tokens=False)["input_ids"] + [EOS_IDS[0]] for label in LABELS}
    out = []
    for i in range(0, len(examples), batch):
        chunk = examples[i : i + batch]
        enc = tok(build_prompts(tok, [e["text"] for e in chunk]), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logp = {}
        for label, c in conts.items():
            c = torch.tensor(c, device=model.device).expand(len(chunk), -1)
            r = {"input_ids": torch.cat([enc["input_ids"], c], 1), "prompt_len": enc["input_ids"].shape[1],
                 "attention_mask": torch.cat([enc["attention_mask"], torch.ones_like(c)], 1)}
            logp[label] = logprobs(model, r).sum(1)
        out += (logp["POSITIVE"] - logp["NEGATIVE"]).tolist()
    return out



def win_matrix(scores, labels):
    """W[i, j] = 1[s_pos_i > s_neg_j] + 0.5 * 1[tie], positives and negatives in example order; AUC = W.mean()."""
    s = torch.tensor(scores, dtype=torch.float64)
    is_pos = torch.tensor([label == "POSITIVE" for label in labels])
    p, n = s[is_pos][:, None], s[~is_pos][None, :]
    return (p > n).double() + 0.5 * (p == n).double()



def paired_auc_bootstrap(W_new, W_ref, n=BOOTSTRAP, seed=42, chunk=500, level=0.95):
    """AUC difference with a class-stratified paired bootstrap: the same resampled examples score both models.

    With multiplicity vectors c1, c0 of resampled positives/negatives, the resampled AUC is c1^T W c0 / (n1 n0).
    """
    assert n % chunk == 0, (n, chunk)
    n1, n0 = W_new.shape
    gen = torch.Generator().manual_seed(seed)
    counts = lambda m: torch.zeros(chunk, m, dtype=torch.float64).scatter_add_(
        1, torch.randint(m, (chunk, m), generator=gen), torch.ones(chunk, m, dtype=torch.float64))
    diffs = []
    for _ in range(n // chunk):
        c1, c0 = counts(n1), counts(n0)
        diffs.append((((c1 @ W_new) * c0).sum(1) - ((c1 @ W_ref) * c0).sum(1)) / (n1 * n0))
    d = torch.cat(diffs)
    ci = list(percentile_ci(d, level))
    return {"auc_diff": (W_new.mean() - W_ref.mean()).item(), "ci": ci, **({"ci95": ci} if level == 0.95 else {})}



def evaluate(model, tok, examples, batch=BATCH, max_new=MAX_NEW, generator=None):
    return predict(model, tok, examples, [e["id"] for e in examples], batch, max_new, generator)[0]


def settings(ev):
    """Keyword arguments of evaluate() / label_scores() from a config.Evaluation."""
    return {"batch": ev.batch_size, "max_new": ev.max_new_tokens}, {"batch": ev.score_batch_size}



TABLE_ROWS = ("n", "accuracy", "macro_f1", "recall_NEGATIVE", "recall_POSITIVE", "predicted_positive_rate", "invalid_rate", "auc")


def metrics_table(columns):
    """Markdown table: one column per {name: metrics dict (optionally with "auc")}, one row per TABLE_ROWS entry."""
    cell = lambda v: "—" if v is None else str(v) if isinstance(v, int) else f"{v:.4f}"
    lines = ["| metric | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    lines += [f"| {r} | " + " | ".join(cell(m.get(r)) for m in columns.values()) + " |" for r in TABLE_ROWS]
    return "\n".join(lines)



def confusion_table(m):
    """Rows: predicted (None = invalid), columns: gold."""
    lines = ["| predicted \\ gold | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    lines += [f"| {k} | " + " | ".join(str(row[g]) for g in LABELS) + " |" for k, row in m["confusion"].items()]
    return "\n".join(lines)



def summary(m):
    return (f"acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}  rec POS {m['recall_POSITIVE']:.4f} NEG {m['recall_NEGATIVE']:.4f}"
            f"  invalid {m['invalid_rate']:.4f}  pred POS {m['predicted_positive_rate']:.4f}  (n={m['n']})")

