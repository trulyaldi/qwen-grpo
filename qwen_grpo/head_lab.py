"""Ticket 05 lab case: logprob_equivalence (H03), the chunked output head and the streamed GRPO log-probabilities
against the full-logits resident reference.

Grid: vocabulary V in {64, 1000, 151936 (Qwen3)} x fp32/bf16 x response length R in {1, 4, 16} x micro-batch size
{whole batch, 2}. Model: tiny Qwen3 (0.6b-like shape) with LoRA on every projection of the first and last layers.
Sides, each a log-prob function plugged into the unchanged training.prepare / training.grpo_update:
  full            resident, explicit forward + full [N, R, V] logits + response_logprobs arithmetic: the reference.
                  It must equal policy.compute_logprobs (the HF forward) bitwise (reference_integrity).
  chunked<C>      resident, output_head chunked selected-token log-probs over C vocabulary rows (two C per V, at least
                  one not dividing V, so a remainder chunk is always exercised)
  streamed_full   L2L (backward.CheckpointedLayerExecutor) + full logits
  streamed<C>     L2L + chunked head: the Ticket 05 candidate
Optimizer iterations: the reference trains ITERATIONS AdamW steps with grpo_update (LR chosen so ratios move off 1
and tokens clip; checked: power). At each reference parameter snapshot k every side runs grpo_update(iteration=k)
against its own old log-probs (from its own prepare(), as training would), with a zero-LR probe optimizer that copies
the accumulated gradients; iteration 0 runs the existing old/current check (training.INIT_TOL) on every side.
Compared, candidate side vs full at the same parameters:
  logp.<tag>.<side>.old / .it<k>            selected log-probs [N, R]
  loss.<tag>.<side>.it<k>                   the micro-batch-accumulated GRPO loss grpo_update reports
  grads.<tag>.<side>.it<k>.<param>          every LoRA gradient
  head_grads.<tag>.<side>.it<k>             dL/d(head input) [N, R+1, H] (last column: zero), micro-batches concatenated
  hidden_grads.<tag>.<side>.it<k>.<i>       dL/d(output of decoder layer i) [N, T, H], micro-batches concatenated
                                            (per step, not per micro-batch: changed after t05_logprob_equivalence_cpu,
                                            where a fully clipped micro-batch had an exactly zero reference)
Measurements (not thresholds): each candidate's own ITERATIONS-step training run (final LoRA distance to the
reference's, its logs); output-head memory (largest tensor allocated and non-parameter bytes saved for backward, by
a dispatch mode / saved-tensor hooks, plus the CUDA allocator peak on a GPU) and time (interleaved trials) for the
tiny models and for the real Qwen3-0.6B head shape (H 1024, V 151936). Execution requirements: the chunked head's
largest allocation is bounded by its chunk (max(M*C, M*H, C*H) fp32 elements), and its saved bytes do not grow with V.
Fault injection (perturbation "raw_logit"): candidate sides return the selected token's raw logit, unnormalized.
"""
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from qwen_grpo import lab
from qwen_grpo import backward_lab  # its build(); module import: backward_lab imports lab, which imports this
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.layer_stream import place_resident
from qwen_grpo.execution.output_head import output_head, selected_logprobs, streamed_logprobs
from qwen_grpo.execution.qwen3_forward import explicit_hidden
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.grpo import group_advantages, response_mask
from qwen_grpo.policy import compute_logprobs
from qwen_grpo.training import check_grouping, grpo_update, micro_batches, prepare

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
QWEN3_VOCAB = 151936
VOCABS = {64: (7, 64), 1000: (7, 256), QWEN3_VOCAB: (8192, 32768)}  # chunk sizes per V; 64 % 7, 1000 % 7, V % 32768 != 0
LENGTHS = (1, 4, 16)
MICRO = (None, 2)
SHAPE = "0.6b-like"
ITERATIONS, LR = 4, 5e-2
P, PROMPT_LENGTHS = 6, (6, 3)
REAL_HEAD = {"hidden": 1024, "vocab": QWEN3_VOCAB, "rows": 4, "R": 128, "chunk": 8192}  # Qwen3-0.6B output head


def rollouts(V, R):
    """B=2 prompts x G=3 completions, left-padded prompts, responses of width R: truncated, EOS last, EOS first, EOS
    mid-row with a padded tail, the second EOS id, truncated. Seeded by (V, R)."""
    g = torch.Generator().manual_seed(V * 100 + R)
    B, G = len(PROMPT_LENGTHS), 3
    prompt_mask = torch.stack([torch.arange(P) >= P - n for n in PROMPT_LENGTHS]).long()
    prompts = torch.randint(3, V, (B, P), generator=g).masked_fill(prompt_mask == 0, F.PAD)
    resp = torch.randint(3, V, (B * G, R), generator=g)
    for row, (at, eos) in enumerate([(None, None), (R - 1, 1), (0, 1), (R // 2, 1), (R // 3, 2), (None, None)]):
        if at is not None:
            resp[row, at] = eos
            resp[row, at + 1:] = F.PAD
    completion = response_mask(resp, F.EOS_IDS)
    r = {"input_ids": torch.cat([prompts.repeat_interleave(G, 0), resp], 1),
         "attention_mask": torch.cat([prompt_mask.repeat_interleave(G, 0), completion.long()], 1),
         "response_ids": resp, "completion_mask": completion, "prompt_len": P}
    check_grouping(r, prompts, G)
    return r, group_advantages(torch.tensor(F.REWARDS, dtype=torch.float32)).flatten()


def raw_logit(model, r, executor, layer_callback, head_callback):
    """Fault: the selected token's raw logit, never normalized over the vocabulary."""
    ids, mask = r["input_ids"], r["attention_mask"]
    hidden = explicit_hidden(model, ids, mask, (mask.cumsum(1) - 1).clamp(min=0), executor, layer_callback)
    h = hidden[:, r["prompt_len"] - 1:-1]
    head_callback(h)
    w = output_head(model)
    return (h * w[ids[:, r["prompt_len"]:]]).sum(-1).float()


class Capture:
    """A log-prob function for prepare/grpo_update that keeps, per micro-batch, the log-probs and the graph tensors
    whose gradients are compared (head input, decoder-layer outputs)."""

    def __init__(self, model, chunk, executor=None, fault=False):
        self.model, self.chunk, self.executor, self.fault = model, chunk, executor, fault
        self.reset()

    def reset(self):
        self.logp, self.head, self.layers = [], [], []

    def __call__(self, model, r):
        layers, head = {}, []

        def keep(i, h):
            if h.requires_grad:
                h.retain_grad()
                layers[i] = h

        def keep_head(h):
            if h.requires_grad:
                h.retain_grad()
                head.append(h)

        if self.fault:
            lp = raw_logit(model, r, self.executor, keep, keep_head)
        else:
            lp = streamed_logprobs(model, r, self.chunk, self.executor, keep, keep_head)
        self.logp.append(lp.detach())
        self.layers.append(layers)
        self.head.append(head)
        return lp


def probe_optimizer(params):
    """Zero-LR SGD whose step pre-hook copies the accumulated gradients (grpo_update zeroes them after the step)."""
    opt, grads = torch.optim.SGD(params.values(), lr=0.0), {}
    opt.register_step_pre_hook(lambda o, a, k: grads.update({n: p.grad.clone() for n, p in params.items()}))
    return opt, grads


def load(params, snapshot):
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(snapshot[n])


def side_at(model, capture, r, adv, old, k, mb, snapshot):
    """grpo_update(iteration=k) at the snapshot's parameters -> (outputs to compare, update logs)."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    load(params, snapshot)
    opt, grads = probe_optimizer(params)
    capture.reset()
    logs = grpo_update(model, params, opt, r, adv, old, F.CLIP_EPS, k, mb, logprobs=capture)
    out = {"logp": torch.cat(capture.logp), "loss": torch.tensor(logs["loss"], dtype=torch.float64), "grads": grads,
           "head": [h[0].grad for h in capture.head],
           "layers": [{i: t.grad for i, t in d.items()} for d in capture.layers]}
    return out, logs


def _concat(grads):
    """The micro-batches' gradients concatenated along rows -> [N, ...]; None if any micro-batch has none (the
    comparison then reports the quantity as missing)."""
    return None if not grads or any(g is None for g in grads) else torch.cat(grads, dim=0)


def flat(out, prefix):
    """Per-step tensors. Head-input and layer-output gradients are compared for the whole step (micro-batches
    concatenated): one micro-batch whose tokens are all clipped has an exactly zero gradient there, legitimately, and
    the nonzero-reference requirement applies to the step (changed after t05_logprob_equivalence_cpu)."""
    t = {f"logp.{prefix}": out["logp"], f"loss.{prefix}": out["loss"]}
    t.update({f"grads.{prefix}.{n}": g for n, g in out["grads"].items()})
    if (head := _concat(out["head"])) is not None:
        t[f"head_grads.{prefix}"] = head
    for i in sorted(set().union(*(d.keys() for d in out["layers"]))):
        if (g := _concat([d.get(i) for d in out["layers"]])) is not None:
            t[f"hidden_grads.{prefix}.{i}"] = g
    return t


def train(model, capture, r, adv, mb, iterations):
    """The unchanged training loop for one rollout batch: prepare once, then `iterations` AdamW grpo_updates."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(params.values(), lr=LR, weight_decay=0.0)
    _, old = prepare(model, r, torch.tensor(F.REWARDS, dtype=torch.float32), mb, logprobs=capture)
    snaps, logs = [], []
    for k in range(iterations):
        snaps.append({n: p.detach().clone() for n, p in params.items()})
        logs.append(grpo_update(model, params, opt, r, adv, old, F.CLIP_EPS, k, mb, logprobs=capture))
    return old, snaps, logs, {n: p.detach().clone() for n, p in params.items()}


def run_vocab(cfg, result, V, dt, perturbation, tol):
    dtype = DTYPES[dt]
    n_layers = F.SHAPES[SHAPE]["num_hidden_layers"]
    layers = [0, n_layers - 1]
    with F.isolated():
        ref = backward_lab.build(SHAPE, "eager", dtype, layers, vocab=V)
        l2l = backward_lab.build(SHAPE, "eager", dtype, layers, vocab=V)
    l2l.load_state_dict(ref.state_dict(), strict=True)
    store = WeightStore.build(l2l, cfg.execution.weight_storage)
    try:
        place_resident(l2l, store, "cpu")
        ex = CheckpointedLayerExecutor(store, gpu_buffers(cfg.execution.weight_buffers, store.max_slab_bytes, "cpu"))
        fault = perturbation == "raw_logit"
        C0 = VOCABS[V][0]
        sides = {"full": (ref, Capture(ref, None))}
        sides.update({f"chunked{C}": (ref, Capture(ref, C, fault=fault)) for C in VOCABS[V]})
        sides["streamed_full"] = (l2l, Capture(l2l, None, ex, fault=fault))
        sides[f"streamed{C0}"] = (l2l, Capture(l2l, C0, ex, fault=fault))
        initial = {n: p.detach().clone() for n, p in ref.named_parameters() if p.requires_grad}
        for R in LENGTHS:
            r, adv = rollouts(V, R)
            for mb in MICRO:
                tag = f"V{V}.{dt}.R{R}.mb{mb or 'all'}"
                ref_params = {n: p for n, p in ref.named_parameters() if p.requires_grad}
                load(ref_params, initial)
                with F.isolated():
                    old_ref, snaps, ref_logs, ref_final = train(ref, sides["full"][1], r, adv, mb, ITERATIONS)
                    load(ref_params, initial)
                    hf = torch.cat([compute_logprobs(ref, c) for _, c in micro_batches(r, mb)])
                if not torch.equal(hf, old_ref):
                    result["reference_integrity"].append(f"{tag}: full-logits reference != policy.compute_logprobs")
                if not any(log["clip_frac"] > 0 for log in ref_logs[1:]):
                    result["errors"].append(f"{tag}: no token clipped in iterations 1..{ITERATIONS - 1}: no power")
                result["measurements"]["reference_logs"][tag] = [{k: v for k, v in log.items() if k != "grad_norms"}
                                                                for log in ref_logs]
                olds = {"full": old_ref}
                for name, (model, cap) in sides.items():
                    if name != "full":
                        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
                        load(params, initial)
                        with F.isolated(), torch.no_grad():
                            olds[name] = torch.cat([cap(model, c) for _, c in micro_batches(r, mb)])
                ref_t, got_t = {}, {}
                for k, snap in enumerate(snaps):
                    with F.isolated():
                        want, logs = side_at(ref, sides["full"][1], r, adv, old_ref, k, mb, snap)
                    if logs["loss"] != ref_logs[k]["loss"]:
                        result["reference_integrity"].append(f"{tag}.it{k}: probe loss != the training run's")
                    for name, (model, cap) in sides.items():
                        if name == "full":
                            continue
                        try:
                            with F.isolated():
                                got, _ = side_at(model, cap, r, adv, olds[name], k, mb, snap)
                        except RuntimeError as e:  # e.g. the INIT_TOL check: a candidate failure, recorded as such
                            result["errors"].append(f"{tag}.{name}.it{k}: {e}")
                            continue
                        prefix = f"{tag}.{name}.it{k}"
                        ref_t.update(flat(want, prefix))
                        got_t.update(flat(got, prefix))
                for name in sides:
                    if name != "full":
                        ref_t[f"logp.{tag}.{name}.old"], got_t[f"logp.{tag}.{name}.old"] = old_ref, olds[name]
                result["comparisons"] += [{"model": tag, **row} for row in lab.compare_all(ref_t, got_t, tol)]
                # the candidate's own training run: how far a trajectory drifts (a measurement, not a threshold)
                name = f"streamed{C0}"
                model, cap = sides[name]
                load({n: p for n, p in model.named_parameters() if p.requires_grad}, initial)
                try:
                    with F.isolated():
                        _, _, logs, final = train(model, cap, r, adv, mb, ITERATIONS)
                    drift = max(((final[n] - ref_final[n]).abs().max() / (ref_final[n] - initial[n]).abs().max()).item()
                                for n in final)
                    result["measurements"]["trajectory"][f"{tag}.{name}"] = {
                        "max_final_lora_diff_over_max_update": drift,
                        "init_diff": logs[0]["init_diff"], "loss": [log["loss"] for log in logs],
                        "clip_frac": [log["clip_frac"] for log in logs]}
                except RuntimeError as e:
                    result["errors"].append(f"{tag}.{name} training run: {e}")
        if problems := store.verify():
            result["errors"] += [f"V{V}.{dt}: store: {p}" for p in problems[:5]]
    finally:
        store.close()


# --- output-head memory and time ------------------------------------------------------------------------------------

class Allocations(TorchDispatchMode):
    """The largest tensor storage any op allocates (outputs that do not share an input's storage)."""

    def __init__(self):
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        inputs = {t.untyped_storage().data_ptr() for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)}
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in inputs:
                self.largest = max(self.largest, t.untyped_storage().nbytes())
        return out


def head_inputs(N, R, H, V, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter((torch.randn(V, H, generator=g) * H ** -0.5).to(dtype).to(device), requires_grad=False)
    h = torch.randn(N, R, H, generator=g).to(dtype).to(device)
    t = torch.randint(0, V, (N, R), generator=g).to(device)
    return h, w, t, torch.randn(N, R, generator=g).to(device)


def head_step(h, w, t, g, chunk):
    x = h.detach().requires_grad_()
    (selected_logprobs(x, w, t, chunk) * g).sum().backward()
    return x.grad


def head_memory(h, w, t, g, chunk):
    """{largest allocation, non-parameter bytes saved for backward, CUDA peak above the start} of one fwd + bwd."""
    saved = [0]

    def pack(x):
        if not isinstance(x, torch.nn.Parameter):
            saved[0] += x.numel() * x.element_size()
        return x

    cuda = h.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
    x = h.detach().requires_grad_()
    with Allocations() as a:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda y: y):
            lp = selected_logprobs(x, w, t, chunk)
        (lp * g).sum().backward()
    out = {"largest_allocation_bytes": a.largest, "saved_bytes": saved[0]}
    if cuda:
        torch.cuda.synchronize()
        out["cuda_peak_above_start_bytes"] = torch.cuda.max_memory_allocated() - start
    return out


def head_timing(h, w, t, g, chunk, trials, warmup):
    sync = torch.cuda.synchronize if h.device.type == "cuda" else (lambda: None)

    def timed(c):
        sync()
        t0 = time.perf_counter()
        head_step(h, w, t, g, c)
        sync()
        return time.perf_counter() - t0

    for _ in range(warmup):
        timed(None), timed(chunk)
    full, chunked = [], []
    for _ in range(trials):
        full.append(timed(None))
        chunked.append(timed(chunk))
    return {"full": lab.summarize(full), "chunked": lab.summarize(chunked),
            "chunked_over_full": lab.ratio_ci(chunked, full)}


def head_resources(result, label, N, R, H, V, dtype, chunk, device, trials, warmup):
    h, w, t, g = head_inputs(N, R, H, V, dtype, device)
    acc = 8 if dtype == torch.float64 else 4
    M = N * R
    rec = {"N": N, "R": R, "H": H, "V": V, "dtype": str(dtype), "chunk": chunk,
           "full_fp32_logits_bytes": M * V * acc,
           "full": head_memory(h, w, t, g, None), "chunked": head_memory(h, w, t, g, chunk),
           "time": head_timing(h, w, t, g, chunk, trials, warmup)}
    c = rec["chunked"]
    bound = max(M * chunk, M * H, chunk * H) * acc  # one chunk's fp32 logits, the fp32 dL/dh, one fp32 weight chunk
    if c["largest_allocation_bytes"] > bound:
        result["errors"].append(f"{label}: chunked head allocated {c['largest_allocation_bytes']} B > bound {bound} B")
    saved_bound = M * H * h.element_size() + M * (8 + acc)  # hidden, targets, logsumexp
    if c["saved_bytes"] > saved_bound:
        result["errors"].append(f"{label}: chunked head saved {c['saved_bytes']} B > {saved_bound} B")
    if chunk < V and not c["largest_allocation_bytes"] < rec["full"]["largest_allocation_bytes"]:
        result["errors"].append(f"{label}: chunked head's largest allocation is not below the full head's")
    result["measurements"]["head"][label] = rec


def logprob_equivalence(cfg, result, perturbation=None, device="cpu", trials=7, warmup=2, **_):
    if torch.device(device).type != "cpu":
        raise lab.Blocked("logprob_equivalence runs its equivalence grid on the CPU; --device selects only cpu for now")
    if perturbation not in (None, "raw_logit"):
        raise lab.SetupError(f"logprob_equivalence supports perturbation raw_logit, not {perturbation!r}")
    result.update(comparisons=[], reference_integrity=[], device="cpu",
                  measurements={"reference_logs": {}, "trajectory": {}, "head": {}})
    if perturbation:
        result["perturbation"] = {"spec": perturbation, "effect": "candidates return the raw selected logit"}
    tolerances = {dt: lab.tolerances_for("logprob_equivalence", cfg, dt) for dt in DTYPES}
    for V in VOCABS:
        for dt in DTYPES:
            run_vocab(cfg, result, V, dt, perturbation, tolerances[dt])
    if perturbation is None:
        H, q = F.SHAPES[SHAPE]["hidden_size"], REAL_HEAD
        for dt, dtype in DTYPES.items():
            with F.isolated():
                head_resources(result, f"tiny.V{QWEN3_VOCAB}.{dt}", 6, 16, H, QWEN3_VOCAB, dtype, q["chunk"], "cpu",
                               trials, warmup)
            head_resources(result, f"qwen3_0.6b_head.{dt}", q["rows"], q["R"], q["hidden"], q["vocab"], dtype,
                           q["chunk"], "cpu", trials, warmup)
    result["memory"] = lab.memory("cpu")

