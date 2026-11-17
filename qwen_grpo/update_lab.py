"""Ticket 07 lab case: update_equivalence (H05), complete GRPO + LoRA AdamW updates, resident vs L2L backend.

Fixed rollout fixture: fixtures.build(shape), the Ticket 00 batch from code and seeds: input ids, response ids,
completion masks, rewards, group advantages and old-policy log-probs (the resident prepare(); recomputed by prepare()
on the bf16-base model and on a GPU). Its integer tensors must equal the saved Ticket 00 fixture's
(reference_integrity; machine-independent). Both sides start from the same base weights, LoRA weights and AdamW state
(fresh, or a seeded warm state loaded into both), and run ITERATIONS complete training.grpo_update calls (the unchanged
GRPO loss, micro-batch accumulation, AdamW step) against the same fixed old log-probs; only the backend's log-prob
function differs (backend.ResidentBackend / backend.L2LBackend).
Grid: both fixture shapes x {fp32 model, bf16 base with fp32 adapters (as training)} x micro-batch size {all, 1, 2, 4}
x AdamW state {fresh, warm}. Compared per iteration k, L2L vs resident:
  logp.<tag>.it<k>                    current-policy log-probs [N, R] (concatenated micro-batches)
  loss.<tag>.it<k>                    the GRPO loss grpo_update reports
  stats.<tag>.it<k>.<name>            ratio_min, ratio_max, clip_frac, approx_kl, grad_norm, update_norm
  grads.<tag>.it<k>.<param>           every LoRA gradient (accumulated, before the step)
  updates.<tag>.it<k>.<param>         parameter after - before the step
  optimizer.<tag>.it<k>.<param>.<s>   AdamW exp_avg, exp_avg_sq, step after the step
Errors (execution FAIL): a frozen parameter hash that changed or differs between the sides, old log-probs that changed,
a non-finite gradient, an iteration-0 init_diff that is not 0, changed Parameter identities.
Fault injection (perturbation "advantage=<i>"): the candidate's advantage i is negated and shifted by 1.
"""
import copy

import torch

from qwen_grpo import lab
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backend import L2LBackend, ResidentBackend
from qwen_grpo.model import digest
from qwen_grpo.training import grpo_update, prepare

DTYPES = ("float32", "bf16-base")
MICRO = (None, 1, 2, 4)
OPT_STATES = ("fresh", "warm")
ITERATIONS = 3
LR = 5e-2  # large enough that ratios leave 1 and tokens clip within ITERATIONS (checked: power)
STATS = ("ratio_min", "ratio_max", "clip_frac", "approx_kl", "grad_norm", "update_norm")


def fixture_model(shape, variant, device):
    """(resident model on `device`, an identical CPU copy for the L2L side, fixture on `device`), from code and seeds.
    bf16-base: frozen weights in bf16, adapters fp32 (as the trainer). Off the fp32 CPU fixture, the advantages and old
    log-probs are recomputed by the resident prepare() on that model and device, as training would."""
    with F.isolated():
        model, fx = F.build(shape)
    if variant == "bf16-base":
        for p in model.parameters():
            if not p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
    cand = copy.deepcopy(model)
    model.to(device)
    fx["rollouts"] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in fx["rollouts"].items()}
    fx["advantages"], fx["old_logp"] = fx["advantages"].to(device), fx["old_logp"].to(device)
    if variant == "bf16-base" or device.type != "cpu":
        with F.isolated():
            fx["advantages"], fx["old_logp"] = prepare(model, fx["rollouts"], fx["rewards"].to(device))
    return model, cand, fx


def warm_state(optimizer, params, seed=11):
    """A seeded, nontrivial AdamW state (step 3) loaded into `optimizer`."""
    g = torch.Generator().manual_seed(seed)
    state = optimizer.state_dict()
    for i, p in enumerate(params.values()):
        state["state"][i] = {"step": torch.tensor(3.0),
                             "exp_avg": (torch.randn(p.shape, generator=g, dtype=p.dtype) * 1e-3).to(p.device),
                             "exp_avg_sq": (torch.rand(p.shape, generator=g, dtype=p.dtype) * 1e-6).to(p.device)}
    optimizer.load_state_dict(state)


class Probe:
    """The backend's log-prob function, recording each micro-batch's log-probs; and an optimizer step pre-hook that
    copies the accumulated gradients and the parameters before the step."""

    def __init__(self, backend, params, optimizer):
        self.backend, self.params, self.logp, self.grads, self.before = backend, params, [], {}, {}
        optimizer.register_step_pre_hook(self.pre_step)

    def __call__(self, model, rollouts):
        lp = self.backend.logprobs(model, rollouts)
        self.logp.append(lp.detach())
        return lp

    def pre_step(self, optimizer, args, kwargs):
        self.grads = {n: p.grad.clone() for n, p in self.params.items()}
        self.before = {n: p.detach().clone() for n, p in self.params.items()}


def run_side(model, backend, fx, adv, mb, opt_state, tag):
    params = backend.trainable()
    ids = {n: id(p) for n, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=0.0)
    if opt_state == "warm":
        warm_state(optimizer, params)
    probe = Probe(backend, params, optimizer)
    index = {id(p): n for n, p in params.items()}
    out, logs = {}, []
    for k in range(ITERATIONS):
        probe.logp = []
        with F.isolated():
            log = grpo_update(model, params, optimizer, fx["rollouts"], adv, fx["old_logp"], F.CLIP_EPS, k, mb,
                              logprobs=probe)
        logs.append(log)
        p = f"{tag}.it{k}"
        out[f"logp.{p}"] = torch.cat(probe.logp)
        out[f"loss.{p}"] = torch.tensor(log["loss"], dtype=torch.float64)
        out.update({f"stats.{p}.{s}": torch.tensor(float(log[s]), dtype=torch.float64) for s in STATS})
        out.update({f"grads.{p}.{n}": g for n, g in probe.grads.items()})
        out.update({f"updates.{p}.{n}": params[n].detach() - probe.before[n] for n in params})
        for prm, st in optimizer.state.items():
            out.update({f"optimizer.{p}.{index[id(prm)]}.{s}": v.detach().clone() for s, v in st.items()})
    same_ids = {n: id(p) for n, p in backend.trainable().items()} == ids
    return out, logs, same_ids


def run_config(cfg, result, shape, variant, perturbation, tol, device):
    ref_model, cand_model, fx = fixture_model(shape, variant, device)
    old_digest = digest(fx["old_logp"])
    resident = ResidentBackend(ref_model)
    frozen0 = resident.frozen_digests()
    l2l = L2LBackend(cand_model, device, cfg.execution.weight_storage, cfg.execution.weight_buffers)
    try:
        if l2l.frozen_digests() != frozen0:
            result["errors"].append(f"{shape}.{variant}: the sides do not start from the same frozen weights")
        initial = {n: p.detach().clone() for n, p in resident.trainable().items()}
        adv_c = fx["advantages"].clone()
        if perturbation:
            i = int(perturbation.split("=")[1])
            adv_c[i] = -adv_c[i] + 1.0
        for mb in MICRO:
            for opt_state in OPT_STATES:
                tag = f"{shape}.{variant}.mb{mb or 'all'}.{opt_state}"
                for side in (resident, l2l):  # identical LoRA start for every configuration
                    with torch.no_grad():
                        for n, p in side.trainable().items():
                            p.copy_(initial[n])
                ref_t, ref_logs, ok_r = run_side(ref_model, resident, fx, fx["advantages"], mb, opt_state, tag)
                got_t, got_logs, ok_c = run_side(cand_model, l2l, fx, adv_c, mb, opt_state, tag)
                if not (ok_r and ok_c):
                    result["errors"].append(f"{tag}: LoRA Parameter identities changed")
                for name, logs in (("resident", ref_logs), ("l2l", got_logs)):
                    if logs[0]["init_diff"] != 0.0:
                        result["errors"].append(f"{tag}.{name}: iteration-0 init_diff {logs[0]['init_diff']}")
                for key, g in got_t.items():
                    if key.startswith("grads.") and not torch.isfinite(g).all():
                        result["errors"].append(f"{key}: non-finite gradient")
                result["measurements"]["clip_frac"][tag] = [log["clip_frac"] for log in ref_logs]
                result["comparisons"] += [{"model": tag, **row} for row in lab.compare_all(ref_t, got_t, tol)]
        if digest(fx["old_logp"]) != old_digest:
            result["errors"].append(f"{shape}.{variant}: the old log-probs changed during the updates")
        for name, side in (("resident", resident), ("l2l", l2l)):
            if (changed := [n for n, d in side.frozen_digests().items() if d != frozen0[n]]):
                result["errors"].append(f"{shape}.{variant}.{name}: frozen parameters changed: {changed[:5]}")
        if not any(c > 0 for t, cs in result["measurements"]["clip_frac"].items() if t.startswith(f"{shape}.{variant}.")
                   for c in cs[1:]):
            result["errors"].append(f"{shape}.{variant}: no token clipped after iteration 0: no power")
    finally:
        l2l.close()


def saved_rollouts_match(result):
    """The fixture's integer tensors equal the saved Ticket 00 fixture's (built from seeds; machine-independent)."""
    for shape in F.SHAPES:
        if not F.path(shape).exists():
            result["reference_integrity"].append(f"{shape}: saved fixture {F.path(shape)} missing")
            continue
        saved = torch.load(F.path(shape), weights_only=True)
        with F.isolated():
            _, fx = F.build(shape)
        for k in ("input_ids", "attention_mask", "response_ids", "completion_mask"):
            if not torch.equal(saved["rollouts"][k], fx["rollouts"][k]):
                result["reference_integrity"].append(f"{shape}: rollouts {k} differ from the saved fixture")
        if not torch.equal(saved["rewards"], fx["rewards"]):
            result["reference_integrity"].append(f"{shape}: rewards differ from the saved fixture")


def update_equivalence(cfg, result, perturbation=None, device="cpu", **_):
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise lab.Blocked("update_equivalence --device cuda: no CUDA device is available")
    device = torch.empty(0, device=device).device
    if perturbation and not (perturbation.startswith("advantage=") and perturbation.split("=")[1].isdigit()):
        raise lab.SetupError(f"update_equivalence supports perturbation advantage=<i>, not {perturbation!r}")
    result.update(comparisons=[], reference_integrity=[], device=str(device), measurements={"clip_frac": {}})
    if perturbation:
        result["perturbation"] = {"spec": perturbation, "effect": "candidate advantage i := 1 - advantage i"}
    saved_rollouts_match(result)
    tol = lab.tolerances_for("update_equivalence", cfg, "float32")
    for shape in F.SHAPES:
        for variant in DTYPES:
            run_config(cfg, result, shape, variant, perturbation, tol, device)
    result["memory"] = lab.memory(device)
