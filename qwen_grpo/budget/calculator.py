"""Theoretical GPU/CPU memory and PCIe transfer budget of GRPO + LoRA + layer streaming (L2L). Predictions only.

A phase is a list of moments; a moment is the set of tensors that are live at the same time. A phase's peak is the max
over its moments, never the sum of tensors that do not coexist. Every item is tagged:
  exact     shape known (checkpoint metadata or configured batch shapes) and certainly live at that moment
  estimate  formula with an assumption (activation internals, allocator behaviour, runtime footprints)
  unknown   not quantified; `critical` unknowns force verdict UNKNOWN
GPU totals map onto the later measurements: allocated ~ torch.cuda.max_memory_allocated(), reserved = allocated *
(1 + allocator_overhead) ~ max_memory_reserved(), device = reserved + CUDA context ~ nvidia-smi.
Memory is reported in GiB/MiB (2^30, 2^20 B); transfer volumes in GB (10^9 B), matching bandwidth in GB/s.
"""
import math
from dataclasses import asdict, dataclass, field

import yaml

GiB, MiB, GB = 2**30, 2**20, 10**9
DTYPE = {"bfloat16": 2, "float16": 2, "float32": 4}
EXACT, ESTIMATE, UNKNOWN = "exact", "estimate", "unknown"
PASS, FAIL, UNSURE = "PASS", "FAIL", "UNKNOWN"
OVERALL_PHASES = ("initialization", "rollout", "training", "evaluation", "merge")  # DFlash is optional, excluded

SCHEMA = {
    "run_name": str,
    "model": {"id": str, "weight_dtype": str},
    "hardware": {"gpu_vram_gib": float, "cpu_ram_gib": float, "cpu_ram_source": str, "pcie_gb_per_s": float,
                 "pcie_source": str, "copy_compute_overlap": bool},
    "runtime": {"cuda_context_gib": float, "allocator_overhead": float, "cpu_process_gib": float,
                "safety_margin": float},
    "grpo": {"prompt_batch_size": int, "group_size": int, "rollout_batch_size": int, "micro_batch_size": int,
             "prompt_tokens": int, "generation_tokens": int, "num_iterations": int, "reference_logprobs": bool},
    "lora": {"rank": int, "targets": (str, dict), "dtype": str, "optimizer": str},
    "streaming": {"resident_decoder_layers": int, "gpu_layer_buffers": int, "weight_placement": str,
                  "pinned_allocation": str, "non_decoder_placement": str, "loading_copy": str, "schedule": str,
                  "activation_checkpoints": str, "kv_cache": str, "logits": str},
    "evaluation": {"examples": int, "batch_size": int, "score_batch_size": int, "generation_tokens": int,
                   "during_training": bool},
    "merge": {"strategy": str, "device": str},
    "dflash": {"enabled": bool, "estimate": bool, "draft_model_id": str},
}
CHOICES = {
    ("model", "weight_dtype"): ("bfloat16", "float16", "float32"),
    ("lora", "dtype"): ("float32", "bfloat16"),
    ("lora", "optimizer"): ("adamw",),
    ("streaming", "weight_placement"): ("pinned_cpu", "pageable_cpu"),
    ("streaming", "pinned_allocation"): ("per_layer", "per_tensor"),
    ("streaming", "non_decoder_placement"): ("gpu",),
    ("streaming", "loading_copy"): ("tensor", "shard", "model"),
    ("streaming", "schedule"): ("layer_major", "microbatch_major"),
    ("streaming", "activation_checkpoints"): ("gpu", "cpu"),
    ("streaming", "kv_cache"): ("gpu", "cpu"),
    ("streaming", "logits"): ("response_only", "full_sequence"),
    ("merge", "strategy"): ("tensorwise", "shardwise"),
    ("merge", "device"): ("gpu", "cpu"),
}


# ---------------------------------------------------------------------------------------------------- units and sizes

def tensor_bytes(shape, dtype: str) -> int:
    return math.prod(shape) * DTYPE[dtype]


def to_gib(b: int) -> float:
    return b / GiB


def to_gb(b: int) -> float:
    return b / GB


def pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 0


def fmt(b: int | None) -> str:
    if b is None:
        return "unknown"
    if abs(b) >= GiB:
        return f"{b / GiB:.3f} GiB"
    return f"{b / MiB:.1f} MiB" if abs(b) >= MiB else f"{b / 2**10:.1f} KiB"


def lora_numel(targets, rank: int) -> int:
    """A [r, in] + B [out, r] for every adapted weight [out, in]."""
    return sum(rank * (t.shape[0] + t.shape[1]) for t in targets)


def adamw_state_bytes(numel: int, dtype: str = "float32") -> int:
    """exp_avg + exp_avg_sq, same dtype as the parameter (torch.optim.AdamW); the step counters are negligible."""
    return 2 * numel * DTYPE[dtype]


def kv_bytes(n_seq: int, tokens: int, layers: int, n_kv_heads: int, head_dim: int, dtype: str) -> int:
    """K and V of every layer: GQA stores n_kv_heads (not n_attention_heads) heads of head_dim per token."""
    return 2 * n_seq * tokens * layers * n_kv_heads * head_dim * DTYPE[dtype]


def checkpoint_bytes(boundaries: int, n_seq: int, seq_len: int, hidden: int, dtype: str) -> int:
    """Layer-boundary activation checkpoints: one [n_seq, seq_len, hidden] hidden state per stored boundary."""
    return boundaries * n_seq * seq_len * hidden * DTYPE[dtype]


def logits_bytes(n_seq: int, kept: int, scored: int, vocab: int, dtype: str) -> dict:
    """The repo's log-prob path (grpo.response_logprobs): lm_head gives [n, kept, V] in the weight dtype; the `scored`
    response columns are upcast to fp32 and log_softmax'ed (saved for backward). kept = R+1 (response_only,
    logits_to_keep) or T (full_sequence); scored = R either way."""
    native, fp32 = n_seq * kept * vocab * DTYPE[dtype], n_seq * scored * vocab * 4
    return {"logits": native, "fp32_copy": fp32, "log_softmax": fp32,
            "forward_peak": native + 2 * fp32,
            # log_softmax backward: saved output + incoming grad + grad input (fp32), then the grad cast back into a
            # zero-filled [n, kept, V] tensor (slice backward) in the weight dtype
            "backward_peak": 3 * fp32 + native}


# --------------------------------------------------------------------------------------------------------- reporting

@dataclass(frozen=True)
class Item:
    name: str
    bytes: int | None
    kind: str
    note: str = ""
    critical: bool = False


@dataclass
class Moment:
    name: str
    gpu: list[Item]
    cpu: list[Item]


@dataclass
class Transfer:
    name: str
    sweeps: int          # full passes over the streamed layer weights
    weight_bytes: int
    activation_bytes: int  # KV-cache / checkpoint / other activation traffic
    tokens_per_sweep: int  # tokens processed per weight byte moved: the compute reuse of one transfer
    note: str = ""

    @property
    def bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes


@dataclass
class Phase:
    name: str
    moments: list[Moment]
    notes: list[str] = field(default_factory=list)


def known(items) -> int:
    return sum(i.bytes for i in items if i.bytes is not None)


def moment_totals(m: Moment, cfg: dict) -> dict:
    rt = cfg["runtime"]
    exact = sum(i.bytes for i in m.gpu if i.kind == EXACT)
    allocated = known(m.gpu)
    uses_gpu = bool(m.gpu)
    reserved = round(allocated * (1 + rt["allocator_overhead"]))
    device = reserved + (round(rt["cuda_context_gib"] * GiB) if uses_gpu else 0)
    cpu = known(m.cpu) + round(rt["cpu_process_gib"] * GiB)
    return {"gpu_exact": exact, "gpu_allocated": allocated, "gpu_reserved": reserved, "gpu_device": device,
            "cpu": cpu, "critical_unknowns": [i.name for i in m.gpu + m.cpu if i.kind == UNKNOWN and i.critical]}


def verdict(total: int, limit: int, margin: float, critical_unknowns: bool) -> str:
    """FAIL if the known allocations already exceed the limit; UNKNOWN if they fit but inside the safety margin kept for
    unquantified allocations, or a critical allocation is unknown; PASS otherwise."""
    if total > limit:
        return FAIL
    if critical_unknowns or total > limit * (1 - margin):
        return UNSURE
    return PASS


def combine(verdicts) -> str:
    verdicts = list(verdicts)
    return FAIL if FAIL in verdicts else UNSURE if UNSURE in verdicts else PASS


def phase_summary(p: Phase, cfg: dict) -> dict:
    hw, margin = cfg["hardware"], cfg["runtime"]["safety_margin"]
    totals = [moment_totals(m, cfg) for m in p.moments]
    g = max(range(len(totals)), key=lambda k: totals[k]["gpu_device"])
    c = max(range(len(totals)), key=lambda k: totals[k]["cpu"])
    gpu_limit, cpu_limit = round(hw["gpu_vram_gib"] * GiB), round(hw["cpu_ram_gib"] * GiB)
    crit = any(t["critical_unknowns"] for t in totals)
    gv = verdict(totals[g]["gpu_device"], gpu_limit, margin, crit)
    cv = verdict(totals[c]["cpu"], cpu_limit, margin, crit)
    return {"phase": p.name, "gpu_peak_moment": p.moments[g].name, "cpu_peak_moment": p.moments[c].name,
            "gpu_peak": totals[g], "cpu_peak_bytes": totals[c]["cpu"],
            "gpu_headroom_bytes": gpu_limit - totals[g]["gpu_device"], "cpu_headroom_bytes": cpu_limit - totals[c]["cpu"],
            "gpu_verdict": gv, "cpu_verdict": cv, "verdict": combine([gv, cv]), "moment_totals": totals}


# ----------------------------------------------------------------------------------------------------------- config

def validate(cfg: dict, schema: dict = SCHEMA, path: str = "") -> dict:
    if unknown := set(cfg) - set(schema):
        raise ValueError(f"unknown keys {sorted(unknown)} in {path or 'config'}")
    if missing := set(schema) - set(cfg):
        raise ValueError(f"missing keys {sorted(missing)} in {path or 'config'}")
    for key, typ in schema.items():
        value, where = cfg[key], f"{path}.{key}".lstrip(".")
        if isinstance(typ, dict):
            validate(value, typ, where)
            continue
        ok = isinstance(value, typ) or (typ is float and isinstance(value, int) and not isinstance(value, bool))
        if not ok or (typ is int and isinstance(value, bool)):
            raise ValueError(f"{where}: expected {typ}, got {value!r}")
    for (sec, key), allowed in CHOICES.items():
        if cfg.get(sec, {}).get(key) not in allowed and path == "":
            raise ValueError(f"{sec}.{key}: {cfg[sec][key]!r} not in {allowed}")
    if path == "":
        if cfg["dflash"]["enabled"]:
            raise ValueError("dflash.enabled: DFlash is not implemented; only dflash.estimate is available")
        g = cfg["grpo"]
        if min(g["prompt_batch_size"], g["group_size"], g["rollout_batch_size"], g["micro_batch_size"],
               g["prompt_tokens"], g["generation_tokens"], g["num_iterations"]) < 1:
            raise ValueError("grpo sizes must be >= 1")
        if g["reference_logprobs"]:
            raise ValueError("grpo.reference_logprobs: the repo's loss has beta = 0 (no KL); not modelled")
    return cfg


def load_config(path) -> dict:
    with open(path) as f:
        return validate(yaml.safe_load(f))


# ------------------------------------------------------------------------------------------------------- the budget

class Budget:
    """All phases for one configuration and one Architecture (qwen_grpo.budget.architecture)."""

    def __init__(self, cfg: dict, arch, draft_bytes: int | None = None, resident: int | None = None):
        self.cfg, self.arch, self.draft_bytes = cfg, arch, draft_bytes
        s, a = cfg["streaming"], arch
        self.dt, self.w = cfg["model"]["weight_dtype"], DTYPE[cfg["model"]["weight_dtype"]]
        self.L = a.num_hidden_layers
        r = s["resident_decoder_layers"] if resident is None else resident
        if not 0 <= r <= self.L:
            raise ValueError(f"resident_decoder_layers {r}: the model has {self.L} decoder layers")
        if r < self.L and s["gpu_layer_buffers"] < 1:
            raise ValueError("streamed layers need gpu_layer_buffers >= 1")
        # The last r layers stay resident: they hold the LoRA targets, which the backward sweep needs again.
        self.resident = list(range(self.L - r, self.L))
        self.streamed = list(range(self.L - r))
        self.layer_bytes = [a.layer_numel(i) * self.w for i in range(self.L)]
        self.first_grad = a.first_adapted_layer
        self.lora = lora_numel(a.lora_targets, cfg["lora"]["rank"])
        self.lora_b = self.lora * DTYPE[cfg["lora"]["dtype"]]
        g = cfg["grpo"]
        self.N = g["prompt_batch_size"] * g["group_size"]
        self.P, self.R = g["prompt_tokens"], g["generation_tokens"]
        self.S = self.P + self.R

    # -------------------------------------------------------------------------------- static weights and state
    def sweep_bytes(self, layers=None) -> int:
        return sum(self.layer_bytes[i] for i in (self.streamed if layers is None else layers))

    def buffer_bytes(self) -> int:
        if not self.streamed:
            return 0
        return self.cfg["streaming"]["gpu_layer_buffers"] * max(self.layer_bytes[i] for i in self.streamed)

    def gpu_weights(self) -> list[Item]:
        a = self.arch
        tied = a.lm_head is None
        items = [Item("embedding" + (" (= tied LM head)" if tied else ""), a.embed.numel * self.w, EXACT),
                 Item("final norm", a.norm.numel * self.w, EXACT)]
        if not tied:
            items.append(Item("LM head", a.lm_head.numel * self.w, EXACT))
        items += [Item(f"resident decoder layers x{len(self.resident)}", self.sweep_bytes(self.resident), EXACT),
                  Item(f"layer buffers x{self.cfg['streaming']['gpu_layer_buffers'] if self.streamed else 0}",
                       self.buffer_bytes(), EXACT, "each sized for the largest streamed layer"),
                  Item(f"LoRA parameters ({self.cfg['lora']['dtype']})", self.lora_b, EXACT)]
        return items

    def gpu_optimizer(self, grads: bool) -> list[Item]:
        items = [Item("AdamW exp_avg + exp_avg_sq", adamw_state_bytes(self.lora, self.cfg["lora"]["dtype"]), EXACT)]
        return ([Item("LoRA gradients", self.lora_b, EXACT)] if grads else []) + items

    def cpu_weights(self) -> list[Item]:
        s = self.cfg["streaming"]
        exact = self.sweep_bytes()
        label = "pinned" if s["weight_placement"] == "pinned_cpu" else "pageable"
        items = [Item(f"frozen streamed layers x{len(self.streamed)} ({label})", exact, EXACT)]
        if s["weight_placement"] == "pinned_cpu" and self.streamed:
            if s["pinned_allocation"] == "per_layer":
                rounded = sum(pow2_ceil(self.layer_bytes[i]) for i in self.streamed)
            else:
                rounded = sum(pow2_ceil(t.numel * self.w) for i in self.streamed for t in self.arch.layers[i])
            items.append(Item("pinned allocator power-of-two rounding", rounded - exact, ESTIMATE,
                              f"torch CachingHostAllocator rounds each pin_memory() block ({s['pinned_allocation']})"))
        return items

    # ------------------------------------------------------------------------------------------- activations
    def layer_transient(self, tokens: int, grad: bool, adapted: bool = False) -> int:
        """Largest live intermediates of one decoder layer on `tokens` tokens (estimate). No grad: the larger of the
        attention block (q, k, v, output; SDPA, no [T, T] score matrix) and the MLP (gate, up, act*up). With grad
        (checkpoint recomputation): everything autograd saves for the layer: norm inputs/outputs, q/k pre- and
        post-norm/RoPE, v, attention output, gate, up, act, act*up; plus PEFT's fp32 inputs to LoRA A."""
        a = self.arch
        q, kv, h, inter = a.num_attention_heads * a.head_dim, a.num_key_value_heads * a.head_dim, a.hidden_size, \
            a.intermediate_size
        if not grad:
            return tokens * max(2 * q + 2 * kv, 3 * inter) * self.w
        saved = tokens * (4 * h + 4 * q + 3 * kv + 4 * inter) * self.w
        if adapted:
            saved += sum(tokens * (t.shape[1] + self.cfg["lora"]["rank"]) * 4 for t in self.arch.lora_targets
                         if f".{self.L - 1}." in t.name)
        return saved

    def hidden(self, n: int, T: int) -> int:
        return n * T * self.arch.hidden_size * self.w

    def kv_layer(self, n: int, T: int) -> int:
        a = self.arch
        return kv_bytes(n, T, 1, a.num_key_value_heads, a.head_dim, self.dt)

    def kv_items(self, n: int, T: int) -> tuple[list[Item], list[Item]]:
        total = self.kv_layer(n, T) * self.L
        if self.cfg["streaming"]["kv_cache"] == "gpu":
            return [Item(f"KV cache [{n} seq x {T} tok x {self.L} layers]", total, EXACT)], []
        return ([Item("KV cache of current + prefetched layer", 2 * self.kv_layer(n, T), EXACT)],
                [Item(f"offloaded KV cache [{n} x {T} x {self.L}] (pinned)", total, EXACT)])

    def generation_moments(self, n: int, R: int, label: str, base: list[Item]) -> list[Moment]:
        a, P, V = self.arch, self.P, self.arch.vocab_size
        T = P + R
        ids = Item("token ids + attention mask (int64)", 2 * n * T * 8, EXACT)
        sample = Item("next-token logits + sampling probs (fp32)", 2 * n * V * 4, ESTIMATE,
                      "generate() upcasts the last position's logits; logits processors/softmax copy them")
        cpu = self.cpu_weights()
        gk, ck = self.kv_items(n, P)
        prefill = Moment(f"{label}: prefill, last layer", base + gk + [
            ids, Item("hidden state in/out", 2 * self.hidden(n, P), ESTIMATE),
            Item("layer intermediates (no grad)", self.layer_transient(n * P, grad=False), ESTIMATE), sample], cpu + ck)
        gk, ck = self.kv_items(n, T)
        grow = Item("KV append copy of one layer (DynamicCache torch.cat)", self.kv_layer(n, T), ESTIMATE)
        decode = Moment(f"{label}: last decode step", base + gk + [ids, grow, sample], cpu + ck)
        return [prefill, decode]

    def forward_logits_moment(self, label: str, base: list[Item], n_hidden: int, m: int, cpu: list[Item]) -> Moment:
        """No-grad teacher-forced pass (old log-probs, scoring): layer transient on one micro-batch, hidden states of
        `n_hidden` sequences, then the logits path."""
        lg = self.logits(m)
        return Moment(label, base + [
            Item("hidden state in/out", 2 * self.hidden(n_hidden, self.S), ESTIMATE),
            Item("layer intermediates (no grad)", self.layer_transient(m * self.S, grad=False), ESTIMATE),
            Item("logits + fp32 copy + log_softmax", lg["forward_peak"], EXACT, self.logits_note())], cpu)

    def logits(self, m: int, R: int | None = None) -> dict:
        R = self.R if R is None else R
        kept = R + 1 if self.cfg["streaming"]["logits"] == "response_only" else self.P + R
        return logits_bytes(m, kept, R, self.arch.vocab_size, self.dt)

    def logits_note(self) -> str:
        return f"{self.cfg['streaming']['logits']}: [m, {'R+1' if self.cfg['streaming']['logits'] == 'response_only' else 'T'}, V={self.arch.vocab_size}]"

    # ------------------------------------------------------------------------------------------------ phases
    def initialization(self) -> Phase:
        s, a = self.cfg["streaming"], self.arch
        stored = [t for ts in a.layers for t in ts] + a.non_decoder + a.tied_duplicates
        copy = {"tensor": max(t.stored_bytes for t in stored), "shard": max(a.shard_bytes.values()),
                "model": a.checkpoint_bytes}[s["loading_copy"]]
        cpu = self.cpu_weights() + [Item(f"temporary loading copy ({s['loading_copy']})", copy, ESTIMATE,
                                         "pageable tensors read from safetensors before pinning / GPU upload")]
        return Phase("initialization", [Moment("weights loaded, before the first forward", self.gpu_weights(), cpu)],
                     ["safetensors are memory-mapped: page cache is not counted as process memory"])

    def rollout(self) -> Phase:
        g = self.cfg["grpo"]
        n = min(g["rollout_batch_size"], self.N)
        base = self.gpu_weights() + self.gpu_optimizer(grads=False)
        return Phase("rollout", self.generation_moments(n, self.R, "rollout", base),
                     [f"{n} sequences per generate() call, {math.ceil(self.N / n)} call(s) per rollout batch",
                      "LoRA grads were freed by zero_grad(set_to_none=True); AdamW states persist across steps"])

    def training(self) -> Phase:
        s, g, h = self.cfg["streaming"], self.cfg["grpo"], self.arch.hidden_size
        m = min(g["micro_batch_size"], self.N)
        layer_major = s["schedule"] == "layer_major"
        n_ckpt = self.N if layer_major else m
        boundaries = self.L - self.first_grad + 1  # inputs of the adapted span + the final hidden state
        ckpt = checkpoint_bytes(boundaries, n_ckpt, self.S, h, self.dt)
        if s["activation_checkpoints"] == "gpu":
            gck, cck = [Item(f"boundary checkpoints [{boundaries} x {n_ckpt} seq x {self.S} tok]", ckpt, EXACT)], []
        else:
            gck = [Item("current + prefetched checkpoint", 2 * checkpoint_bytes(1, m, self.S, h, self.dt), EXACT)]
            cck = [Item(f"offloaded checkpoints [{boundaries} x {n_ckpt} x {self.S}] (pinned)", ckpt, EXACT)]
        targets = Item("rollout ids/masks + old log-probs + advantages",
                       2 * self.N * self.S * 8 + self.N * self.R * (4 + 1) + self.N * 4, EXACT)
        base = self.gpu_weights() + [targets]
        cpu = self.cpu_weights()
        lg = self.logits(m)
        old = self.forward_logits_moment("old log-probs forward (no grad)", base + self.gpu_optimizer(False),
                                         self.N if layer_major else m, m, cpu)
        trained = base + self.gpu_optimizer(grads=True)
        head_fwd = Moment("forward: LM head of a micro-batch", trained + gck + [
            Item("logits + fp32 copy + log_softmax", lg["forward_peak"], EXACT, self.logits_note())], cpu + cck)
        head_bwd = Moment("backward: log_softmax / LM head", trained + gck + [
            Item("log_softmax backward (saved out, grad out, grad in) + logits grad", lg["backward_peak"], ESTIMATE,
                 self.logits_note())], cpu + cck)
        grad_buf = [Item(f"hidden-state grads of all {self.N} seq at the boundary", self.hidden(self.N, self.S),
                         EXACT)] if layer_major else []
        layer_bwd = Moment("backward: recompute + backward of the last decoder layer", trained + gck + grad_buf + [
            Item("recomputed layer activations (autograd-saved)",
                 self.layer_transient(m * self.S, grad=True, adapted=True), ESTIMATE),
            Item("grad of hidden state in/out + MLP grads", m * self.S * (2 * h + 2 * self.arch.intermediate_size)
                 * self.w, ESTIMATE)], cpu + cck)
        notes = [f"N = {self.N} sequences of T = {self.S} tokens, micro-batches of {m}; schedule {s['schedule']}",
                 f"layers 0..{self.first_grad - 1} hold no LoRA targets: no-grad forward, no checkpoints, "
                 f"no backward sweep; layers {self.first_grad}..{self.L - 1} are checkpointed and swept backward",
                 "no reference-model forward: the loss has beta = 0 (qwen_grpo/grpo.py)"]
        return Phase("training", [old, head_fwd, head_bwd, layer_bwd], notes)

    def evaluation(self) -> Phase:
        e = self.cfg["evaluation"]
        base = self.gpu_weights() + (self.gpu_optimizer(grads=False) if e["during_training"] else [])
        R = e["generation_tokens"]
        moments = self.generation_moments(e["batch_size"], R, "greedy generation", base)
        m = e["score_batch_size"]
        lg = self.logits(m, R)
        moments.append(Moment("label scoring forward (no grad)", base + [
            Item("hidden state in/out", 2 * self.hidden(m, self.P + R), ESTIMATE),
            Item("layer intermediates (no grad)", self.layer_transient(m * (self.P + R), grad=False), ESTIMATE),
            Item("logits + fp32 copy + log_softmax", lg["forward_peak"], EXACT)], self.cpu_weights()))
        return Phase("evaluation", moments, [f"greedy, batches of {e['batch_size']}, {R} new tokens; "
                                             f"optimizer state {'kept' if e['during_training'] else 'absent'}"])

    def merge(self) -> Phase:
        """Standalone after training: W' = W + (alpha/r) B A per adapted weight, one tensor at a time."""
        mg, a = self.cfg["merge"], self.arch
        biggest = max(t.numel for t in a.lora_targets)
        work = Item("W (native) + W fp32 + B@A fp32 + W' (native), largest target", biggest * (2 * self.w + 8),
                    EXACT)
        lora = Item("LoRA A, B (all targets)", self.lora_b, EXACT)
        io = biggest * self.w if mg["strategy"] == "tensorwise" else 2 * max(a.shard_bytes.values())
        cpu = [Item(f"{mg['strategy']} read/write buffers", io, ESTIMATE,
                    "one tensor" if mg["strategy"] == "tensorwise" else "one input shard + its merged copy")]
        gpu = [lora, work] if mg["device"] == "gpu" else []
        if mg["device"] == "cpu":
            cpu += [lora, work]
        return Phase("merge", [Moment(f"{mg['strategy']} merge on {mg['device']}", gpu, cpu)],
                     ["assumes a standalone merge (training state and streamed weights released)"])

    def dflash(self) -> Phase:
        """Optional, not implemented: rollout with a DFlash draft model resident beside the streamed target."""
        rollout = self.rollout().moments[-1]
        draft = (Item(f"draft weights ({self.cfg['dflash']['draft_model_id']})", self.draft_bytes, EXACT)
                 if self.draft_bytes is not None else
                 Item("draft weights", None, UNKNOWN, "draft checkpoint metadata unavailable", critical=True))
        extra = [draft, Item("draft KV cache + target hidden-feature buffers", None, UNKNOWN,
                             "depends on the DFlash implementation; not modelled")]
        return Phase("dflash (optional estimate)", [Moment("rollout last decode step + draft", rollout.gpu + extra,
                                                           rollout.cpu)],
                     ["DFlash is disabled and not implemented; this only sizes its resident draft weights"])

    def phases(self) -> list[Phase]:
        out = [self.initialization(), self.rollout(), self.training(), self.evaluation(), self.merge()]
        return out + ([self.dflash()] if self.cfg["dflash"]["estimate"] else [])

    # ------------------------------------------------------------------------------------------------ PCIe
    def transfers(self) -> list[Transfer]:
        g, s, e, a = self.cfg["grpo"], self.cfg["streaming"], self.cfg["evaluation"], self.arch
        n = min(g["rollout_batch_size"], self.N)
        calls = math.ceil(self.N / n)
        m = min(g["micro_batch_size"], self.N)
        n_mb = math.ceil(self.N / m)
        layer_major = s["schedule"] == "layer_major"
        passes = 1 if layer_major else n_mb
        K, P, R, S = g["num_iterations"], self.P, self.R, self.S
        sweep, grad_sweep = self.sweep_bytes(), self.sweep_bytes([i for i in self.streamed if i >= self.first_grad])
        kv_off = s["kv_cache"] == "cpu"
        kv_layer_tok = self.kv_layer(1, 1)
        prefill_kv = n * P * self.L * kv_layer_tok * calls if kv_off else 0
        decode_kv = sum(n * ((P + t) + 1) * self.L * kv_layer_tok for t in range(1, R)) * calls if kv_off else 0
        ck = checkpoint_bytes(self.L - self.first_grad + 1, self.N, S, a.hidden_size, self.dt) \
            if s["activation_checkpoints"] == "cpu" else 0
        ev_batches, sc_batches = math.ceil(e["examples"] / e["batch_size"]), math.ceil(e["examples"] /
                                                                                      e["score_batch_size"])
        adapted = sum(t.numel for t in a.lora_targets) * self.w
        merge_bytes = 2 * adapted + self.lora_b if self.cfg["merge"]["device"] == "gpu" else 0
        return [
            Transfer("rollout prefill", calls, calls * sweep, prefill_kv, n * P),
            Transfer("rollout decode", calls * (R - 1), calls * (R - 1) * sweep, decode_kv, n,
                     "one weight sweep per generated token after the first"),
            Transfer("old log-probs forward", passes, passes * sweep, 0, self.N * S // passes),
            Transfer(f"training forward (x{K} iterations)", K * passes, K * passes * sweep, K * ck,
                     self.N * S // passes),
            Transfer(f"backward recomputation (x{K}, layers >= {self.first_grad})", K * passes,
                     K * passes * grad_sweep, K * ck, self.N * S // passes,
                     "recompute and backward of a layer share one load"),
            Transfer(f"evaluation ({e['examples']} examples)", ev_batches * e["generation_tokens"] + sc_batches,
                     (ev_batches * e["generation_tokens"] + sc_batches) * sweep, 0, e["batch_size"],
                     f"{ev_batches} generate batches x {e['generation_tokens']} sweeps + {sc_batches} scoring passes"),
            Transfer(f"merge ({self.cfg['merge']['device']})", 0, merge_bytes, 0, 0,
                     "adapted weights to the device and back" if merge_bytes else "no PCIe traffic on CPU"),
        ]

    def transfer_rows(self) -> list[dict]:
        bw = self.cfg["hardware"]["pcie_gb_per_s"] * GB
        overlap = self.cfg["hardware"]["copy_compute_overlap"]
        rows = []
        for t in self.transfers():
            sec = t.bytes / bw
            rows.append({**asdict(t), "bytes": t.bytes, "transfer_s": sec, "compute_s": None,
                         "critical_path": (f">= {sec:.3f} s (max(T_transfer, T_compute); compute unknown)" if overlap
                                           else f">= {sec:.3f} s (T_transfer + T_compute; compute unknown)")})
        return rows


# ------------------------------------------------------------------------------------------------ whole report

def residency_sweep(cfg: dict, arch, draft_bytes=None) -> list[dict]:
    """GPU/CPU peaks and verdicts for every resident-layer count r = 0..L (the last r layers resident)."""
    rows = []
    for r in range(arch.num_hidden_layers + 1):
        b = Budget(cfg, arch, draft_bytes, resident=r)
        summ = [phase_summary(p, cfg) for p in b.phases() if p.name in OVERALL_PHASES]
        rows.append({"resident": r, "streamed": len(b.streamed),
                     "gpu_peak_bytes": max(s["gpu_peak"]["gpu_device"] for s in summ),
                     "cpu_peak_bytes": max(s["cpu_peak_bytes"] for s in summ),
                     "sweep_bytes": b.sweep_bytes(),
                     "gpu_verdict": combine(s["gpu_verdict"] for s in summ),
                     "cpu_verdict": combine(s["cpu_verdict"] for s in summ)})
    return rows


def recommend(rows: list[dict]) -> dict:
    """Most resident layers (fewest bytes per sweep) whose GPU budget passes with the safety margin, and the fewest
    resident layers whose CPU budget passes. A feasible residency needs max_gpu >= min_cpu."""
    gpu_ok = [r["resident"] for r in rows if r["gpu_verdict"] == PASS]
    cpu_ok = [r["resident"] for r in rows if r["cpu_verdict"] == PASS]
    hi, lo = (max(gpu_ok) if gpu_ok else None), (min(cpu_ok) if cpu_ok else None)
    return {"max_resident_gpu_pass": hi, "min_resident_cpu_pass": lo,
            "feasible": hi is not None and lo is not None and lo <= hi,
            "recommended_resident": hi if hi is not None and lo is not None and lo <= hi else None}


def build_report(cfg: dict, arch, draft_bytes: int | None = None) -> dict:
    b = Budget(cfg, arch, draft_bytes)
    phases = b.phases()
    summaries = [phase_summary(p, cfg) for p in phases]
    sweep = residency_sweep(cfg, arch, draft_bytes)
    overall = combine(s["verdict"] for s in summaries if s["phase"] in OVERALL_PHASES)
    transfers = b.transfer_rows()
    return {
        "run_name": cfg["run_name"], "config": cfg, "overall_verdict": overall,
        "model": metadata(arch, b),
        "phases": [{**s, "notes": p.notes, "moments": [asdict(m) for m in p.moments]} for p, s in zip(phases, summaries)],
        "residency": {"configured_resident": len(b.resident), "streamed": len(b.streamed),
                      "resident_layers": b.resident, "gpu_buffer_bytes": b.buffer_bytes(),
                      "cpu_weight_bytes": known(b.cpu_weights()),
                      "non_decoder_placement": "GPU-resident: embedding (tied LM head), final norm",
                      "sweep": sweep, "recommendation": recommend(sweep)},
        "transfers": transfers,
        "transfers_per_grpo_step": {k: sum(t[k] for t in transfers if not t["name"].startswith(("evaluation", "merge")))
                                    for k in ("bytes", "transfer_s")},
        "uncertainties": uncertainties(cfg, phases),
    }


def metadata(arch, b: Budget) -> dict:
    a = arch
    return {"id": a.model_id, "revision": a.revision, "source": a.source, "config": a.config,
            "live_parameters": a.live_numel,
            "embedding": {"shape": a.embed.shape, "bytes": a.embed.numel * b.w},
            "final_norm": {"shape": a.norm.shape, "bytes": a.norm.numel * b.w},
            "lm_head": ("tied to embedding" if a.lm_head is None else {"shape": a.lm_head.shape,
                                                                        "bytes": a.lm_head.numel * b.w}),
            "tied_duplicates_in_checkpoint": [{"name": t.name, "shape": t.shape, "stored_bytes": t.stored_bytes}
                                              for t in a.tied_duplicates],
            "decoder_layer_bytes": b.layer_bytes, "decoder_bytes_total": sum(b.layer_bytes),
            "checkpoint_shards": a.shard_bytes, "checkpoint_bytes": a.checkpoint_bytes,
            "lora_targets": [{"name": t.name, "shape": t.shape} for t in a.lora_targets],
            "lora_parameters": b.lora, "first_adapted_layer": b.first_grad}


def uncertainties(cfg: dict, phases: list[Phase]) -> list[str]:
    hw, rt = cfg["hardware"], cfg["runtime"]
    out = [f"PCIe bandwidth {hw['pcie_gb_per_s']} GB/s: {hw['pcie_source']}",
           f"CPU RAM limit {hw['cpu_ram_gib']} GiB: {hw['cpu_ram_source']}",
           f"CUDA context {rt['cuda_context_gib']} GiB, allocator overhead {rt['allocator_overhead']:.0%}, "
           f"CPU process baseline {rt['cpu_process_gib']} GiB: estimates, to be replaced by measurements",
           "layer intermediates assume SDPA attention (no [T, T] score matrix); eager attention adds "
           "n_heads * T^2 * 4 B per sequence",
           "pinned host memory may be capped below physical RAM (NVIDIA's CUDA-on-WSL known limitations list "
           "limited pinned system memory); the pinnable amount on this host is unmeasured",
           "the CPU limit is total RAM; other processes' usage at run time is not subtracted",
           "compute time is not estimated: critical paths are lower bounds from transfer time alone",
           "HF generate() internals beyond KV cache and next-token logits are covered only by the safety margin"]
    for p in phases:
        for m in p.moments:
            out += [f"{p.name} / {m.name}: {i.name} unknown ({i.note})" for i in m.gpu + m.cpu if i.kind == UNKNOWN]
    return out


# ------------------------------------------------------------------------------------------------------ output files

def table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def render_markdown(rep: dict) -> str:
    cfg, mdl, res = rep["config"], rep["model"], rep["residency"]
    hw, rt = cfg["hardware"], cfg["runtime"]
    L = mdl["config"]["num_hidden_layers"]
    lines = [f"# Step 0 budget prediction: {rep['run_name']}", "",
             f"**Overall verdict: {rep['overall_verdict']}** (phases {', '.join(OVERALL_PHASES)}; "
             f"PASS = fits with a {rt['safety_margin']:.0%} margin for unquantified allocations).",
             "", "Theoretical prediction only: no memory or time below is measured.", "",
             f"Provenance: {rep.get('provenance', {})}", "",
             "## Configuration and assumptions", "", "```yaml", yaml.safe_dump(cfg, sort_keys=False).strip(), "```", "",
             "## Model metadata (exact, from config.json and safetensors headers)", "",
             f"- `{mdl['id']}` revision `{mdl['revision']}`, source: {mdl['source']}",
             f"- {L} decoder layers, hidden {mdl['config']['hidden_size']}, intermediate "
             f"{mdl['config']['intermediate_size']}, {mdl['config']['num_attention_heads']} query / "
             f"{mdl['config']['num_key_value_heads']} KV heads x {mdl['config']['head_dim']}, vocab "
             f"{mdl['config']['vocab_size']}, tie_word_embeddings={mdl['config']['tie_word_embeddings']}",
             f"- live parameters {mdl['live_parameters']:,}; checkpoint {to_gb(mdl['checkpoint_bytes']):.3f} GB in "
             f"{len(mdl['checkpoint_shards'])} shard(s)",
             f"- embedding {tuple(mdl['embedding']['shape'])} {fmt(mdl['embedding']['bytes'])}; LM head: "
             f"{mdl['lm_head'] if isinstance(mdl['lm_head'], str) else fmt(mdl['lm_head']['bytes'])}; "
             f"tied copies stored in the checkpoint but aliased when loaded: "
             f"{[d['name'] for d in mdl['tied_duplicates_in_checkpoint']] or 'none'}",
             f"- decoder layer sizes: {sorted({fmt(b) for b in mdl['decoder_layer_bytes']})} each "
             f"({len(set(mdl['decoder_layer_bytes']))} distinct), total {fmt(mdl['decoder_bytes_total'])}",
             f"- LoRA: {len(mdl['lora_targets'])} target weights in layers >= {mdl['first_adapted_layer']}, "
             f"{mdl['lora_parameters']:,} parameters", "",
             "## Peak memory by phase", "",
             f"Limits: GPU {hw['gpu_vram_gib']} GiB, CPU {hw['cpu_ram_gib']} GiB. GPU device = allocated x "
             f"(1 + {rt['allocator_overhead']}) + CUDA context {rt['cuda_context_gib']} GiB; CPU includes a "
             f"{rt['cpu_process_gib']} GiB process baseline.", ""]
    rows = []
    for p in rep["phases"]:
        g = p["gpu_peak"]
        rows.append([p["phase"], p["gpu_peak_moment"], fmt(g["gpu_exact"]), fmt(g["gpu_allocated"]),
                     fmt(g["gpu_reserved"]), fmt(g["gpu_device"]), fmt(p["gpu_headroom_bytes"]),
                     fmt(p["cpu_peak_bytes"]), fmt(p["cpu_headroom_bytes"]), p["gpu_verdict"], p["cpu_verdict"]])
    lines += [table(["phase", "GPU peak moment", "GPU exact", "allocated", "reserved", "device", "GPU headroom",
                     "CPU peak", "CPU headroom", "GPU", "CPU"], rows), ""]
    for p in rep["phases"]:
        lines += [f"### {p['phase']}", ""] + [f"- {n}" for n in p["notes"]] + [""]
        for m, t in zip(p["moments"], p["moment_totals"]):
            lines += [f"**{m['name']}**: GPU allocated {fmt(t['gpu_allocated'])}, device {fmt(t['gpu_device'])}; "
                      f"CPU {fmt(t['cpu'])}", ""]
            items = [["GPU", i["name"], fmt(i["bytes"]), i["kind"], i["note"]] for i in m["gpu"]] + \
                    [["CPU", i["name"], fmt(i["bytes"]), i["kind"], i["note"]] for i in m["cpu"]]
            lines += [table(["device", "item", "size", "kind", "note"], items), ""]
    rec = res["recommendation"]
    lines += ["## Layer residency", "",
              f"- configured: {res['configured_resident']} resident decoder layers"
              f"{' ' + str(res['resident_layers']) if res['resident_layers'] else ''}, "
              f"{res['streamed']} streamed, GPU layer buffers {fmt(res['gpu_buffer_bytes'])}, CPU weights "
              f"{fmt(res['cpu_weight_bytes'])}",
              f"- non-decoder parameters: {res['non_decoder_placement']}",
              f"- most resident layers with GPU PASS: {rec['max_resident_gpu_pass']}; fewest with CPU PASS: "
              f"{rec['min_resident_cpu_pass']}; feasible: {rec['feasible']}; recommended: "
              f"{rec['recommended_resident']}", ""]
    keep = sorted({0, L, res["configured_resident"]} | {v for v in (rec["max_resident_gpu_pass"],
                                                                     rec["min_resident_cpu_pass"]) if v is not None}
                  | set(range(0, L + 1, max(1, L // 6))))
    lines += [table(["resident", "streamed", "GPU peak (device)", "CPU peak", "bytes per sweep", "GPU", "CPU"],
                    [[r["resident"], r["streamed"], fmt(r["gpu_peak_bytes"]), fmt(r["cpu_peak_bytes"]),
                      f"{to_gb(r['sweep_bytes']):.3f} GB", r["gpu_verdict"], r["cpu_verdict"]]
                     for r in res["sweep"] if r["resident"] in keep]), ""]
    lines += ["## PCIe transfers (configured residency)", "",
              f"T_transfer = bytes / {hw['pcie_gb_per_s']} GB/s ({hw['pcie_source']}). Bandwidth-only service "
              f"times, not wall-clock.", "",
              table(["stage", "weight sweeps", "weights", "activations", "total", "T_transfer", "tokens/sweep",
                     "critical path", "note"],
                    [[t["name"], t["sweeps"], f"{to_gb(t['weight_bytes']):.3f} GB",
                      f"{to_gb(t['activation_bytes']):.3f} GB", f"{to_gb(t['bytes']):.3f} GB",
                      f"{t['transfer_s']:.3f} s", t["tokens_per_sweep"], t["critical_path"], t["note"]]
                     for t in rep["transfers"]]), "",
              f"Per GRPO step (rollout + old log-probs + training): "
              f"{to_gb(rep['transfers_per_grpo_step']['bytes']):.3f} GB, "
              f">= {rep['transfers_per_grpo_step']['transfer_s']:.2f} s of transfer.", "",
              "## Uncertainties", ""] + [f"- {u}" for u in rep["uncertainties"]]
    return "\n".join(lines) + "\n"


def write_prediction(rep: dict, md: str, out_dir) -> tuple:
    """results/budget/predictions/<run_name>.{json,md}; never overwrites: earlier predictions are kept for comparison
    with measurements."""
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    paths = out_dir / f"{rep['run_name']}.json", out_dir / f"{rep['run_name']}.md"
    if existing := [str(p) for p in paths if p.exists()]:
        raise FileExistsError(f"prediction exists, not overwriting: {existing}; change run_name")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(rep, indent=2, default=list) + "\n")
    paths[1].write_text(md)
    return paths
