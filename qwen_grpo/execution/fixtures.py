"""Fixed tiny-Qwen3 GRPO fixtures: the frozen resident reference that later execution backends are compared against.

A fixture is a seeded random Qwen3ForCausalLM (fp32, eager attention, tied embeddings) with fp32 LoRA adapters on
`last2-attention` whose B factors are nonzero, and one hand-built GRPO rollout batch of B prompts x G completions:
left-padded prompts, responses ending at either EOS id with a padded tail, one truncated response, mixed rewards in
every group. The mask, advantages and old log-probs come from the unchanged resident code (grpo.response_mask,
training.prepare). The rollouts are written by hand, not sampled, so generation is NOT covered by these fixtures.

reference() runs the unchanged resident code on a fixture: HF logits (checked bitwise against explicit_forward),
teacher-forced log-probs, the iteration-0 loss and LoRA gradients, then NUM_ITERATIONS training.grpo_update steps
(AdamW) against the fixed old log-probs, with a learning rate at which clipping binds.

Bitwise reproducibility needs the same global state each time, so build/load/reference run inside isolated(): fp32
default dtype (tests/test_grpo.py sets float64 at import), THREADS CPU threads (reduction order of CPU kernels can
depend on the thread count) and a forked, seeded RNG (no effect on the caller's random stream).

    python -m qwen_grpo.execution.fixtures      writes research/l2l/fixtures/<shape>.pt

Shapes: N = B*G rows, P prompt columns, R response columns, T = P + R, V = VOCAB.
"""
import contextlib
import hashlib
import importlib.metadata
import platform
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.grpo import grpo_loss, response_mask
from qwen_grpo.model import PRESETS, digest, lora_config, lora_modules, target_modules
from qwen_grpo.policy import compute_logprobs
from qwen_grpo.training import check_grouping, grpo_update, prepare

DIR = Path(__file__).resolve().parents[2] / "research" / "l2l" / "fixtures"
SAVED_DIR = DIR  # the repository's saved fixtures (tests may redirect DIR to fixtures built for their environment)

# The two tiny shapes of tests/test_qwen3_forward.py: the structural ratios of Qwen3-0.6B (GQA 2:1, attention width
# 2H) and Qwen3-4B (GQA 4:1, attention width > H).
SHAPES = {
    "0.6b-like": dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16),
    "4b-like": dict(hidden_size=40, intermediate_size=96, num_hidden_layers=6, num_attention_heads=8,
                    num_key_value_heads=2, head_dim=8),
}
VOCAB = 64
EOS_IDS = [1, 2]  # stand-ins for <|im_end|>, <|endoftext|>; PAD = EOS_IDS[1] as in Qwen3, so pads are EOS ids too
PAD = EOS_IDS[1]
B, G, P, R = 2, 3, 6, 4
PROMPT_LENGTHS = [6, 3]  # real tokens of prompt b, left-padded to P
# Response layout per row: (EOS position or None = truncated, EOS id). Covers: truncated, EOS last, EOS first (length 1),
# EOS mid-row with a padded tail, and the second EOS id.
RESPONSES = [(None, None), (3, 1), (1, 1), (0, 1), (2, 1), (1, 2)]
REWARDS = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]  # [B, G], both groups mixed
SEED = 0
RANK, ALPHA, LORA_B_STD = 8, 16, 0.5
LR, CLIP_EPS, NUM_ITERATIONS = 2e-2, 0.2, 3
THREADS = 1


@contextlib.contextmanager
def isolated(seed=SEED):
    """fp32 default dtype, THREADS CPU threads and a forked RNG seeded with `seed`; all restored on exit."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(THREADS)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)


def tiny_model(shape):
    """Random Qwen3 + LoRA (last2-attention, A by PEFT's init, B ~ N(0, LORA_B_STD)); draws from the global RNG."""
    config = Qwen3Config(vocab_size=VOCAB, tie_word_embeddings=True, attn_implementation="eager", **SHAPES[shape])
    model = Qwen3ForCausalLM(config).eval()
    model = get_peft_model(model, lora_config(target_modules(model, *PRESETS["last2-attention"]), RANK, ALPHA)).eval()
    for _, m in lora_modules(model):
        torch.nn.init.normal_(m.lora_B["default"].weight, std=LORA_B_STD)
    return model


def rollout_batch():
    """Hand-built rollouts in policy.generate_rollouts' format; rows b*G .. b*G+G-1 answer prompt b."""
    g = torch.Generator().manual_seed(SEED + 1)
    prompt_mask = torch.stack([torch.arange(P) >= P - n for n in PROMPT_LENGTHS]).long()            # [B, P]
    prompts = torch.randint(3, VOCAB, (B, P), generator=g).masked_fill(prompt_mask == 0, PAD)        # [B, P]
    response_ids = torch.randint(3, VOCAB, (B * G, R), generator=g)                                  # [N, R]
    for row, (at, eos) in enumerate(RESPONSES):
        if at is not None:
            response_ids[row, at] = eos
            response_ids[row, at + 1:] = PAD
    completion_mask = response_mask(response_ids, EOS_IDS)                                           # [N, R] bool
    rollouts = {"input_ids": torch.cat([prompts.repeat_interleave(G, dim=0), response_ids], dim=1),  # [N, T]
                "attention_mask": torch.cat([prompt_mask.repeat_interleave(G, dim=0), completion_mask.long()], dim=1),
                "response_ids": response_ids, "completion_mask": completion_mask, "prompt_len": P}
    check_grouping(rollouts, prompts, G)
    return rollouts, prompts


def build(shape):
    """(model, fixture inputs) from code and seeds alone; call inside isolated()."""
    model = tiny_model(shape)
    rollouts, prompts = rollout_batch()
    rewards = torch.tensor(REWARDS, dtype=torch.float32)
    advantages, old_logp = prepare(model, rollouts, rewards)  # resident: group_advantages + compute_logprobs, no grad
    fixture = {"shape": shape, "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
               "rollouts": rollouts, "prompts": prompts, "rewards": rewards, "advantages": advantages,
               "old_logp": old_logp}
    return model, fixture


def lora_params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def reference(model, fixture):
    """Outputs of the unchanged resident code on the fixture. Mutates the model's LoRA parameters (AdamW steps)."""
    rollouts, adv, old_logp = fixture["rollouts"], fixture["advantages"], fixture["old_logp"]
    ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    params = lora_params(model)
    frozen = {n: digest(p) for n, p in model.named_parameters() if not p.requires_grad}
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits  # [N, T, V]
        if not torch.equal(explicit_forward(model, ids, mask, position_ids), logits):
            raise AssertionError("explicit_forward differs from the HF forward on the fixture")
    logp = compute_logprobs(model, rollouts)                                                          # [N, R]
    loss = grpo_loss(logp, old_logp, adv, rollouts["completion_mask"], CLIP_EPS)
    grads = dict(zip(params, torch.autograd.grad(loss, list(params.values()))))
    initial = {n: p.detach().clone() for n, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=0.0)
    updates = [grpo_update(model, params, optimizer, rollouts, adv, old_logp, CLIP_EPS, k) for k in range(NUM_ITERATIONS)]
    if changed := [n for n, p in model.named_parameters() if not p.requires_grad and digest(p) != frozen[n]]:
        raise AssertionError(f"frozen parameters changed: {changed}")
    return {"logits": logits, "logp": logp.detach(), "loss": loss.detach(), "grads": grads, "initial_lora": initial,
            "updates": updates, "final_lora": {n: p.detach().clone() for n, p in params.items()},
            "optimizer": optimizer.state_dict()}


def metadata():
    return {"seed": SEED, "threads": THREADS, "dtype": "float32", "attn_implementation": "eager", "vocab": VOCAB,
            "eos_ids": EOS_IDS, "pad": PAD, "B": B, "G": G, "P": P, "R": R, "prompt_lengths": PROMPT_LENGTHS,
            "responses": RESPONSES, "rewards": REWARDS, "lora": {"targets": "last2-attention", "r": RANK,
            "alpha": ALPHA, "lora_B_std": LORA_B_STD}, "lr": LR, "clip_eps": CLIP_EPS, "num_iterations": NUM_ITERATIONS,
            "optimizer": "AdamW(weight_decay=0)",
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}}


def path(shape):
    return DIR / f"{shape}.pt"


def host():
    """The CPU the fixture was computed on (CPU kernels and their reduction order can differ between CPUs)."""
    name = platform.processor() or platform.machine()
    try:
        name = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": name, "cpu_capability": torch.backends.cpu.get_cpu_capability()}


def saved_environment_differences(shape):
    """How the repository's saved fixture `shape` (SAVED_DIR: tests may redirect DIR) differs from this environment:
    package versions and, for fixtures that record it, the CPU. [] means its bitwise reference applies."""
    meta = torch.load(SAVED_DIR / f"{shape}.pt", weights_only=True)["meta"]
    out = [f"{p} {meta['packages'].get(p)} (saved) vs {v}" for p, v in metadata()["packages"].items()
           if meta["packages"].get(p) != v]
    if "host" in meta and meta["host"] != host():
        out.append(f"host {meta['host']} (saved) vs {host()}")
    return out


def save(shape):
    with isolated():
        model, fixture = build(shape)
        fixture["reference"] = reference(model, fixture)
    fixture["meta"] = {**metadata(), "shape": SHAPES[shape], "host": host()}
    path(shape).parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, path(shape))
    return path(shape)


def load(shape):
    """(model with the saved weights, saved fixture). The architecture is rebuilt from code, then every tensor of
    the state dict (base and LoRA) is overwritten by the saved one, strictly."""
    fixture = torch.load(path(shape), weights_only=True)
    with isolated():
        model = tiny_model(shape)
    model.load_state_dict(fixture["state"], strict=True)
    return model, fixture


def mismatches(a, b, prefix=""):
    """Keys at which two nested {str: tensor | number | list | dict} structures are not exactly equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{prefix}<keys>"]
        return [m for k in a for m in mismatches(a[k], b[k], f"{prefix}{k}.")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{prefix}<len>"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in mismatches(x, y, f"{prefix}{i}.")]
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        same = a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        return [] if same else [prefix.rstrip(".")]
    return [] if a == b else [prefix.rstrip(".")]


def file_sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    for s in SHAPES:
        print(save(s), file_sha256(path(s)))
