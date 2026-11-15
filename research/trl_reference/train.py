"""TRL GRPOTrainer reference run of the manual balanced B=2 GRPO + LoRA experiment (Qwen3-0.6B, 30 steps).

Run from the repo root:  .venv/bin/python -m research.trl_reference.train
Data, prompt, reward, LoRA model and prompt stream come from the manual implementation; TRL owns generation,
advantages, log-probs, loss, backward and the optimizer. The only TRL override is the sampler (prompt order).
"""
import itertools
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from peft import get_peft_model_state_dict, load_peft_weights
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from qwen_grpo.model import (ALPHA, DTYPE, EOS_IDS, EXPECTED_MODULES, MODEL, RANK, TARGETS, digest, expected_lora_params,
                             lora_model, lora_modules)
from qwen_grpo.sentiment import build_prompts, load_splits, parse_label, reward
from qwen_grpo.training import balanced_subset, batches

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())
G = CONFIG["grpo_config"]["num_generations"]
NAME = "trl_grpo_last2_b2_balanced"
MANUAL_RUN = ROOT / "results" / "grpo_lora_last2_b2_balanced"
MANUAL_TRAINING, MANUAL_PREDICTIONS = MANUAL_RUN / "training.json", MANUAL_RUN / "validation.json"
PACKAGES = ("trl", "torch", "transformers", "peft", "accelerate", "datasets")


def train_examples():
    """The manual run's 32 training examples (16 per label), same seed and order."""
    return balanced_subset(load_splits()["train"], CONFIG["train_samples"], CONFIG["seed"])


def make_dataset(tok, examples):
    """Standard (non-conversational) TRL format: the prompt is the manual chat-templated string, so TRL tokenizes it as is."""
    prompts = build_prompts(tok, [e["text"] for e in examples])
    return Dataset.from_list([{"prompt": p, "ground_truth": e["label"], "example_id": e["id"]} for p, e in zip(prompts, examples)])


def decode_completion(tok, ids):
    """As training.decode: drop only a terminating EOS. TRL's own `completions` use skip_special_tokens=True instead."""
    return tok.decode(ids[:-1] if ids and ids[-1] in EOS_IDS else ids)


def stream(examples, n_batches):
    """First n_batches [POSITIVE, NEGATIVE] pairs of the manual balanced stream -> (pairs, dataset indices, each repeated G times)."""
    pairs = list(itertools.islice(batches(examples, 2, "balanced", CONFIG["seed"]), n_batches))
    index = {e["id"]: i for i, e in enumerate(examples)}
    return pairs, [index[e["id"]] for pair in pairs for e in pair for _ in range(G)]


def make_reward(tok, examples, pairs=None, log=None):
    """TRL reward function: sentiment.reward on the manually decoded completion ids.

    With `pairs`, every call is checked against the expected batch: 2G rows, two G-blocks (POSITIVE prompt, then
    NEGATIVE), each row's prompt and ground truth belonging to its example id.
    """
    by_id = {e["id"]: e for e in examples}
    prompt_of = dict(zip(by_id, build_prompts(tok, [e["text"] for e in by_id.values()])))

    def correctness(prompts, completion_ids, ground_truth, example_id, **kwargs):
        responses = [decode_completion(tok, ids) for ids in completion_ids]
        rewards = [reward(r, g) for r, g in zip(responses, ground_truth)]
        assert all(p == prompt_of[i] and g == by_id[i]["label"] for p, g, i in zip(prompts, ground_truth, example_id))
        if pairs is not None:
            k = len(log)
            assert len(example_id) == 2 * G, len(example_id)
            assert example_id == [e["id"] for e in pairs[k] for _ in range(G)], (k, example_id)
            assert [ground_truth[0], ground_truth[G]] == ["POSITIVE", "NEGATIVE"]
        if log is not None:
            groups = [rewards[b * G : (b + 1) * G] for b in range(len(rewards) // G)]
            log.append({"batch": len(log) + 1, "example_ids": example_id[::G], "labels": ground_truth[::G],
                        "responses": responses, "rewards": rewards, "mean_reward": sum(rewards) / len(rewards),
                        "mixed_groups": sum(len(set(g)) > 1 for g in groups),
                        "invalid_responses": sum(parse_label(r) is None for r in responses),
                        "truncated": sum(ids[-1] not in EOS_IDS for ids in completion_ids)})
        return rewards

    return correctness


class StreamSampler(torch.utils.data.Sampler):
    def __init__(self, order):
        self.order = order

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


class BalancedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with the manual balanced prompt stream instead of RepeatSampler's random prompt order."""

    def __init__(self, *args, sampler_order, **kwargs):
        self.sampler_order = sampler_order
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        return StreamSampler(self.sampler_order)


def grpo_args(output_dir, **overrides):
    return GRPOConfig(output_dir=str(output_dir), **{**CONFIG["grpo_config"], **overrides})


def check_lora(model, dtype=DTYPE):
    """Exactly the six q/v/o adapters of layers 26-27, r*(in+out) fp32 params each, B = 0; base frozen in `dtype`."""
    modules = lora_modules(model)
    assert [n for n, _ in modules] == EXPECTED_MODULES[CONFIG["lora"]["targets"]], [n for n, _ in modules]
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    assert len(params) == 2 * len(modules) and all(".lora_A." in n or ".lora_B." in n for n in params)
    assert sum(p.numel() for p in params.values()) == expected_lora_params(modules)
    assert all(p.dtype == torch.float32 for p in params.values()) and all(p.dtype == dtype for p in frozen.values())
    assert all(m.r["default"] == RANK and m.scaling["default"] == ALPHA / RANK for _, m in modules)
    return params, frozen


class StepLog(TrainerCallback):
    """Per optimizer step: gradient norm before the step, update norm, distance from init, memory."""

    def __init__(self, params, init, reward_log):
        self.params, self.init, self.reward_log, self.rows = params, init, reward_log, []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.grad_norm = math.sqrt(sum(p.grad.double().pow(2).sum().item() for p in self.params.values()))
        self.before = {n: p.detach().clone() for n, p in self.params.items()}

    def on_step_end(self, args, state, control, **kwargs):
        with torch.no_grad():
            self.rows.append({
                "step": state.global_step, "generations_so_far": len(self.reward_log), "grad_norm": self.grad_norm,
                "update_norm": math.sqrt(sum((p - self.before[n]).double().pow(2).sum().item() for n, p in self.params.items())),
                "change_from_init": math.sqrt(sum((p - self.init[n]).double().pow(2).sum().item() for n, p in self.params.items())),
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        r = self.rows[-1]
        print(f"step {r['step']:2d}  R {self.reward_log[-1]['mean_reward']:.3f}  mixed {self.reward_log[-1]['mixed_groups']}  "
              f"|g| {r['grad_norm']:.3e}  |dtheta| {r['update_norm']:.3e}  from init {r['change_from_init']:.3e}  "
              f"peak res {r['peak_reserved_gib']:.2f} GiB")


def manual_rollout_match(reward_log):
    """Compare TRL batch k with manual attempt k (same prompt pair by construction): identical sampled responses?"""
    manual = json.loads(MANUAL_TRAINING.read_text())
    attempts = {g["attempt"]: g for g in manual["training"] + manual["zero_signal"]}
    rows = []
    for r in reward_log:
        m = attempts.get(r["batch"])
        rows.append({"batch": r["batch"], "same_prompts": m is not None and m["prompt_ids"] == r["example_ids"],
                     "same_responses": m is not None and m["responses"] == r["responses"]})
    first_diff = next((x["batch"] for x in rows if not x["same_responses"]), None)
    return {"per_batch": rows, "first_differing_batch": first_diff,
            "identical_batches": sum(x["same_responses"] for x in rows),
            "manual_zero_signal_attempts": [g["attempt"] for g in manual["zero_signal"]]}


def main():
    cfg = CONFIG["grpo_config"]
    torch.manual_seed(CONFIG["seed"])  # LoRA A init, as train_multistep.py
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = lora_model(DTYPE, "cuda", TARGETS[CONFIG["lora"]["targets"]])
    params, frozen = check_lora(model)
    n_trainable = sum(p.numel() for p in params.values())
    assert n_trainable == CONFIG["lora"]["expected_trainable_params"], n_trainable
    init = {n: p.detach().clone() for n, p in params.items()}
    frozen_digest = {n: digest(p) for n, p in frozen.items()}
    torch.cuda.reset_peak_memory_stats()

    examples = train_examples()
    manual_ids = json.loads(MANUAL_TRAINING.read_text())["config"]["train_ids"]
    assert [e["id"] for e in examples] == manual_ids
    pairs, order = stream(examples, cfg["max_steps"])
    reward_log = []
    steplog = StepLog(params, init, reward_log)
    trainer = BalancedGRPOTrainer(
        model=model, reward_funcs=[make_reward(tok, examples, pairs, reward_log)],
        args=grpo_args(ROOT / "research" / "trl_reference" / "checkpoints" / "trainer_output"),
        train_dataset=make_dataset(tok, examples), processing_class=tok, callbacks=[steplog], sampler_order=order)

    # The trainer must have left the model as built: same adapters, values, dtypes; and the intended batch semantics.
    assert trainer.model is model
    check_lora(model)
    assert all(torch.equal(p, init[n]) for n, p in params.items())
    assert trainer.eos_token_ids == EOS_IDS and trainer.num_generations == G and trainer.args.generation_batch_size == 2 * G
    assert (trainer.loss_type, trainer.scale_rewards, trainer.beta, trainer.epsilon_low, trainer.epsilon_high) == ("grpo", "group", 0.0, 0.2, 0.2)
    assert trainer.args.max_grad_norm == 0 and not trainer.args.bf16 and not trainer.args.gradient_checkpointing

    t0 = time.time()
    trainer.train()
    seconds = time.time() - t0
    assert trainer.state.global_step == cfg["max_steps"] == len(reward_log) == len(steplog.rows)  # one generation per step
    assert [r["generations_so_far"] for r in steplog.rows] == list(range(1, cfg["max_steps"] + 1))

    ckpt = ROOT / "research" / "trl_reference" / "checkpoints" / f"{NAME}_step{cfg['max_steps']}"
    model.save_pretrained(ckpt)
    live = get_peft_model_state_dict(model)
    loaded = load_peft_weights(str(ckpt))
    assert live.keys() == loaded.keys() and all(torch.equal(live[k], loaded[k].to(live[k].device)) for k in live)

    zero_signal = [r["batch"] for r in reward_log if r["mixed_groups"] == 0]
    torch.cuda.synchronize()
    result = {
        "model": MODEL, "method": "TRL GRPOTrainer + LoRA", "packages": {p: version(p) for p in PACKAGES},
        "config": CONFIG, "train_ids": manual_ids, "trainable_params": n_trainable,
        "prompt_pairs": [[e["id"] for e in p] for p in pairs],
        "steps": [{**s, **{k: v for k, v in r.items() if k != "batch"}} for s, r in zip(steplog.rows, reward_log)],
        "trl_log_history": trainer.state.log_history,
        "zero_signal_steps": zero_signal,
        "zero_grad_steps": [r["step"] for r in steplog.rows if r["grad_norm"] == 0],
        "mixed_group_counts": {str(k): sum(r["mixed_groups"] == k for r in reward_log) for k in range(3)},
        "frozen_changed": [n for n, p in frozen.items() if digest(p) != frozen_digest[n]],
        "params_equal_init": all(torch.equal(p, init[n]) for n, p in params.items()),
        "checkpoint": {"path": str(ckpt.relative_to(ROOT)), "reloaded_bitwise_equal": True},
        "rollouts_vs_manual": manual_rollout_match(reward_log),
        "train_seconds": seconds,
        "memory": {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                   "device_gib": torch.cuda.mem_get_info()[1] / 2**30},
    }
    out = HERE / "results" / "training.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))

    R = [r["mean_reward"] for r in reward_log]
    print(f"\nsteps {trainer.state.global_step}, zero-signal steps {zero_signal}, zero-grad steps {result['zero_grad_steps']}")
    print(f"mean reward first 10 {sum(R[:10]) / 10:.3f}  last 10 {sum(R[-10:]) / 10:.3f}  mixed-group counts {result['mixed_group_counts']}")
    print(f"frozen changed {len(result['frozen_changed'])}/{len(frozen)}  params equal init {result['params_equal_init']}  "
          f"change from init {steplog.rows[-1]['change_from_init']:.4f}")
    rv = result["rollouts_vs_manual"]
    print(f"rollouts identical to manual for {rv['identical_batches']} batches; first difference at batch {rv['first_differing_batch']}")
    print(f"peak allocated {result['memory']['peak_allocated_gib']:.2f} GiB, peak reserved {result['memory']['peak_reserved_gib']:.2f} GiB")
    if result["frozen_changed"] or result["params_equal_init"]:
        raise SystemExit("FAILED: frozen weights changed or adapter did not move")


if __name__ == "__main__":
    main()
