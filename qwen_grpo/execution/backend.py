"""Policy-execution backends (Ticket 07): what training, evaluation and checkpointing need from the policy model, with
a resident and an L2L implementation. The GRPO mathematics stays in grpo.py / training.py: a backend only supplies
how the model is executed.

Interface (both backends):
  generate_rollouts(tok, prompts, G, max_new_tokens)   policy.generate_rollouts (sampling, temperature 1)
  logprobs(model, rollouts)                           the log-prob function for training.prepare / grpo_update and
                                                      evaluation.label_scores (keeps autograd)
  greedy                                              the generator for evaluation.generate (None: model.generate)
  trainable() / frozen_digests() / adapter_state()    model-state access for the optimizer and checkpointing (the
                                                      model's own Parameter objects; PEFT save/load work unchanged)
ResidentBackend: the unchanged resident code (model.generate, policy.compute_logprobs).
L2LBackend: frozen decoder weights in a WeightStore (CPU), everything else on the device (layer_stream.place_resident),
one CheckpointedLayerExecutor for every pass: rollouts and greedy evaluation through execution.generator (never
model.generate), log-probs through output_head.streamed_logprobs with the full head (exact; the chunked head is not
used for training until H03 is resolved), backward through the checkpointed layers. kv_device: where the rollout KV
cache lives.
"""
import torch
from peft import get_peft_model_state_dict

from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.prefetch import PrefetchingCheckpointedExecutor
from qwen_grpo.execution.generator import StreamedGenerator
from qwen_grpo.execution.layer_stream import place_resident
from qwen_grpo.execution.output_head import streamed_logprobs
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.model import digest
from qwen_grpo.policy import compute_logprobs, generate_rollouts


class ResidentBackend:
    name = "resident"
    greedy = None  # evaluation.generate uses model.generate

    def __init__(self, model):
        self.model = model

    def generate_rollouts(self, tok, prompts, G, max_new_tokens):
        return generate_rollouts(self.model, tok, prompts, G=G, max_new_tokens=max_new_tokens)

    def logprobs(self, model, rollouts):
        return compute_logprobs(model, rollouts)

    def trainable(self):
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def frozen_digests(self):
        return {n: digest(p) for n, p in self.model.named_parameters() if not p.requires_grad}

    def adapter_state(self):
        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    def close(self):
        pass


class L2LBackend(ResidentBackend):
    name = "l2l_sync"

    def __init__(self, model, device, weight_storage="pinned", weight_buffers=2, kv_device=None, store=None,
                 prefetch_depth=0, verify=False):
        """`model` with its weights anywhere (typically loaded on the CPU); afterwards its frozen decoder weights live in
        the store and every other tensor on `device`. prefetch_depth 0: synchronous streaming (l2l_sync); >= 1: the
        asynchronous prefetching executor (l2l_async, Ticket 08); verify: its buffer-content check (diagnostic)."""
        super().__init__(model)
        self.store = store or WeightStore.build(model, weight_storage)
        place_resident(model, self.store, device)
        buffers = gpu_buffers(weight_buffers, self.store.max_slab_bytes, device)
        if prefetch_depth:
            self.name = "l2l_async"
            self.executor = PrefetchingCheckpointedExecutor(self.store, buffers, prefetch_depth, verify=verify)
        else:
            self.executor = CheckpointedLayerExecutor(self.store, buffers)
        self.sampler = StreamedGenerator(self.executor, kv_device, do_sample=True)
        self.greedy = StreamedGenerator(self.executor, kv_device, do_sample=False)

    def generate_rollouts(self, tok, prompts, G, max_new_tokens):
        return generate_rollouts(self.model, tok, prompts, G=G, max_new_tokens=max_new_tokens, generator=self.sampler)

    def logprobs(self, model, rollouts):
        if model is not self.model:
            raise RuntimeError("L2LBackend.logprobs: not this backend's model")
        return streamed_logprobs(model, rollouts, None, self.executor)

    def close(self):
        self.store.close()


def make_backend(cfg, model, device):
    """The backend cfg.execution selects (config.unimplemented has refused what is not implemented)."""
    x = cfg.execution
    if x.backend == "resident":
        return ResidentBackend(model)
    if x.backend in ("l2l_sync", "l2l_async"):
        kv = torch.device(device) if x.kv_cache == "gpu" else torch.device("cpu")
        return L2LBackend(model, device, x.weight_storage, x.weight_buffers, kv_device=kv,
                          prefetch_depth=x.prefetch_depth if x.backend == "l2l_async" else 0)
    raise SystemExit(f"execution.backend {x.backend!r} is not implemented")
