"""Ticket 11 lab case: depth_residency (H07), GPU memory vs decoder depth for resident and streamed execution.

Tiny Qwen3 models with the same layer shape (hidden 256, intermediate 768, 4 heads, 2 KV heads, head_dim 64, vocab
2000; LoRA on q/v/o of the last two layers) and depth in DEPTHS. For each depth, on one CUDA device:
  resident   the whole model on the GPU (explicit forward, plain layer call)
  sync       L2L: CPU WeightStore + 2 GPU buffers (StreamingLayerExecutor / CheckpointedLayerExecutor)
  async      L2L with prefetching (depth 1, 2 buffers)
measured: PyTorch peak allocated bytes during a no-grad forward of a fixed batch (weights + buffers + activations;
the weight-residency figure, rows' streamed_peak_allocated = sync, resident_peak_allocated = resident) and during a
training step (forward + backward); streamed training memory grows with depth only by the activation checkpoints
(one [N, T, H] boundary per layer), recorded next to their analytic size. H07's rule (lab.hypothesis_result):
SUPPORTED iff the streamed peak is the same at every depth and the resident peak strictly increases.
"""
import gc

import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import lab
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.prefetch import PrefetchingCheckpointedExecutor, PrefetchingLayerExecutor
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.model import lora_config, target_modules

DEPTHS = (2, 4, 8, 16, 32)
SHAPE = dict(hidden_size=256, intermediate_size=768, num_attention_heads=4, num_key_value_heads=2, head_dim=64)
N, T, VOCAB = 4, 64, 2000


def model(depth):
    with F.isolated():
        m = Qwen3ForCausalLM(Qwen3Config(vocab_size=VOCAB, num_hidden_layers=depth, tie_word_embeddings=True, **SHAPE))
        m = get_peft_model(m.eval(), lora_config(target_modules(m, ("q_proj", "v_proj", "o_proj"), 2), 8, 16)).eval()
    return m


def peak(fn, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


def run(m, ex, ids, device):
    def fwd():
        with torch.no_grad():
            explicit_forward(m, ids, None, None, 0, layer_executor=ex)

    def train():
        explicit_forward(m, ids, None, None, 0, layer_executor=ex).float().square().mean().backward()
        m.zero_grad(set_to_none=True)
    return peak(fwd, device), peak(train, device)


def depth_residency(cfg, result, depths=DEPTHS, device="cuda", perturbation=None, **_):
    if not torch.cuda.is_available():
        raise lab.Blocked("depth_residency measures GPU memory; no CUDA device is available")
    device = torch.device("cuda")
    ids = torch.randint(0, VOCAB, (N, T), generator=torch.Generator().manual_seed(0)).to(device)
    rows = []
    for d in depths:
        gc.collect()
        torch.cuda.empty_cache()
        row = {"depth": d, "view_problems": 0, "build_allocated_delta": None}
        res = model(d).to(device)
        row["resident_peak_allocated"], row["resident_train_peak"] = run(res, None, ids, device)
        del res
        gc.collect()
        torch.cuda.empty_cache()
        cand = model(d)
        store = WeightStore.build(cand, cfg.execution.weight_storage)
        try:
            place_resident(cand, store, device)
            row["max_slab_bytes"] = store.max_slab_bytes
            row["planned_buffer_bytes"] = 2 * store.max_slab_bytes
            for name, fwd_ex, trn_ex in (
                    ("streamed", StreamingLayerExecutor, CheckpointedLayerExecutor),
                    ("async", lambda s, b: PrefetchingLayerExecutor(s, b, 1),
                     lambda s, b: PrefetchingCheckpointedExecutor(s, b, 1))):
                f_ex = fwd_ex(store, gpu_buffers(2, store.max_slab_bytes, device))
                t_ex = trn_ex(store, gpu_buffers(2, store.max_slab_bytes, device))
                row[f"{name}_peak_allocated"] = run(cand, f_ex, ids, device)[0]
                row[f"{name}_train_peak"] = run(cand, t_ex, ids, device)[1]
                del f_ex, t_ex
            row["checkpoint_bytes_analytic"] = d * N * T * SHAPE["hidden_size"] * 4
            if problems := store.verify():
                result["errors"] += [f"depth {d}: store: {p}" for p in problems[:3]]
        finally:
            store.close()
        del cand
        rows.append(row)
    asyncs = [r["async_peak_allocated"] for r in rows]
    if len(set(asyncs)) > 1:
        result["errors"].append(f"async streamed forward peak differs across depths: {asyncs}")
    result["measurements"] = {"depths": rows, "shape": SHAPE, "batch": [N, T]}
    result["memory"] = lab.memory(device)
