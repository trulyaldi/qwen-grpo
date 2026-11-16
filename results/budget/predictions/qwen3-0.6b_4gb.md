# Step 0 budget prediction: qwen3-0.6b_4gb

**Overall verdict: PASS** (phases initialization, rollout, training, evaluation, merge; PASS = fits with a 10% margin for unquantified allocations).

Theoretical prediction only: no memory or time below is measured.

Provenance: {'created': '2026-11-16T14:32:29+00:00', 'config': 'configs/budget/qwen3-0.6b.yaml', 'config_sha256': '49d3d3b9e3ebbd8d3f8c946d08782519843323c4c51a049e6245263c313d1e9f', 'git_commit': 'f373e8d8785ab8e4701769d7c6cd0b054efc9bbd', 'kind': 'theoretical prediction (no measurement)'}

## Configuration and assumptions

```yaml
run_name: qwen3-0.6b_4gb
model:
  id: Qwen/Qwen3-0.6B
  weight_dtype: bfloat16
hardware:
  gpu_vram_gib: 4.0
  cpu_ram_gib: 6.7
  cpu_ram_source: 'measured: WSL2 MemTotal of the development host (free -b); ~3.1
    GiB was available at the time'
  pcie_gb_per_s: 12.0
  pcie_source: 'assumption, unmeasured: effective pinned host-to-device copy rate
    (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)'
  copy_compute_overlap: true
runtime:
  cuda_context_gib: 0.5
  allocator_overhead: 0.1
  cpu_process_gib: 2.0
  safety_margin: 0.1
grpo:
  prompt_batch_size: 5
  group_size: 6
  rollout_batch_size: 30
  micro_batch_size: 6
  prompt_tokens: 100
  generation_tokens: 4
  num_iterations: 1
  reference_logprobs: false
lora:
  rank: 32
  targets: last2-attention
  dtype: float32
  optimizer: adamw
streaming:
  resident_decoder_layers: 0
  gpu_layer_buffers: 2
  weight_placement: pinned_cpu
  pinned_allocation: per_layer
  non_decoder_placement: gpu
  loading_copy: shard
  schedule: layer_major
  activation_checkpoints: gpu
  kv_cache: gpu
  logits: response_only
evaluation:
  examples: 512
  batch_size: 32
  score_batch_size: 8
  generation_tokens: 4
  during_training: true
merge:
  strategy: tensorwise
  device: gpu
dflash:
  enabled: false
  estimate: false
  draft_model_id: none
```

## Model metadata (exact, from config.json and safetensors headers)

- `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, source: local safetensors headers
- 28 decoder layers, hidden 1024, intermediate 3072, 16 query / 8 KV heads x 128, vocab 151936, tie_word_embeddings=True
- live parameters 596,049,920; checkpoint 1.503 GB in 1 shard(s)
- embedding (151936, 1024) 296.8 MiB; LM head: tied to embedding; tied copies stored in the checkpoint but aliased when loaded: ['lm_head.weight']
- decoder layer sizes: ['30.0 MiB'] each (1 distinct), total 840.1 MiB
- LoRA: 6 target weights in layers >= 26, 524,288 parameters

## Peak memory by phase

Limits: GPU 4.0 GiB, CPU 6.7 GiB. GPU device = allocated x (1 + 0.1) + CUDA context 0.5 GiB; CPU includes a 2.0 GiB process baseline.

| phase | GPU peak moment | GPU exact | allocated | reserved | device | GPU headroom | CPU peak | CPU headroom | GPU | CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| initialization | weights loaded, before the first forward | 358.8 MiB | 358.8 MiB | 394.6 MiB | 906.6 MiB | 3.115 GiB | 4.275 GiB | 2.425 GiB | PASS | PASS |
| rollout | rollout: prefill, last layer | 690.9 MiB | 790.2 MiB | 869.2 MiB | 1.349 GiB | 2.651 GiB | 2.875 GiB | 3.825 GiB | PASS | PASS |
| training | backward: recompute + backward of the last decoder layer | 389.2 MiB | 441.8 MiB | 486.0 MiB | 998.0 MiB | 3.025 GiB | 2.875 GiB | 3.825 GiB | PASS | PASS |
| evaluation | greedy generation: prefill, last layer | 712.8 MiB | 818.7 MiB | 900.5 MiB | 1.379 GiB | 2.621 GiB | 2.875 GiB | 3.825 GiB | PASS | PASS |
| merge | tensorwise merge on gpu | 26.0 MiB | 26.0 MiB | 28.6 MiB | 540.6 MiB | 3.472 GiB | 2.004 GiB | 4.696 GiB | PASS | PASS |

### initialization

- safetensors are memory-mapped: page cache is not counted as process memory

**weights loaded, before the first forward**: GPU allocated 358.8 MiB, device 906.6 MiB; CPU 4.275 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |
| CPU | temporary loading copy (shard) | 1.400 GiB | estimate | pageable tensors read from safetensors before pinning / GPU upload |

### rollout

- 30 sequences per generate() call, 1 call(s) per rollout batch
- LoRA grads were freed by zero_grad(set_to_none=True); AdamW states persist across steps

**rollout: prefill, last layer**: GPU allocated 790.2 MiB, device 1.349 GiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | KV cache [30 seq x 100 tok x 28 layers] | 328.1 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | hidden state in/out | 11.7 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 52.7 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 34.8 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**rollout: last decode step**: GPU allocated 751.0 MiB, device 1.307 GiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | KV cache [30 seq x 104 tok x 28 layers] | 341.2 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | KV append copy of one layer (DynamicCache torch.cat) | 12.2 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 34.8 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### training

- N = 30 sequences of T = 104 tokens, micro-batches of 6; schedule layer_major
- layers 0..25 hold no LoRA targets: no-grad forward, no checkpoints, no backward sweep; layers 26..27 are checkpointed and swept backward
- no reference-model forward: the loss has beta = 0 (qwen_grpo/grpo.py)

**old log-probs forward (no grad)**: GPU allocated 422.5 MiB, device 976.7 MiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | hidden state in/out | 12.2 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 11.0 MiB | estimate |  |
| GPU | logits + fp32 copy + log_softmax | 36.5 MiB | exact | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**forward: LM head of a micro-batch**: GPU allocated 419.6 MiB, device 973.6 MiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 18.3 MiB | exact |  |
| GPU | logits + fp32 copy + log_softmax | 36.5 MiB | exact | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**backward: log_softmax / LM head**: GPU allocated 433.5 MiB, device 988.9 MiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 18.3 MiB | exact |  |
| GPU | log_softmax backward (saved out, grad out, grad in) + logits grad | 50.4 MiB | estimate | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**backward: recompute + backward of the last decoder layer**: GPU allocated 441.8 MiB, device 998.0 MiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 18.3 MiB | exact |  |
| GPU | hidden-state grads of all 30 seq at the boundary | 6.1 MiB | exact |  |
| GPU | recomputed layer activations (autograd-saved) | 42.9 MiB | estimate |  |
| GPU | grad of hidden state in/out + MLP grads | 9.8 MiB | estimate |  |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### evaluation

- greedy, batches of 32, 4 new tokens; optimizer state kept

**greedy generation: prefill, last layer**: GPU allocated 818.7 MiB, device 1.379 GiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | KV cache [32 seq x 100 tok x 28 layers] | 350.0 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 52.0 KiB | exact |  |
| GPU | hidden state in/out | 12.5 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 56.2 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 37.1 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**greedy generation: last decode step**: GPU allocated 776.9 MiB, device 1.335 GiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | KV cache [32 seq x 104 tok x 28 layers] | 364.0 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 52.0 KiB | exact |  |
| GPU | KV append copy of one layer (DynamicCache torch.cat) | 13.0 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 37.1 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**label scoring forward (no grad)**: GPU allocated 429.3 MiB, device 984.3 MiB; CPU 2.875 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 296.8 MiB | exact |  |
| GPU | final norm | 2.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 60.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 2.0 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 4.0 MiB | exact |  |
| GPU | hidden state in/out | 3.2 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 14.6 MiB | estimate |  |
| GPU | logits + fp32 copy + log_softmax | 48.7 MiB | exact |  |
| CPU | frozen streamed layers x28 (pinned) | 840.1 MiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 55.9 MiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### merge

- assumes a standalone merge (training state and streamed weights released)

**tensorwise merge on gpu**: GPU allocated 26.0 MiB, device 540.6 MiB; CPU 2.004 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | LoRA A, B (all targets) | 2.0 MiB | exact |  |
| GPU | W (native) + W fp32 + B@A fp32 + W' (native), largest target | 24.0 MiB | exact |  |
| CPU | tensorwise read/write buffers | 4.0 MiB | estimate | one tensor |

## Layer residency

- configured: 0 resident decoder layers, 28 streamed, GPU layer buffers 60.0 MiB, CPU weights 896.0 MiB
- non-decoder parameters: GPU-resident: embedding (tied LM head), final norm
- most resident layers with GPU PASS: 28; fewest with CPU PASS: 0; feasible: True; recommended: 28

| resident | streamed | GPU peak (device) | CPU peak | bytes per sweep | GPU | CPU |
|---|---|---|---|---|---|---|
| 0 | 28 | 1.379 GiB | 4.275 GiB | 0.881 GB | PASS | PASS |
| 4 | 24 | 1.508 GiB | 4.150 GiB | 0.755 GB | PASS | PASS |
| 8 | 20 | 1.637 GiB | 4.025 GiB | 0.629 GB | PASS | PASS |
| 12 | 16 | 1.766 GiB | 3.900 GiB | 0.503 GB | PASS | PASS |
| 16 | 12 | 1.895 GiB | 3.775 GiB | 0.378 GB | PASS | PASS |
| 20 | 8 | 2.024 GiB | 3.650 GiB | 0.252 GB | PASS | PASS |
| 24 | 4 | 2.153 GiB | 3.525 GiB | 0.126 GB | PASS | PASS |
| 28 | 0 | 2.217 GiB | 3.400 GiB | 0.000 GB | PASS | PASS |

## PCIe transfers (configured residency)

T_transfer = bytes / 12.0 GB/s (assumption, unmeasured: effective pinned host-to-device copy rate (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)). Bandwidth-only service times, not wall-clock.

| stage | weight sweeps | weights | activations | total | T_transfer | tokens/sweep | critical path | note |
|---|---|---|---|---|---|---|---|---|
| rollout prefill | 1 | 0.881 GB | 0.000 GB | 0.881 GB | 0.073 s | 3000 | >= 0.073 s (max(T_transfer, T_compute); compute unknown) |  |
| rollout decode | 3 | 2.643 GB | 0.000 GB | 2.643 GB | 0.220 s | 30 | >= 0.220 s (max(T_transfer, T_compute); compute unknown) | one weight sweep per generated token after the first |
| old log-probs forward | 1 | 0.881 GB | 0.000 GB | 0.881 GB | 0.073 s | 3120 | >= 0.073 s (max(T_transfer, T_compute); compute unknown) |  |
| training forward (x1 iterations) | 1 | 0.881 GB | 0.000 GB | 0.881 GB | 0.073 s | 3120 | >= 0.073 s (max(T_transfer, T_compute); compute unknown) |  |
| backward recomputation (x1, layers >= 26) | 1 | 0.063 GB | 0.000 GB | 0.063 GB | 0.005 s | 3120 | >= 0.005 s (max(T_transfer, T_compute); compute unknown) | recompute and backward of a layer share one load |
| evaluation (512 examples) | 128 | 112.759 GB | 0.000 GB | 112.759 GB | 9.397 s | 32 | >= 9.397 s (max(T_transfer, T_compute); compute unknown) | 16 generate batches x 4 sweeps + 64 scoring passes |
| merge (gpu) | 0 | 0.044 GB | 0.000 GB | 0.044 GB | 0.004 s | 0 | >= 0.004 s (max(T_transfer, T_compute); compute unknown) | adapted weights to the device and back |

Per GRPO step (rollout + old log-probs + training): 5.349 GB, >= 0.45 s of transfer.

## Uncertainties

- PCIe bandwidth 12.0 GB/s: assumption, unmeasured: effective pinned host-to-device copy rate (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)
- CPU RAM limit 6.7 GiB: measured: WSL2 MemTotal of the development host (free -b); ~3.1 GiB was available at the time
- CUDA context 0.5 GiB, allocator overhead 10%, CPU process baseline 2.0 GiB: estimates, to be replaced by measurements
- layer intermediates assume SDPA attention (no [T, T] score matrix); eager attention adds n_heads * T^2 * 4 B per sequence
- pinned host memory may be capped below physical RAM (NVIDIA's CUDA-on-WSL known limitations list limited pinned system memory); the pinnable amount on this host is unmeasured
- the CPU limit is total RAM; other processes' usage at run time is not subtracted
- compute time is not estimated: critical paths are lower bounds from transfer time alone
- HF generate() internals beyond KV cache and next-token logits are covered only by the safety margin
