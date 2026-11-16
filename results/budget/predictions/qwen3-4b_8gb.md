# Step 0 budget prediction: qwen3-4b_8gb

**Overall verdict: FAIL** (phases initialization, rollout, training, evaluation, merge; PASS = fits with a 10% margin for unquantified allocations).

Theoretical prediction only: no memory or time below is measured.

Provenance: {'created': '2026-11-16T14:32:38+00:00', 'config': 'configs/budget/qwen3-4b.yaml', 'config_sha256': 'c70a1df297d40b1b7b2d1ec18466c18a5c5f38b5cfb7d4b36b0fd1f0d8eae644', 'git_commit': 'f373e8d8785ab8e4701769d7c6cd0b054efc9bbd', 'kind': 'theoretical prediction (no measurement)'}

## Configuration and assumptions

```yaml
run_name: qwen3-4b_8gb
model:
  id: Qwen/Qwen3-4B
  weight_dtype: bfloat16
hardware:
  gpu_vram_gib: 8.0
  cpu_ram_gib: 6.7
  cpu_ram_source: 'measured: WSL2 MemTotal of the development host (free -b); the
    assignment machine''s RAM is unknown'
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
  estimate: true
  draft_model_id: z-lab/Qwen3-4B-DFlash-b16
```

## Model metadata (exact, from config.json and safetensors headers)

- `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`, source: remote safetensors headers
- 36 decoder layers, hidden 2560, intermediate 9728, 32 query / 8 KV heads x 128, vocab 151936, tie_word_embeddings=True
- live parameters 4,022,468,096; checkpoint 8.045 GB in 3 shard(s)
- embedding (151936, 2560) 741.9 MiB; LM head: tied to embedding; tied copies stored in the checkpoint but aliased when loaded: none
- decoder layer sizes: ['192.5 MiB'] each (1 distinct), total 6.768 GiB
- LoRA: 6 target weights in layers >= 34, 1,081,344 parameters

## Peak memory by phase

Limits: GPU 8.0 GiB, CPU 6.7 GiB. GPU device = allocated x (1 + 0.1) + CUDA context 0.5 GiB; CPU includes a 2.0 GiB process baseline.

| phase | GPU peak moment | GPU exact | allocated | reserved | device | GPU headroom | CPU peak | CPU headroom | GPU | CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| initialization | weights loaded, before the first forward | 1.105 GiB | 1.105 GiB | 1.215 GiB | 1.715 GiB | 6.285 GiB | 14.714 GiB | -8.014 GiB | PASS | FAIL |
| rollout | rollout: prefill, last layer | 1.525 GiB | 1.750 GiB | 1.925 GiB | 2.425 GiB | 5.575 GiB | 11.000 GiB | -4.300 GiB | PASS | FAIL |
| training | backward: recompute + backward of the last decoder layer | 1.176 GiB | 1.306 GiB | 1.437 GiB | 1.937 GiB | 6.063 GiB | 11.000 GiB | -4.300 GiB | PASS | FAIL |
| evaluation | greedy generation: prefill, last layer | 1.552 GiB | 1.793 GiB | 1.972 GiB | 2.472 GiB | 5.528 GiB | 11.000 GiB | -4.300 GiB | PASS | FAIL |
| merge | tensorwise merge on gpu | 124.1 MiB | 124.1 MiB | 136.5 MiB | 648.5 MiB | 7.367 GiB | 2.020 GiB | 4.680 GiB | PASS | PASS |
| dflash (optional estimate) | rollout last decode step + draft | 2.542 GiB | 2.588 GiB | 2.847 GiB | 3.347 GiB | 4.653 GiB | 11.000 GiB | -4.300 GiB | PASS | FAIL |

### initialization

- safetensors are memory-mapped: page cache is not counted as process memory

**weights loaded, before the first forward**: GPU allocated 1.105 GiB, device 1.715 GiB; CPU 14.714 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |
| CPU | temporary loading copy (shard) | 3.714 GiB | estimate | pageable tensors read from safetensors before pinning / GPU upload |

### rollout

- 30 sequences per generate() call, 1 call(s) per rollout batch
- LoRA grads were freed by zero_grad(set_to_none=True); AdamW states persist across steps

**rollout: prefill, last layer**: GPU allocated 1.750 GiB, device 2.425 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 100 tok x 36 layers] | 421.9 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | hidden state in/out | 29.3 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 167.0 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 34.8 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**rollout: last decode step**: GPU allocated 1.587 GiB, device 2.246 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 104 tok x 36 layers] | 438.8 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | KV append copy of one layer (DynamicCache torch.cat) | 12.2 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 34.8 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### training

- N = 30 sequences of T = 104 tokens, micro-batches of 6; schedule layer_major
- layers 0..33 hold no LoRA targets: no-grad forward, no checkpoints, no backward sweep; layers 34..35 are checkpointed and swept backward
- no reference-model forward: the loss has beta = 0 (qwen_grpo/grpo.py)

**old log-probs forward (no grad)**: GPU allocated 1.212 GiB, device 1.833 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | hidden state in/out | 30.5 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 34.7 MiB | estimate |  |
| GPU | logits + fp32 copy + log_softmax | 36.5 MiB | exact | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**forward: LM head of a micro-batch**: GPU allocated 1.197 GiB, device 1.817 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 45.7 MiB | exact |  |
| GPU | logits + fp32 copy + log_softmax | 36.5 MiB | exact | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**backward: log_softmax / LM head**: GPU allocated 1.211 GiB, device 1.832 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 45.7 MiB | exact |  |
| GPU | log_softmax backward (saved out, grad out, grad in) + logits grad | 50.4 MiB | estimate | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**backward: recompute + backward of the last decoder layer**: GPU allocated 1.306 GiB, device 1.937 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 45.7 MiB | exact |  |
| GPU | hidden-state grads of all 30 seq at the boundary | 15.2 MiB | exact |  |
| GPU | recomputed layer activations (autograd-saved) | 103.8 MiB | estimate |  |
| GPU | grad of hidden state in/out + MLP grads | 29.2 MiB | estimate |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### evaluation

- greedy, batches of 32, 4 new tokens; optimizer state kept

**greedy generation: prefill, last layer**: GPU allocated 1.793 GiB, device 2.472 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [32 seq x 100 tok x 36 layers] | 450.0 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 52.0 KiB | exact |  |
| GPU | hidden state in/out | 31.2 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 178.1 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 37.1 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**greedy generation: last decode step**: GPU allocated 1.619 GiB, device 2.280 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [32 seq x 104 tok x 36 layers] | 468.0 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 52.0 KiB | exact |  |
| GPU | KV append copy of one layer (DynamicCache torch.cat) | 13.0 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 37.1 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**label scoring forward (no grad)**: GPU allocated 1.213 GiB, device 1.835 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | hidden state in/out | 8.1 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 46.3 MiB | estimate |  |
| GPU | logits + fp32 copy + log_softmax | 48.7 MiB | exact |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### merge

- assumes a standalone merge (training state and streamed weights released)

**tensorwise merge on gpu**: GPU allocated 124.1 MiB, device 648.5 MiB; CPU 2.020 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | LoRA A, B (all targets) | 4.1 MiB | exact |  |
| GPU | W (native) + W fp32 + B@A fp32 + W' (native), largest target | 120.0 MiB | exact |  |
| CPU | tensorwise read/write buffers | 20.0 MiB | estimate | one tensor |

### dflash (optional estimate)

- DFlash is disabled and not implemented; this only sizes its resident draft weights

**rollout last decode step + draft**: GPU allocated 2.588 GiB, device 3.347 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 104 tok x 36 layers] | 438.8 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | KV append copy of one layer (DynamicCache torch.cat) | 12.2 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 34.8 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| GPU | draft weights (z-lab/Qwen3-4B-DFlash-b16) | 1.001 GiB | exact |  |
| GPU | draft KV cache + target hidden-feature buffers | unknown | unknown | depends on the DFlash implementation; not modelled |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

## Layer residency

- configured: 0 resident decoder layers, 36 streamed, GPU layer buffers 385.0 MiB, CPU weights 9.000 GiB
- non-decoder parameters: GPU-resident: embedding (tied LM head), final norm
- most resident layers with GPU PASS: 22; fewest with CPU PASS: 35; feasible: False; recommended: None

| resident | streamed | GPU peak (device) | CPU peak | bytes per sweep | GPU | CPU |
|---|---|---|---|---|---|---|
| 0 | 36 | 2.472 GiB | 14.714 GiB | 7.267 GB | PASS | FAIL |
| 6 | 30 | 3.713 GiB | 13.214 GiB | 6.056 GB | PASS | FAIL |
| 12 | 24 | 4.954 GiB | 11.714 GiB | 4.845 GB | PASS | FAIL |
| 18 | 18 | 6.194 GiB | 10.214 GiB | 3.634 GB | PASS | FAIL |
| 22 | 14 | 7.022 GiB | 9.214 GiB | 2.826 GB | PASS | FAIL |
| 24 | 12 | 7.435 GiB | 8.714 GiB | 2.422 GB | UNKNOWN | FAIL |
| 30 | 6 | 8.676 GiB | 7.214 GiB | 1.211 GB | FAIL | FAIL |
| 35 | 1 | 9.710 GiB | 5.964 GiB | 0.202 GB | FAIL | PASS |
| 36 | 0 | 9.503 GiB | 5.714 GiB | 0.000 GB | FAIL | PASS |

## PCIe transfers (configured residency)

T_transfer = bytes / 12.0 GB/s (assumption, unmeasured: effective pinned host-to-device copy rate (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)). Bandwidth-only service times, not wall-clock.

| stage | weight sweeps | weights | activations | total | T_transfer | tokens/sweep | critical path | note |
|---|---|---|---|---|---|---|---|---|
| rollout prefill | 1 | 7.267 GB | 0.000 GB | 7.267 GB | 0.606 s | 3000 | >= 0.606 s (max(T_transfer, T_compute); compute unknown) |  |
| rollout decode | 3 | 21.801 GB | 0.000 GB | 21.801 GB | 1.817 s | 30 | >= 1.817 s (max(T_transfer, T_compute); compute unknown) | one weight sweep per generated token after the first |
| old log-probs forward | 1 | 7.267 GB | 0.000 GB | 7.267 GB | 0.606 s | 3120 | >= 0.606 s (max(T_transfer, T_compute); compute unknown) |  |
| training forward (x1 iterations) | 1 | 7.267 GB | 0.000 GB | 7.267 GB | 0.606 s | 3120 | >= 0.606 s (max(T_transfer, T_compute); compute unknown) |  |
| backward recomputation (x1, layers >= 34) | 1 | 0.404 GB | 0.000 GB | 0.404 GB | 0.034 s | 3120 | >= 0.034 s (max(T_transfer, T_compute); compute unknown) | recompute and backward of a layer share one load |
| evaluation (512 examples) | 128 | 930.178 GB | 0.000 GB | 930.178 GB | 77.515 s | 32 | >= 77.515 s (max(T_transfer, T_compute); compute unknown) | 16 generate batches x 4 sweeps + 64 scoring passes |
| merge (gpu) | 0 | 0.193 GB | 0.000 GB | 0.193 GB | 0.016 s | 0 | >= 0.016 s (max(T_transfer, T_compute); compute unknown) | adapted weights to the device and back |

Per GRPO step (rollout + old log-probs + training): 44.006 GB, >= 3.67 s of transfer.

## Uncertainties

- PCIe bandwidth 12.0 GB/s: assumption, unmeasured: effective pinned host-to-device copy rate (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)
- CPU RAM limit 6.7 GiB: measured: WSL2 MemTotal of the development host (free -b); the assignment machine's RAM is unknown
- CUDA context 0.5 GiB, allocator overhead 10%, CPU process baseline 2.0 GiB: estimates, to be replaced by measurements
- layer intermediates assume SDPA attention (no [T, T] score matrix); eager attention adds n_heads * T^2 * 4 B per sequence
- pinned host memory may be capped below physical RAM (NVIDIA's CUDA-on-WSL known limitations list limited pinned system memory); the pinnable amount on this host is unmeasured
- the CPU limit is total RAM; other processes' usage at run time is not subtracted
- compute time is not estimated: critical paths are lower bounds from transfer time alone
- HF generate() internals beyond KV cache and next-token logits are covered only by the safety margin
- dflash (optional estimate) / rollout last decode step + draft: draft KV cache + target hidden-feature buffers unknown (depends on the DFlash implementation; not modelled)
