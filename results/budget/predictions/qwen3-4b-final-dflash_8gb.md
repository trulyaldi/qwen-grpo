# Step 0 budget prediction: qwen3-4b-final-dflash_8gb

**Overall verdict: PASS** (phases initialization, rollout, training, evaluation, merge; PASS = fits with a 10% margin for unquantified allocations).

Theoretical prediction only: no memory or time below is measured.

Provenance: {'created': '2026-11-18T03:10:29+00:00', 'config': 'configs/budget/qwen3-4b_final_dflash.yaml', 'config_sha256': '97b86dfb5829df1ea778edaf00a1ecc3197e358abfbb09e9df6e028a0f32880f', 'git_commit': '8aa63eef89f1ba978e3ab5dd6985bedd7cd3d623', 'kind': 'theoretical prediction (no measurement)'}

## Configuration and assumptions

```yaml
run_name: qwen3-4b-final-dflash_8gb
model:
  id: Qwen/Qwen3-4B
  weight_dtype: bfloat16
hardware:
  gpu_vram_gib: 7.4506
  cpu_ram_gib: 29.8
  cpu_ram_source: 'requirement, not a measurement: the orchestrator''s minimum host
    RAM for Qwen3-4B (host_requirements in configs/studies/assignment.yaml); the rented
    host''s MemTotal is recorded in environment/host.json'
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
  micro_batch_size: 30
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
  schedule: microbatch_major
  activation_checkpoints: gpu
  kv_cache: gpu
  logits: response_only
evaluation:
  examples: 128
  batch_size: 32
  score_batch_size: 8
  generation_tokens: 4
  during_training: true
merge:
  strategy: tensorwise
  device: cpu
dflash:
  enabled: true
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

Limits: GPU 7.4506 GiB, CPU 29.8 GiB. GPU device = allocated x (1 + 0.1) + CUDA context 0.5 GiB; CPU includes a 2.0 GiB process baseline.

| phase | GPU peak moment | GPU exact | allocated | reserved | device | GPU headroom | CPU peak | CPU headroom | GPU | CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| initialization | weights loaded, before the first forward | 2.106 GiB | 2.106 GiB | 2.316 GiB | 2.816 GiB | 4.634 GiB | 14.714 GiB | 15.086 GiB | PASS | PASS |
| rollout | rollout (DFlash): prefill LM head over all prompt positions | 5.298 GiB | 5.329 GiB | 5.862 GiB | 6.362 GiB | 1.088 GiB | 11.000 GiB | 18.800 GiB | PASS | PASS |
| training | backward: recompute + backward of the last decoder layer | 2.162 GiB | 2.812 GiB | 3.093 GiB | 3.593 GiB | 3.857 GiB | 11.000 GiB | 18.800 GiB | PASS | PASS |
| evaluation | greedy generation: prefill, last layer | 2.553 GiB | 2.794 GiB | 3.073 GiB | 3.573 GiB | 3.877 GiB | 11.000 GiB | 18.800 GiB | PASS | PASS |
| merge | tensorwise merge on cpu | 0.0 KiB | 0.0 KiB | 0.0 KiB | 0.0 KiB | 7.451 GiB | 2.141 GiB | 27.659 GiB | PASS | PASS |

### initialization

- safetensors are memory-mapped: page cache is not counted as process memory

**weights loaded, before the first forward**: GPU allocated 2.106 GiB, device 2.816 GiB; CPU 14.714 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |
| CPU | temporary loading copy (shard) | 3.714 GiB | estimate | pageable tensors read from safetensors before pinning / GPU upload |

### rollout

- 30 sequences per generate() call, 1 call(s) per rollout batch
- LoRA grads were freed by zero_grad(set_to_none=True); AdamW states persist across steps
- DFlash rollouts (z-lab/Qwen3-4B-DFlash-b16): block size 16, 5 target feature layers, 5 drafter layers; verifier = the current LoRA policy through the layer executor (dflash_generate)
- the drafter weights are GPU-resident in every phase (loaded once by make_backend)

**rollout (DFlash): prefill, last layer**: GPU allocated 2.885 GiB, device 3.673 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 120 tok x 36 layers] | 506.2 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | KV prefix copy of one layer (KVCache.update returns a contiguous copy) | 14.1 MiB | estimate |  |
| GPU | hidden state in/out | 29.3 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 167.0 MiB | estimate |  |
| GPU | captured target features [5 target layers x 30 seq x 100 tok x H] | 73.2 MiB | exact |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**rollout (DFlash): prefill LM head over all prompt positions**: GPU allocated 5.329 GiB, device 6.362 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 120 tok x 36 layers] | 506.2 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | captured target features [5 target layers x 30 seq x 100 tok x H] | 73.2 MiB | exact |  |
| GPU | concatenated features (torch.cat) [5 target layers x 30 seq x 100 tok x H] | 73.2 MiB | exact |  |
| GPU | final-norm output | 14.6 MiB | estimate |  |
| GPU | prefill logits [30, 100, V] in the weight dtype | 869.4 MiB | exact | transient of .float() |
| GPU | prefill logits fp32 [30, 100, V] (all prompt positions, kept until the first verify's logits exist) | 1.698 GiB | exact | dflash._target_step applies the LM head to every position; only the last is used |
| GPU | sampling probs of the last position (fp32) | 17.4 MiB | estimate |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**rollout (DFlash): first drafter round (context = whole prompt)**: GPU allocated 4.661 GiB, device 5.627 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 120 tok x 36 layers] | 506.2 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | prefill logits fp32 [30, 100, V] (all prompt positions, kept until the first verify's logits exist) | 1.698 GiB | exact | dflash._target_step applies the LM head to every position; only the last is used |
| GPU | drafter context features [5 target layers x 30 seq x 100 tok x H] | 73.2 MiB | exact |  |
| GPU | fc + hidden_norm output (context) | 29.3 MiB | estimate |  |
| GPU | drafter K/V of context + block, GQA-repeated for attention | 60.9 MiB | estimate |  |
| GPU | drafter context cache [5 layers] + its concatenated copy | 121.9 MiB | estimate |  |
| GPU | draft logits [30, 3, V] (+ fp32 for argmax) | 78.2 MiB | exact |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**rollout (DFlash): first verify pass, LM head over the block**: GPU allocated 4.556 GiB, device 5.512 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [30 seq x 120 tok x 36 layers] | 506.2 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 48.8 KiB | exact |  |
| GPU | KV prefix copy of one layer (KVCache.update returns a contiguous copy) | 14.1 MiB | estimate |  |
| GPU | prefill logits fp32 [30, 100, V] (all prompt positions, kept until the first verify's logits exist) | 1.698 GiB | exact | dflash._target_step applies the LM head to every position; only the last is used |
| GPU | drafter context cache [5 layers] | 60.9 MiB | estimate |  |
| GPU | hidden state in/out (block) | 1.2 MiB | estimate |  |
| GPU | captured block features [5 target layers x 30 seq x 4 tok x H] | 2.9 MiB | exact |  |
| GPU | concatenated block features (torch.cat) [5 target layers x 30 seq x 4 tok x H] | 2.9 MiB | exact |  |
| GPU | block logits [30, 4, V] native + fp32 | 104.3 MiB | exact |  |
| GPU | block sampling probs (fp32) | 69.6 MiB | estimate |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### training

- N = 30 sequences of T = 104 tokens, micro-batches of 30; schedule microbatch_major
- layers 0..33 hold no LoRA targets: no-grad forward, no checkpoints, no backward sweep; layers 34..35 are checkpointed and swept backward
- no reference-model forward: the loss has beta = 0 (qwen_grpo/grpo.py)

**old log-probs forward (no grad)**: GPU allocated 2.491 GiB, device 3.240 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | hidden state in/out | 30.5 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 173.7 MiB | estimate |  |
| GPU | logits + fp32 copy + log_softmax | 182.6 MiB | exact | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**forward: LM head of a micro-batch**: GPU allocated 2.341 GiB, device 3.075 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 45.7 MiB | exact |  |
| GPU | logits + fp32 copy + log_softmax | 182.6 MiB | exact | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**backward: log_softmax / LM head**: GPU allocated 2.409 GiB, device 3.149 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 45.7 MiB | exact |  |
| GPU | log_softmax backward (saved out, grad out, grad in) + logits grad | 252.1 MiB | estimate | response_only: [m, R+1, V=151936] |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**backward: recompute + backward of the last decoder layer**: GPU allocated 2.812 GiB, device 3.593 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | rollout ids/masks + old log-probs + advantages | 49.5 KiB | exact |  |
| GPU | LoRA gradients | 4.1 MiB | exact |  |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | boundary checkpoints [3 x 30 seq x 104 tok] | 45.7 MiB | exact |  |
| GPU | recomputed layer activations (autograd-saved) | 519.1 MiB | estimate |  |
| GPU | grad of hidden state in/out + MLP grads | 146.2 MiB | estimate |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### evaluation

- greedy, batches of 32, 4 new tokens; optimizer state kept

**greedy generation: prefill, last layer**: GPU allocated 2.794 GiB, device 3.573 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [32 seq x 100 tok x 36 layers] | 450.0 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 52.0 KiB | exact |  |
| GPU | hidden state in/out | 31.2 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 178.1 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 37.1 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**greedy generation: last decode step**: GPU allocated 2.620 GiB, device 3.382 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | KV cache [32 seq x 104 tok x 36 layers] | 468.0 MiB | exact |  |
| GPU | token ids + attention mask (int64) | 52.0 KiB | exact |  |
| GPU | KV append copy of one layer (DynamicCache torch.cat) | 13.0 MiB | estimate |  |
| GPU | next-token logits + sampling probs (fp32) | 37.1 MiB | estimate | generate() upcasts the last position's logits; logits processors/softmax copy them |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

**label scoring forward (no grad)**: GPU allocated 2.214 GiB, device 2.936 GiB; CPU 11.000 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| GPU | embedding (= tied LM head) | 741.9 MiB | exact |  |
| GPU | final norm | 5.0 KiB | exact |  |
| GPU | resident decoder layers x0 | 0.0 KiB | exact |  |
| GPU | layer buffers x2 | 385.0 MiB | exact | each sized for the largest streamed layer |
| GPU | LoRA parameters (float32) | 4.1 MiB | exact |  |
| GPU | DFlash drafter weights (z-lab/Qwen3-4B-DFlash-b16, frozen, resident) | 1.001 GiB | exact | safetensors header |
| GPU | AdamW exp_avg + exp_avg_sq | 8.2 MiB | exact |  |
| GPU | hidden state in/out | 8.1 MiB | estimate |  |
| GPU | layer intermediates (no grad) | 46.3 MiB | estimate |  |
| GPU | logits + fp32 copy + log_softmax | 48.7 MiB | exact |  |
| CPU | frozen streamed layers x36 (pinned) | 6.768 GiB | exact |  |
| CPU | pinned allocator power-of-two rounding | 2.232 GiB | estimate | torch CachingHostAllocator rounds each pin_memory() block (per_layer) |

### merge

- assumes a standalone merge (training state and streamed weights released)

**tensorwise merge on cpu**: GPU allocated 0.0 KiB, device 0.0 KiB; CPU 2.141 GiB

| device | item | size | kind | note |
|---|---|---|---|---|
| CPU | tensorwise read/write buffers | 20.0 MiB | estimate | one tensor |
| CPU | LoRA A, B (all targets) | 4.1 MiB | exact |  |
| CPU | W (native) + W fp32 + B@A fp32 + W' (native), largest target | 120.0 MiB | exact |  |

## Layer residency

- configured: 0 resident decoder layers, 36 streamed, GPU layer buffers 385.0 MiB, CPU weights 9.000 GiB
- non-decoder parameters: GPU-resident: embedding (tied LM head), final norm
- most resident layers with GPU PASS: 1; fewest with CPU PASS: 0; feasible: True; recommended: 1

| resident | streamed | GPU peak (device) | CPU peak | bytes per sweep | GPU | CPU |
|---|---|---|---|---|---|---|
| 0 | 36 | 6.362 GiB | 14.714 GiB | 7.267 GB | PASS | PASS |
| 1 | 35 | 6.569 GiB | 14.464 GiB | 7.065 GB | PASS | PASS |
| 6 | 30 | 7.603 GiB | 13.214 GiB | 6.056 GB | FAIL | PASS |
| 12 | 24 | 8.844 GiB | 11.714 GiB | 4.845 GB | FAIL | PASS |
| 18 | 18 | 10.085 GiB | 10.214 GiB | 3.634 GB | FAIL | PASS |
| 24 | 12 | 11.325 GiB | 8.714 GiB | 2.422 GB | FAIL | PASS |
| 30 | 6 | 12.566 GiB | 7.214 GiB | 1.211 GB | FAIL | PASS |
| 36 | 0 | 13.393 GiB | 5.714 GiB | 0.000 GB | FAIL | PASS |

## PCIe transfers (configured residency)

T_transfer = bytes / 12.0 GB/s (assumption, unmeasured: effective pinned host-to-device copy rate (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)). Bandwidth-only service times, not wall-clock.

| stage | weight sweeps | weights | activations | total | T_transfer | tokens/sweep | critical path | note |
|---|---|---|---|---|---|---|---|---|
| rollout prefill | 1 | 7.267 GB | 0.000 GB | 7.267 GB | 0.606 s | 3000 | >= 0.606 s (max(T_transfer, T_compute); compute unknown) |  |
| rollout verify passes (DFlash, upper bound) | 3 | 21.801 GB | 0.000 GB | 21.801 GB | 1.817 s | 30 | >= 1.817 s (max(T_transfer, T_compute); compute unknown) | one weight sweep per verify pass: between 1 (every draft accepted) and 3 (every draft rejected) per call; the drafter is resident (no PCIe) |
| old log-probs forward | 1 | 7.267 GB | 0.000 GB | 7.267 GB | 0.606 s | 3120 | >= 0.606 s (max(T_transfer, T_compute); compute unknown) |  |
| training forward (x1 iterations) | 1 | 7.267 GB | 0.000 GB | 7.267 GB | 0.606 s | 3120 | >= 0.606 s (max(T_transfer, T_compute); compute unknown) |  |
| backward recomputation (x1, layers >= 34) | 1 | 0.404 GB | 0.000 GB | 0.404 GB | 0.034 s | 3120 | >= 0.034 s (max(T_transfer, T_compute); compute unknown) | recompute and backward of a layer share one load |
| evaluation (128 examples) | 32 | 232.545 GB | 0.000 GB | 232.545 GB | 19.379 s | 32 | >= 19.379 s (max(T_transfer, T_compute); compute unknown) | 4 generate batches x 4 sweeps + 16 scoring passes |
| merge (cpu) | 0 | 0.000 GB | 0.000 GB | 0.000 GB | 0.000 s | 0 | >= 0.000 s (max(T_transfer, T_compute); compute unknown) | no PCIe traffic on CPU |

Per GRPO step (rollout + old log-probs + training): 44.006 GB, >= 3.67 s of transfer.

## Uncertainties

- PCIe bandwidth 12.0 GB/s: assumption, unmeasured: effective pinned host-to-device copy rate (PCIe 4.0 x16 peak is ~31.5 GB/s; laptop links and WSL2 are typically far lower)
- CPU RAM limit 29.8 GiB: requirement, not a measurement: the orchestrator's minimum host RAM for Qwen3-4B (host_requirements in configs/studies/assignment.yaml); the rented host's MemTotal is recorded in environment/host.json
- CUDA context 0.5 GiB, allocator overhead 10%, CPU process baseline 2.0 GiB: estimates, to be replaced by measurements
- layer intermediates assume SDPA attention (no [T, T] score matrix); eager attention adds n_heads * T^2 * 4 B per sequence
- pinned host memory may be capped below physical RAM (NVIDIA's CUDA-on-WSL known limitations list limited pinned system memory); the pinnable amount on this host is unmeasured
- the CPU limit is total RAM; other processes' usage at run time is not subtracted
- compute time is not estimated: critical paths are lower bounds from transfer time alone
- HF generate() internals beyond KV cache and next-token logits are covered only by the safety margin
- DFlash rollout: the prefill LM head over all prompt positions and the drafter's tensors are sized from the code of qwen_grpo/execution/dflash.py as it stands; the drafter's attention internals and SDPA workspaces are estimates, and none of it has been measured on Qwen3-4B
- DFlash rollout: acceptance (verify passes per call) changes PCIe traffic, not peak memory; the transfer rows give the reject-everything upper bound
