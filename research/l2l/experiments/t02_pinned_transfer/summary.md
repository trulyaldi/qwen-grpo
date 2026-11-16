# t02_pinned_transfer: pinned_transfer (performance)

**Status: PASS** (execution PASS; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)


- commit `721d064a5ac5a87045dfec1ff2608c9381aecac9` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- config `research/l2l/configs/store.yaml` sha256 `e84db9dd6c03770e`; fields used: ['execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `0e4e29d6b58b6f1a`

| side | n | median s | IQR s | median 95% CI s |
|---|---|---|---|---|
| reference | 20 | 4.542e-02 | [0.04433445099675737, 0.04705557624583889] | [0.044635654499870725, 0.046776803996181116] |
| candidate | 20 | 3.068e-02 | [0.030634719247245812, 0.030726137747478788] | [0.03063644999929238, 0.03071218149852939] |

candidate / reference median ratio 0.675, 95% CI [0.6553762844606185, 0.6876816492552715]: candidate faster

measurements: `{"nbytes": 201861632, "qwen3_4b_max_slab_bytes": 201861632, "pin_memory_rounding": [{"requested": 1048577, "held": 2097152}, {"requested": 31457280, "held": 33554432}, {"requested": 201861632, "held": 268435456}], "rlimit_memlock": {"soft": 67108864, "hard": 67108864}, "meminfo": {"mem_available": 1314590720, "rss": 1448071168, "peak_rss": 1680502784}, "capacity_probe": {"cap_bytes": 571400192, "chunk_bytes": 268435456, "registered_bytes": 536870912, "failed_with": null, "mem_available_while_held": 1056952320}}`

memory snapshot (snapshot), evidence grade **diagnostic**:
- CPU RSS 1380.98828125 MiB, high-water mark 1602.65234375 MiB (this process)
- PyTorch allocator, this process: peak allocated 192.5 MiB, peak reserved 194.0 MiB (excludes the CUDA context and non-PyTorch allocations)
- cudaMemGetInfo total - free: 999.3 MiB, device-wide (every process and context on the GPU, not process-specific)
- nvidia-smi raw: compute apps `247323, [N/A]`, memory.used `276` (unreliable under WSL2)
