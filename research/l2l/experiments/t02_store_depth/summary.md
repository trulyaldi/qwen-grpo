# t02_store_depth: store_depth (memory)

**Status: PASS** (execution PASS; hypothesis H07: SUPPORTED, streamed peak 3148800 B at every depth; resident peak [3239936, 6389760, 12689408, 25288704, 50487296])


- commit `721d064a5ac5a87045dfec1ff2608c9381aecac9` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- config `research/l2l/configs/store.yaml` sha256 `e84db9dd6c03770e`; fields used: ['execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `0e4e29d6b58b6f1a`


| depth | max slab B | planned buffers B | streamed peak allocated B | resident peak allocated B | build alloc delta B | view problems |
|---|---|---|---|---|---|---|
| 2 | 1574400 | 3148800 | 3148800 | 3239936 | 0 | 0 |
| 4 | 1574400 | 3148800 | 3148800 | 6389760 | 0 | 0 |
| 8 | 1574400 | 3148800 | 3148800 | 12689408 | 0 | 0 |
| 16 | 1574400 | 3148800 | 3148800 | 25288704 | 0 | 0 |
| 32 | 1574400 | 3148800 | 3148800 | 50487296 | 0 | 0 |

memory snapshot (snapshot), evidence grade **diagnostic**:
- CPU RSS 846.2109375 MiB, high-water mark 883.90234375 MiB (this process)
- PyTorch allocator, this process: peak allocated 48.1 MiB, peak reserved 50.0 MiB (excludes the CUDA context and non-PyTorch allocations)
- cudaMemGetInfo total - free: 805.3 MiB, device-wide (every process and context on the GPU, not process-specific)
- nvidia-smi raw: compute apps `247231, [N/A]`, memory.used `82` (unreliable under WSL2)
