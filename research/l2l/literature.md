# L2L literature record

Only sections actually read are cited. Each entry names the version read, the sections, what the paper does, what we
take from it, how our system differs, the numerical risks and the validating experiment. Text was extracted with
`pdftotext` from the PDFs downloaded on 2026-11-17 (sources below).

| key | work | version read | source |
|---|---|---|---|
| L2L | Pudipeddi, Mesmakhosroshahi, Xi, Bharadwaj. *Training Large Neural Networks with Constant Memory using a New Execution Algorithm* (2020) | arXiv 2002.05645 PDF as served on 2026-11-17 | https://arxiv.org/pdf/2002.05645 |
| MegaTrain | Yuan, Sun, Sun, Ye. *MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU* (2026) | arXiv 2604.05091 PDF as served on 2026-11-17; code not yet inspected (HEAD `9d88c68` noted) | https://arxiv.org/pdf/2604.05091 |
| Soup ELS | Makazhan. *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB Laptop GPU* | **v3**, `exact_layer_streaming_v3.pdf` (Zenodo record 21918325, 2026-08-13); code not yet inspected (HEAD `e7fe9b2` noted) | https://doi.org/10.5281/zenodo.21918325 |

Not yet read: Chen et al. 2016, FlexGen, DFlash. They are cited only from the tickets that use them (04, 06, 12).

## Ticket 02: where the frozen weights live

### L2L (read: abstract, §1 contributions, §2.1, §2.2 with Algorithms 1–2 and §2.2.1, §2.3, §3.4.1 with Table 5, §4)

- **Algorithm (§2.1, "graph reduction").** The whole model and the optimizer state live on the host, in the "Eager
  Param-Server" (EPS). The device holds only the executing layer and transit buffers ("we only need a 3 layer model to
  be on device"). Forward stashes only layer-boundary activations. Backward recomputes each layer from its stash. Each
  layer's gradients are sent to the EPS and erased from the device, and the EPS updates the weights on the host.
- **§2.2, Algorithm 2.** The micro-batch loop runs *inside* the layer loop ("inverts the minibatch loop and layer
  loop"), so a layer stays on the device while every micro-batch passes through it. The paper calls this "the key
  principle for depth-independent memory sizing".
- **§2.2.1.** A cost model per layer: transfer `X = L/B`, compute `C`. Forward + backward + recompute is
  `N(4uC + 2X)` with u micro-batches per load. Example from the paper: if X = C, then u = 10 brings the transfer overhead
  under 10%.
- **§3.4.1, Table 5.** GPU memory is 3.69 GB for 24, 96 and 384 BERT layers with the stash on the CPU, and grows with
  the stash on the GPU: 5.22 → 6.76 → 9.83 GB for 24 → 48 → 96 layers.
- **Not in the paper:** pinned or page-locked memory. The pinning requirement comes from the assignment, as relayed by
  the owner, and from Soup ELS, not from L2L.
- **What we take.** GPU weight memory bounded by a constant number of layer buffers, independent of depth (H07). The
  layer-major loop for later tickets.
- **How we differ.** The base is frozen and only LoRA trains. So the host holds **no optimizer state and receives no
  weight gradients**: LoRA, its gradients and AdamW stay on the GPU (about 4 × 4.1 MiB for 4B, `docs/l2l/00_budget.md`).
  The host store is read-only during training. L2L's EPS update path and cross mixed precision (§2.3) do not apply.
- **Risk.** A streamed layer must run with exactly the bytes the resident layer has. A wrong layer, or stale buffer
  contents, would change outputs by an amount that may be invisible: Ticket 00 measured that a one-ulp change of a
  single weight changes no output element.
- **Validation.** Digests of every stored tensor and of every GPU view (`WeightStore.verify`, `check_views`), not only
  outputs. H07 is tested by `store_depth`.

### MegaTrain (read: abstract, §1 contributions, §3.1 including Algorithm 1's header, §3.3, §3.4, Appendix A.1)

- **§3.1.** All persistent training state (parameters, Adam moments, accumulated gradients) lives in host memory.
  Device memory "starts empty" except for a pool of layer templates. Weights stream in layer by layer with prefetch of
  the next layer, and the buffer is released after each layer.
- **§3.3 and A.1, layer-contiguous tiling.** All of a layer's state is packed into **one contiguous block, aligned to
  4 KB pages**, so each layer moves in a single burst DMA.
- **§3.3 and A.1, pinned slab recycling.** MegaTrain **does not pin the whole model**: "pinning the entire model's
  parameters would exhaust host physical memory and OS page table resources". The master store is pageable. Two pinned
  staging buffers sized to the largest layer (P_max) are filled by a CPU worker with a just-in-time copy, double-buffered,
  so "host-side pinning overhead [stays] constant (O(P_max)) regardless of model depth".
- **A.1, tied weights.** The embedding and the LM head map to one physical tile ("Structural Aliasing").
- **§3.4, stateless execution.** Layer templates hold no persistent weight pointers. A Bind step maps views of the
  streaming buffer into a template's input slots. There is no CUDA-graph capture, and buffer assignment is explicit per
  layer.
- **What we take.** One contiguous slab per layer, page-aligned, transferred in one copy. Tied embeddings detected and
  kept as one tensor. Views of a flat buffer bound to a layer (Ticket 03).
- **How we differ.**
  1. **We pin the whole frozen store**, per the owner's instruction (the assignment requires pinned CPU memory) and
     like Soup ELS §3.1. This is the design MegaTrain argues against for host-capacity reasons. MegaTrain's
     pageable-store-plus-staging design is the alternative if the target machine's page-locked ceiling is too low (see
     Soup ELS §5.1). Whether it would satisfy "pinned" as the assignment means it is an owner decision.
  2. Our slabs hold **frozen bf16 weights only**. No gradients or Adam moments are in them, and nothing flows back to
     the host.
  3. Inside the slab every tensor offset is aligned to **256 B**, cudaMalloc's alignment, not only the slab base to
     4 KB. GPU views made from a slab are then aligned like ordinary allocations, so kernel selection by pointer
     alignment cannot differ between resident and streamed runs. That would show up as a bf16 non-equivalence in
     Ticket 03.
  4. We bind views by re-pointing the HF/PEFT modules' own parameters (Ticket 03) rather than writing kernel
     templates, so no attention, norm or RoPE arithmetic is reimplemented.

### Soup ELS v3 (read: the v3 front-matter correction, abstract, §1, the contributions list, §3.1, §5.1 setup)

- **v3 correction.** Version 2 claimed that the method is bound by host-to-device transfer. v3 measured it on the
  laptop and **withdrew the claim**: the streamed step runs at 71.3% of the same-session GEMM ceiling, deleting every
  host-to-device byte makes it only 1.4% faster, and the largest streaming-specific cost is the per-layer NF4
  dequantisation at 9.8% (§5.3a, not yet read). "No measured number in version 2 changes"; the interpretation is
  withdrawn.
  - *Implication:* do not assume L2L is transfer-bound. Measure it (Tickets 08 and 11). We use bf16, not NF4, so the
    dequantisation term does not exist here.
- **§3.1, storage tiers.** GPU: N layer buffers, the embedding (tied with the head), the final norm, and the adapters
  with their gradients and optimizer state. CPU RAM: "every decoder layer, page-locked where the machine allows". Peak
  VRAM is bounded by the buffer pool, plus a vocabulary-proportional floor from the resident embedding (§5.5, not yet
  read).
- **§5.1, page-locked ceiling.** On the borrowed Linux box, `ulimit -l` (62.96 GB) and not the 503 GB installed "is
  the real store limit". The laptop's measured page-locked ceiling was 7.12 GB under Windows.
- **Abstract and §1, WDDM.** "Windows/WDDM spills into host memory instead of raising OOM, so a completed run is not
  evidence of fit." WSL2 runs on the Windows driver, so the same caution applies to every GPU-memory claim on this host
  (H11, Ticket 11).
- **What we take.** A fully page-locked frozen store; the GPU-resident set (buffers + embedding + norm + LoRA state);
  the page-locked ceiling as the binding host limit; bit-exactness against resident as the correctness gate.
- **How we differ.** bf16 base, not NF4. GRPO with stochastic rollouts, not supervised fine-tuning. Pinning by
  `cudaHostRegister` on exact-size allocations (we measured `pin_memory=True` rounding up to powers of two). The
  per-tensor alignment above.

## Measurements of this host that the papers do not give (Ticket 02, `research/l2l/experiments/t02_*`)

These are filled in from the saved results; see `docs/l2l/02_weight_store.md`.
