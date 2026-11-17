"""Ticket 06 lab case: generation_equivalence (H04), the streamed generator against resident generation.

Grid: both fixture shapes x eager/sdpa x fp32/bf16, LoRA on every projection of the first and last layers, left-padded
prompts of different lengths (two prompt sets), G = 2 completions per prompt, up to MAX_NEW new tokens, and, per model,
the KV cache on the compute device and on the CPU. Plus one model with the real Qwen3 vocabulary and tokenizer
(non-thinking chat prompts from sentiment.build_prompts), and one whose EOS logit is boosted so rows end in EOS with
a padded tail (required; stopping once every row has finished is covered by tests/test_generator.py).

Paths compared:
  hf        transformers model.generate (resident, DynamicCache): the reference for cached decoding
  resident  execution.generator.generate with the plain layer call
  streamed  execution.generator.generate with layer_stream.StreamingLayerExecutor (L2L), KV cache placed explicitly
  prefix    the resident model's full-prefix forward (no cache) at every step, along the same tokens
Quantities:
  greedy_sequences.<tag>.<path>       greedy ids of resident / streamed vs hf                         (0)
  cached_logits.<tag>.<path>.<k>      next-token fp32 logits at step k (greedy run) vs hf's            (0)
  sampled_probs.<tag>.<path>.<k>      softmax of next-token logits along an hf-SAMPLED trajectory (teacher-forced
                                      through the path) vs hf's own sampling distribution              (0)
  prefix_logits / prefix_probs.<tag>.<path>.<k>  cached decoding vs the full-prefix forward      (rounding bound)
Greedy sequences of the prefix path are compared step by step only where the reference's top-1 / top-2 logit
margin exceeds twice the logit bound (no unstable near-tie); near-ties are counted and reported.
Measurements: same-seed sample identity (hf vs streamed), KV cache bytes vs the analytic size, weight loads per
generated token, and generation time per token (resident vs streamed).
Faults: "stale_position" (decode steps keep the prompt's last position id), "no_padding_mask" (padding is attended).
"""
import contextlib
import time

import torch
from transformers import GenerationConfig

from qwen_grpo import lab
from qwen_grpo import backward_lab  # its build(); module import: backward_lab imports lab
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution import generator as gen
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
ATTN = ("eager", "sdpa")
PROMPTS = {"p9": [9, 5, 9, 2], "p17": [17, 3, 11, 1]}  # real tokens per left-padded prompt
G, MAX_NEW = 2, 8
QWEN3_VOCAB = 151936
SENTENCES = ["A gorgeous, witty film.", "Two hours of my life I will never get back, and the jokes are stale."]
FAULTS = ("stale_position", "no_padding_mask")
QWEN3_EOS, QWEN3_PAD = (151645, 151643), 151643  # <|im_end|>, <|endoftext|> (= qwen_grpo.model.EOS_IDS, pad)
MAX_ABS_LOGIT = 16.0  # the probability bound below assumes |logits| <= this; checked on every reference


def prompts(lengths, V, device, seed):
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    mask = torch.stack([torch.arange(T) >= T - n for n in lengths]).long()
    ids = torch.randint(3, V, (len(lengths), T), generator=g).masked_fill(mask == 0, F.PAD)
    return ids.to(device), mask.to(device)


@contextlib.contextmanager
def fault(spec):
    if spec is None:
        yield
        return
    original_positions, original_mask = gen._positions, gen.create_causal_mask
    if spec == "stale_position":
        stale = []

        def positions(mask):
            pos = original_positions(mask)
            if not stale:
                stale.append(pos.max(dim=1, keepdim=True).values)  # the prefill's last position, per row
                return pos
            return torch.minimum(pos, stale[0])                    # decode never advances past it
        gen._positions = positions
    elif spec == "no_padding_mask":
        gen.create_causal_mask = lambda **k: original_mask(**{**k, "attention_mask": torch.ones_like(k["attention_mask"])})
    else:
        raise lab.SetupError(f"generation_equivalence supports stale_position, no_padding_mask; not {spec!r}")
    try:
        yield
    finally:
        gen._positions, gen.create_causal_mask = original_positions, original_mask


EOS, PAD = F.EOS_IDS, F.PAD  # tiny models; the chat model switches to Qwen3's ids (set_special_tokens)


def set_special_tokens(eos, pad):
    global EOS, PAD
    EOS, PAD = list(eos), pad


def hf_generate(model, ids, mask, do_sample, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, num_return_sequences=G) if do_sample else \
        dict(do_sample=False)
    return model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW, eos_token_id=EOS,
                          pad_token_id=PAD, output_logits=True, return_dict_in_generate=True, **kw)


def full_prefix(model, seq, prompt_mask, steps):
    """Resident full-prefix forward (no cache): next-token fp32 logits after each of the `steps` prefixes."""
    N, P = prompt_mask.shape
    mask = torch.cat([prompt_mask, prompt_mask.new_ones((N, seq.shape[1] - P))], dim=1)
    pos = gen._positions(mask)
    out = []
    with torch.no_grad():
        for k in range(steps):
            out.append(model(input_ids=seq[:, :P + k], attention_mask=mask[:, :P + k], position_ids=pos[:, :P + k],
                             use_cache=False).logits[:, -1].float())
    return out


def compare_model(cfg, result, tag, ref, l2l, store, ids, mask, device, perturbation, tol):
    buffers = gpu_buffers(cfg.execution.weight_buffers, store.max_slab_bytes, device)
    ex = StreamingLayerExecutor(store, buffers)
    ref_t, got_t = {}, {}
    with torch.no_grad():
        hf = hf_generate(ref, ids, mask, do_sample=False)
        hf_logits = [x.float() for x in hf.logits]
        for kv in sorted({str(device), "cpu"}):
            with fault(perturbation):
                runs = {"resident": gen.generate(ref, ids, mask, EOS, PAD, MAX_NEW, 1, do_sample=False,
                                                 kv_device=kv, keep_logits=True)}
            ex.trace.clear()
            with fault(perturbation):
                runs["streamed"] = gen.generate(l2l, ids, mask, EOS, PAD, MAX_NEW, 1, do_sample=False,
                                                layer_executor=ex, kv_device=kv, keep_logits=True)
            loads = len(ex.trace)
            for name, out in runs.items():
                key = f"{tag}.kv_{kv.split(':')[0]}.{name}"
                ref_t[f"greedy_sequences.{key}"], got_t[f"greedy_sequences.{key}"] = hf.sequences, out["sequences"]
                for k, (a, b) in enumerate(zip(hf_logits, out["step_logits"])):
                    ref_t[f"cached_logits.{key}.{k}"], got_t[f"cached_logits.{key}.{k}"] = a, b
                if len(out["step_logits"]) != len(hf_logits):
                    result["errors"].append(f"{key}: {len(out['step_logits'])} steps, hf {len(hf_logits)}")
            n_layers = len(store.slabs)
            steps = runs["streamed"]["steps"]
            result["measurements"]["weight_loads"][f"{tag}.kv_{kv.split(':')[0]}"] = {
                "loads": loads, "steps": steps, "expected": n_layers * steps}
            if loads != n_layers * steps:
                result["errors"].append(f"{tag}: {loads} weight loads for {steps} steps x {n_layers} layers")
        # cached decoding vs the full-prefix forward, along the greedy sequence
        res = runs["resident"]
        result["measurements"]["greedy_steps"][tag] = res["steps"]
        new = res["sequences"][:, ids.shape[1]:]
        eos_hit = torch.isin(new[:, :-1], torch.tensor(EOS, device=new.device)).any(dim=1)
        result["measurements"]["eos_padded_tail_rows"][tag] = int(eos_hit.sum())
        full = full_prefix(ref, res["sequences"], mask, len(res["step_logits"]))
        near_ties = 0
        for k, (a, b) in enumerate(zip(full, res["step_logits"])):
            if a.abs().max() > MAX_ABS_LOGIT:
                result["errors"].append(f"{tag}.{k}: |logit| {a.abs().max():.1f} > {MAX_ABS_LOGIT}: probability bound void")
            ref_t[f"prefix_logits.{tag}.resident.{k}"], got_t[f"prefix_logits.{tag}.resident.{k}"] = a, b
            ref_t[f"prefix_probs.{tag}.resident.{k}"], got_t[f"prefix_probs.{tag}.resident.{k}"] = a.softmax(-1), b.softmax(-1)
            top2 = a.topk(2, dim=-1).values
            stable = (top2[:, 0] - top2[:, 1]) > 2 * tol["prefix_logits"]["rtol"] * a.abs().max()
            near_ties += int((~stable).sum())
            if not torch.equal(a.argmax(-1)[stable], b.argmax(-1)[stable]):
                result["comparisons"].append({"quantity": "greedy_sequences", "name": f"greedy_sequences.{tag}.prefix.{k}",
                                              "pass": False, "reason": "greedy token differs without a near-tie"})
        result["measurements"]["near_ties"][tag] = near_ties
        # sampling: hf's distribution along its own sampled trajectory, teacher-forced through the other paths
        hs = hf_generate(ref, ids, mask, do_sample=True, seed=1234)
        P = ids.shape[1]
        forced = hs.sequences[:, P:]
        for name, model, e in (("resident", ref, None), ("streamed", l2l, ex)):
            with fault(perturbation):
                out = gen.generate(model, ids, mask, EOS, PAD, forced.shape[1], G, layer_executor=e,
                                   forced=forced, keep_logits=True)
            for k, (a, b) in enumerate(zip(hs.logits, out["step_logits"])):
                ref_t[f"sampled_probs.{tag}.{name}.{k}"] = a.float().softmax(-1)
                got_t[f"sampled_probs.{tag}.{name}.{k}"] = b.softmax(-1)
        torch.manual_seed(1234)
        with fault(perturbation):
            same = gen.generate(l2l, ids, mask, EOS, PAD, MAX_NEW, G, do_sample=True, layer_executor=ex)
        result["measurements"]["same_seed_samples_identical"][tag] = bool(torch.equal(same["sequences"], hs.sequences))
        # memory and time
        cfgm = ref.config
        head_dim = getattr(cfgm, "head_dim", None) or cfgm.hidden_size // cfgm.num_attention_heads
        analytic = (2 * cfgm.num_hidden_layers * ids.shape[0] * cfgm.num_key_value_heads * (P + MAX_NEW) * head_dim
                    * ref.get_input_embeddings().weight.element_size())  # K and V; B rows: the greedy run (G = 1)
        if res["kv_cache_bytes"] != analytic:
            result["errors"].append(f"{tag}: KV cache {res['kv_cache_bytes']} B != analytic {analytic} B")
        timing = {}
        for name, model, e in (("resident", ref, None), ("streamed", l2l, ex)):
            sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.memory_allocated()
            sync()
            t0 = time.perf_counter()
            out = gen.generate(model, ids, mask, EOS, PAD, MAX_NEW, G, do_sample=False, layer_executor=e)
            sync()
            timing[name] = {"seconds_per_token": (time.perf_counter() - t0) / max(out["steps"], 1),
                            "rows": ids.shape[0] * G, "kv_cache_bytes": out["kv_cache_bytes"]}
            if device.type == "cuda":
                timing[name]["cuda_peak_above_start_bytes"] = torch.cuda.max_memory_allocated() - start
        result["measurements"]["generation"][tag] = {"kv_cache_analytic_bytes_greedy_B_rows": analytic, **timing}
    result["comparisons"] += [{"model": tag, **row} for row in lab.compare_all(ref_t, got_t, tol)]


def build_pair(shape, attn, dtype, vocab, device, storage, eos_boost=False):
    with F.isolated():
        ref = backward_lab.build(shape, attn, dtype, [0, F.SHAPES[shape]["num_hidden_layers"] - 1], vocab=vocab)
        l2l = backward_lab.build(shape, attn, dtype, [0, F.SHAPES[shape]["num_hidden_layers"] - 1], vocab=vocab)
    if eos_boost:  # the tied head row of EOS id 1 scaled up: rows emit EOS early
        with torch.no_grad():
            ref.get_input_embeddings().weight[F.EOS_IDS[0]] *= 24
    l2l.load_state_dict(ref.state_dict(), strict=True)
    for m in (ref, l2l):
        m.generation_config = GenerationConfig(eos_token_id=EOS, pad_token_id=PAD)
    store = WeightStore.build(l2l, storage)
    place_resident(l2l, store, device)
    return ref.to(device), l2l, store


def models():
    """(shape, attention, dtype, vocabulary, EOS-boosted) of every model in the grid."""
    out = [(s, a, dt, 64, False) for s in F.SHAPES for a in ATTN for dt in DTYPES]
    return out + [("0.6b-like", "sdpa", "float32", 64, True), ("0.6b-like", "sdpa", "bfloat16", QWEN3_VOCAB, False)]


def generation_equivalence(cfg, result, perturbation=None, device="cpu", **_):
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise lab.Blocked("generation_equivalence --device cuda: no CUDA device is available")
    device = torch.empty(0, device=device).device
    result.update(comparisons=[], reference_integrity=[], device=str(device),
                  measurements={"weight_loads": {}, "near_ties": {}, "greedy_steps": {}, "eos_padded_tail_rows": {}, "same_seed_samples_identical": {},
                                "generation": {}})
    if perturbation not in (None, *FAULTS):
        raise lab.SetupError(f"generation_equivalence supports perturbations {FAULTS}, not {perturbation!r}")
    if perturbation:
        result["perturbation"] = {"spec": perturbation}
    tol = {dt: lab.tolerances_for("generation_equivalence", cfg, dt) for dt in DTYPES}
    for shape, attn, dt, V, eos_boost in models():
        set_special_tokens(*((QWEN3_EOS, QWEN3_PAD) if V == QWEN3_VOCAB else (F.EOS_IDS, F.PAD)))
        ref, l2l, store = build_pair(shape, attn, DTYPES[dt], V, device, cfg.execution.weight_storage, eos_boost)
        try:
            if V == QWEN3_VOCAB:
                sets = {"chat": chat_prompts(device)}
            else:
                sets = {name: prompts(lengths, V, device, seed=i) for i, (name, lengths) in enumerate(PROMPTS.items())}
            for name, (ids, mask) in sets.items():
                tag = f"{shape}.{attn}.{dt}.V{V}{'.eos' if eos_boost else ''}.{name}"
                with F.isolated():
                    compare_model(cfg, result, tag, ref, l2l, store, ids, mask, device, perturbation, tol[dt])
            if eos_boost:
                tags = [t for t in result["measurements"]["greedy_steps"] if ".eos." in t]
                if not all(result["measurements"]["eos_padded_tail_rows"][t] > 0 for t in tags):
                    result["errors"].append("an EOS-model prompt set has no row ending in EOS + padding")
            if problems := store.verify():
                result["errors"] += [f"{shape}.{attn}.{dt}: store: {p}" for p in problems[:5]]
        finally:
            store.close()
    result["memory"] = lab.memory(device)


def chat_prompts(device):
    """Non-thinking Qwen3 chat prompts of two sentences, tokenized as policy.generate_rollouts does."""
    from transformers import AutoTokenizer
    from qwen_grpo.model import MODEL
    from qwen_grpo.sentiment import build_prompts
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    except OSError as e:
        raise lab.SetupError(f"the Qwen3 tokenizer is needed for the chat-prompt model: {e}")
    texts = build_prompts(tok, SENTENCES)
    if not all("<think>\n\n</think>" in t for t in texts):
        raise lab.SetupError("build_prompts no longer produces non-thinking Qwen3 prompts")
    batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)
