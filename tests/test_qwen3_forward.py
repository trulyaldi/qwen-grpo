"""Explicit Qwen3 forward (qwen_grpo/execution/qwen3_forward.py) against Hugging Face, on tiny random Qwen3 models.

Two kinds of comparison, two tolerances:
- same input tensors, explicit vs HF: the same modules run in the same order on the same kernels, so the result must be
  bitwise equal (tolerance 0); any difference means an argument (mask, positions, cache) differs.
- a padded row vs the same tokens unpadded: different shapes take different kernel paths, so only fp32 agreement up to
  rounding is expected. Measured max abs error on these models is <= 3e-7; PAD_TOL = 1e-5 leaves headroom while still
  catching a wrong mask or position (those change logits by O(0.1)).
Relative error is max|a - b| / max|b| (elementwise ratios are meaningless for logits near 0).

`python -m tests.test_qwen3_forward` measures the same quantities on the real Qwen3-0.6B (GPU, fp32 and bf16); pytest
never loads pretrained weights (tests/conftest.py).
"""
import pytest
import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo.execution.qwen3_forward import explicit_forward, explicit_logprobs
from qwen_grpo.model import PRESETS, lora_config, lora_modules, target_modules
from qwen_grpo.policy import compute_logprobs

PAD_TOL = 1e-5
VOCAB = 64
ATTN = ["eager", "sdpa"]


def tiny(attn, dtype=torch.float32, lora=False):
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                         tie_word_embeddings=True, attn_implementation=attn)).to(dtype).eval()
    if lora:
        model = get_peft_model(model, lora_config(target_modules(model, *PRESETS["last2-attention"]))).eval()
        for _, m in lora_modules(model):  # a trained-looking adapter: B != 0, so LoRA changes the output
            torch.nn.init.normal_(m.lora_B["default"].weight, std=0.5)
    return model


def errors(a, b):
    """(max abs error, max abs error / max |reference|)."""
    a, b = a.double(), b.double()
    abs_err = (a - b).abs().max().item()
    return abs_err, abs_err / b.abs().max().item()


def left_padded(lengths, T, seed=1):
    """Random ids [N, T] left-padded to T with mask [N, T] and generate()'s positions cumsum(mask) - 1."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (len(lengths), T), generator=g)
    mask = torch.stack([torch.arange(T) >= T - n for n in lengths]).long()
    ids = ids.masked_fill(mask == 0, 0)
    return ids, mask, (mask.cumsum(1) - 1).clamp(min=0)


def hf_logits(model, ids, mask=None, position_ids=None, logits_to_keep=0):
    return model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False,
                 logits_to_keep=logits_to_keep).logits


# --- 1. explicit forward vs Hugging Face logits ----------------------------------------------------------------------

@pytest.mark.parametrize("attn", ATTN)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lora", [False, True])
@torch.no_grad()
def test_explicit_forward_equals_hf_logits(attn, dtype, lora):
    model = tiny(attn, dtype, lora)
    ids, mask, pos = left_padded([9, 5, 9, 2], T=9)
    full = torch.ones_like(mask)
    cases = [  # (mask, position_ids, logits_to_keep): unpadded, padded, and the column slices HF supports
        (None, None, 0), (full, None, 0), (mask, pos, 0), (mask, pos, 4), (mask, pos, torch.tensor([0, 3, 8]))]
    for m, p, keep in cases:
        explicit = explicit_forward(model, ids, m, p, keep)
        reference = hf_logits(model, ids, m, p, keep)
        assert explicit.dtype == reference.dtype == dtype
        assert torch.equal(explicit, reference), (m is None, keep, errors(explicit, reference))

    if lora:  # the adapters really ran inside the explicit layer loop
        with model.disable_adapter():
            base = explicit_forward(model, ids, mask, pos)
        assert errors(explicit_forward(model, ids, mask, pos), base)[0] > 1e-2


# --- 2. padded and unpadded sequences --------------------------------------------------------------------------------

@pytest.mark.parametrize("attn", ATTN)
@pytest.mark.parametrize("lora", [False, True])
@torch.no_grad()
def test_padded_rows_match_unpadded_sequences(attn, lora):
    """Each row of a padded batch, at its real-token columns, equals the same tokens run alone without padding (fp32).
    Left padding uses positions cumsum(mask) - 1 (as generate() does); right padding works with the default 0..T-1."""
    model = tiny(attn, lora=lora)
    lengths, T = [9, 5, 1, 7], 9
    ids, mask, pos = left_padded(lengths, T)
    right_ids = torch.stack([torch.cat([r[T - n:], r[: T - n]]) for r, n in zip(ids, lengths)])
    right_mask = torch.stack([torch.arange(T) < n for n in lengths]).long()

    left = explicit_forward(model, ids, mask, pos)
    right = explicit_forward(model, right_ids, right_mask)
    assert torch.isfinite(left[mask.bool()]).all() and torch.isfinite(right[right_mask.bool()]).all()
    for i, n in enumerate(lengths):
        alone = explicit_forward(model, ids[i : i + 1, T - n :])[0]                   # [n, V], no padding at all
        for padded in (left[i, T - n :], right[i, :n]):
            abs_err, _ = errors(padded, alone)
            assert abs_err < PAD_TOL, (i, abs_err)
    # the tolerance must be able to fail: without mask and positions the pads leak in, by far more than PAD_TOL. (No mask
    # but cumsum positions is NOT such a control: HF reads restarts of position_ids at 0 as packed sequences and isolates
    # each pad, exactly as the explicit path does.)
    unmasked = explicit_forward(model, ids)
    assert errors(unmasked[1, T - 5 :], left[1, T - 5 :])[0] > 1e3 * PAD_TOL


# --- 3. selected-token log-probabilities -----------------------------------------------------------------------------

@pytest.mark.parametrize("attn", ATTN)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lora", [False, True])
@torch.no_grad()
def test_selected_token_logprobs_equal_compute_logprobs(attn, dtype, lora):
    """explicit_logprobs == policy.compute_logprobs (the GRPO scorer) bitwise, on a hand-built rollout batch: left-padded
    prompts of P columns, responses of R columns with a post-EOS padded tail. Also checked (fp32) against log_softmax of
    the full HF logits gathered at the selected tokens: lm_head then runs on all T columns instead of R + 1, a different
    GEMM shape, so that agreement is up to rounding (PAD_TOL), not bitwise."""
    model = tiny(attn, dtype, lora)
    P, R = 6, 4
    prompt_ids, prompt_mask, _ = left_padded([6, 3, 4], T=P)
    response_ids = torch.randint(0, VOCAB, (3, R), generator=torch.Generator().manual_seed(2))
    completion_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])
    rollouts = {"input_ids": torch.cat([prompt_ids, response_ids], 1),
                "attention_mask": torch.cat([prompt_mask, completion_mask], 1), "prompt_len": P}

    explicit = explicit_logprobs(model, rollouts)                                      # [N, R]
    reference = compute_logprobs(model, rollouts)
    assert explicit.shape == (3, R) and explicit.dtype == torch.float32
    assert torch.equal(explicit, reference), errors(explicit, reference)

    ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
    if dtype != torch.float32:
        return
    full = hf_logits(model, ids, mask, (mask.cumsum(1) - 1).clamp(min=0)).float()      # [N, T, V]
    gathered = full[:, P - 1 : -1].log_softmax(-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    assert errors(explicit, gathered)[0] < PAD_TOL, errors(explicit, gathered)


# --- real Qwen3-0.6B measurement (not collected by pytest) -----------------------------------------------------------

def measure_real(name="Qwen/Qwen3-0.6B", device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    texts = ["The movie was a delight from start to finish.", "Dull.", "I expected more, but the ending saved it.",
             "Neither the acting nor the script could rescue this tired, overlong sequel."]
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
               for t in texts]
    batch = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    lengths = mask.sum(1).tolist()
    T = ids.shape[1]
    R = 4
    rollouts = {"input_ids": ids, "attention_mask": mask, "prompt_len": T - R}  # last R prompt tokens as "response"

    print(f"{name}: N={len(texts)} T={T} row lengths={lengths}")
    for dtype, attn in [(d, a) for d in (torch.float32, torch.bfloat16) for a in ATTN]:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn).to(device).eval()
        with torch.no_grad():
            explicit, reference = explicit_forward(model, ids, mask, pos), hf_logits(model, ids, mask, pos)
            a, r = errors(explicit[mask.bool()], reference[mask.bool()])
            lp_a, lp_r = errors(explicit_logprobs(model, rollouts), compute_logprobs(model, rollouts))
            pad_a, pad_r = max(errors(explicit[i, T - n :], explicit_forward(model, ids[i : i + 1, T - n :])[0])
                               for i, n in enumerate(lengths))
        print(f"  {str(dtype):15s} {attn:5s}: explicit-vs-HF logits abs {a:.3g} rel {r:.3g} bitwise "
              f"{torch.equal(explicit, reference)} | logprobs abs {lp_a:.3g} rel {lp_r:.3g} | "
              f"padded-vs-unpadded logits abs {pad_a:.3g} rel {pad_r:.3g}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    measure_real()
