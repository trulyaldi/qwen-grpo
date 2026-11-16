import torch

from qwen_grpo.grpo import response_logprobs, response_mask


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompts, G=4, max_new_tokens=8):
    """B chat-formatted prompts -> B*G sampled rollouts; rows [b*G, (b+1)*G) belong to prompt b.

    Pure sampling from the policy (temperature 1, no top-k/top-p): compute_logprobs scores untempered logits, so any
    other setting would make the sampler differ from the policy being optimized. Passed as kwargs so they override
    Qwen's shipped generation defaults.
    """
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    P = batch["input_ids"].shape[1]

    input_ids = model.generate(
        **batch,
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        num_return_sequences=G, max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
    )
    response_ids = input_ids[:, P:]                                   # [B*G, R]
    completion_mask = response_mask(response_ids, eos_ids)            # pad id may be an EOS id
    prompt_mask = batch["attention_mask"].repeat_interleave(G, dim=0)  # generate() expands this way
    return {
        "input_ids": input_ids,
        "attention_mask": torch.cat([prompt_mask, completion_mask.long()], dim=1),
        "response_ids": response_ids,
        "completion_mask": completion_mask,
        "prompt_len": P,
    }


def compute_logprobs(model, rollouts):
    """Teacher-forced log pi(o_t | q, o_<t) for every response column -> [N, R]. Keeps autograd.

    Only the R response tokens are scored, so only the positions that predict them go through lm_head: the logit at
    column t-1 scores the token at column t, i.e. columns P-1 .. T-2. logits_to_keep=R+1 keeps the last R+1 columns,
    T-R-1 = P-1 .. T-1, and response_logprobs drops the final one (it predicts past the sequence). The decoder still sees
    the whole sequence; only the [N, T, V] logits (V ~ 152K) of the prompt positions are never materialized.
    """
    ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
    R = ids.shape[1] - rollouts["prompt_len"]
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)  # as generate() does for left-padded rows
    logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False,
                   logits_to_keep=R + 1).logits                                              # [N, R+1, V]
    assert logits.shape[1] == R + 1, f"logits_to_keep ignored: {tuple(logits.shape)}"  # else the memory saving is lost
    return response_logprobs(logits, ids, rollouts["prompt_len"])
