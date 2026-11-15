import torch

from qwen_grpo.grpo import response_logprobs, response_mask


@torch.no_grad()
def generate_rollouts(model, tokenizer, prompts, G=4, max_new_tokens=8, temperature=1.0):
    """B chat-formatted prompts -> B*G sampled rollouts; rows [b*G, (b+1)*G) belong to prompt b.

    Sampling settings are passed as kwargs so they override Qwen's shipped generation defaults.
    """
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    P = batch["input_ids"].shape[1]

    input_ids = model.generate(
        **batch,
        do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
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
    """Teacher-forced log pi(o_t | q, o_<t) for every response column -> [B*G, R]. Keeps autograd."""
    mask = rollouts["attention_mask"]
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)  # as generate() does for left-padded rows
    logits = model(
        input_ids=rollouts["input_ids"], attention_mask=mask, position_ids=position_ids, use_cache=False
    ).logits
    return response_logprobs(logits, rollouts["input_ids"], rollouts["prompt_len"])
