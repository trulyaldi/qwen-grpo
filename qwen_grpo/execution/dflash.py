"""DFlash block-diffusion speculative decoding on the L2L target engine (Ticket 12).

Source read before implementing: z-lab/Qwen3-4B-DFlash-b16 (config.json, dflash.py, utils.py; paper arXiv 2602.06036).
The drafter is re-implemented here from that code (only its weights are loaded; no remote code is executed):
  features   the target's hidden states after decoder layers target_layer_ids (= [1, 9, 17, 25, 33] for Qwen3-4B),
             concatenated [.., k * H] -> fc -> RMSNorm: the drafter's context
  drafter    num_hidden_layers Qwen3-style layers whose attention is non-causal, with keys / values = [context
             positions not yet seen by the drafter, the noise block] (q only for the block); the drafter cache keeps
             the context part only (cropped back after every round)
  block      [last committed token, mask_token_id x (block_size - 1)], embedded with the TARGET's embedding; draft
             tokens = argmax of the TARGET's LM head on the drafter's last block_size - 1 outputs
  verify     the target (current LoRA policy, through the layer executor) runs the whole block in one pass from its
             KV cache, giving logits and next-round features for every block position
Acceptance (official rule, generalized to a batch): per row, the target draws its own token y_j at every block
position j (argmax for greedy, multinomial(softmax(logits)) for sampling: temperature 1, no truncation, as the
plain generator); the row accepts the longest draft prefix with draft_j == y_j and then emits y at the first
mismatch. Rows advance together by m = min over unfinished rows of their accepted length (finished rows ignored), so
the KV caches stay aligned; a row that matched further is simply cut at m and emits y_m, which equals its draft.
Why this is exact: every emitted token is the target's own draw at that position given the emitted prefix (accepted
drafts equal the target's draws), so each row's sequence has the target policy's distribution. For an argmax
(point-mass) drafter this is the same accept / resample law as standard speculative sampling (accept with
probability p(draft), else resample from p restricted to tokens != draft).
KV state after a round: the target cache is cropped to the committed length (the last emitted token is the next
block's first input, as in plain decoding); the drafter cache to the context it has consumed.
Residency: the drafter (about 1.08 GB for Qwen3-4B in bf16) and its context are GPU-resident; the target streams
through the executor exactly as in plain generation.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half

from qwen_grpo.execution.generator import KVCache, _positions, check_model
from qwen_grpo.execution.qwen3_forward import ResidentLayerExecutor, unwrap
from transformers.masking_utils import create_causal_mask


class DraftAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hd, self.nh, self.nkv = c.head_dim, c.num_attention_heads, c.num_key_value_heads
        self.q_proj = nn.Linear(c.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.hd, eps=c.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.hd, eps=c.rms_norm_eps)

    def forward(self, x, ctx, cos, sin, cache, mask):
        """x [N, q, H] noise block, ctx [N, c, H] new context; cache: (k, v) of earlier context or None.
        cos / sin cover the new context and the block ([N, c + q, hd]); mask [N, 1, q, kv] additive or None."""
        N, q, _ = x.shape
        c = ctx.shape[1]
        qh = self.q_norm(self.q_proj(x).view(N, q, self.nh, self.hd)).transpose(1, 2)
        k = self.k_norm(torch.cat([self.k_proj(ctx), self.k_proj(x)], 1).view(N, c + q, self.nkv, self.hd)).transpose(1, 2)
        v = torch.cat([self.v_proj(ctx), self.v_proj(x)], 1).view(N, c + q, self.nkv, self.hd).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        qh = qh * cos[..., -q:, :] + rotate_half(qh) * sin[..., -q:, :]
        k = k * cos + rotate_half(k) * sin
        new_ctx = (k[:, :, :c], v[:, :, :c])
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        out = torch.nn.functional.scaled_dot_product_attention(qh, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(N, q, -1)), new_ctx


class DraftLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = DraftAttention(c)
        self.mlp = Qwen3MLP(c)
        self.input_layernorm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def forward(self, x, ctx, cos, sin, cache, mask):
        a, new_ctx = self.self_attn(self.input_layernorm(x), ctx, cos, sin, cache, mask)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), new_ctx


class DFlashDrafter(nn.Module):
    """The DFlash draft model (keys as in the checkpoint: layers.*, fc, hidden_norm, norm)."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.target_layer_ids = list(c.dflash_config["target_layer_ids"])
        self.mask_token_id = c.dflash_config["mask_token_id"]
        self.block_size = c.block_size
        self.layers = nn.ModuleList(DraftLayer(c) for _ in range(c.num_hidden_layers))
        self.fc = nn.Linear(len(self.target_layer_ids) * c.hidden_size, c.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.norm = Qwen3RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(c)

    @classmethod
    def from_pretrained(cls, name_or_dir, dtype=torch.bfloat16, device="cpu"):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import Qwen3Config
        d = Path(name_or_dir) if Path(name_or_dir).is_dir() else Path(snapshot_download(
            str(name_or_dir), allow_patterns=["config.json", "model.safetensors"]))
        raw = json.loads((d / "config.json").read_text())
        c = Qwen3Config(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map")})
        c.dflash_config, c.block_size = raw["dflash_config"], raw["block_size"]
        m = cls(c)
        state = load_file(str(d / "model.safetensors"))
        missing, unexpected = m.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("rotary_emb")]
        if missing or unexpected:
            raise RuntimeError(f"drafter checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return m.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def propose(self, emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid, committed=None):
        """Draft tokens for block[:, 1:] (argmax of the target's LM head on the drafter's outputs) and the new cache."""
        hidden, cache = self(emb(block), ctx_feat, ctx_pos, block_pos, cache, ctx_valid)
        if block.shape[1] == 1:
            return block[:, 1:], cache
        return head(hidden[:, 1:]).float().argmax(-1), cache

    def nbytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def forward(self, block_emb, ctx_feat, ctx_pos, block_pos, cache, ctx_valid):
        """block_emb [N, q, H]; ctx_feat [N, c, k*H] new context features; ctx_pos [N, c], block_pos [N, q];
        cache: per-layer (k, v) of earlier context or None; ctx_valid [N, c_total] bool (padding masked).
        -> (hidden [N, q, H], new per-layer cache)."""
        ctx = self.hidden_norm(self.fc(ctx_feat))
        pos = torch.cat([ctx_pos, block_pos], 1)
        cos, sin = self.rotary_emb(block_emb, pos)
        q = block_emb.shape[1]
        valid = torch.cat([ctx_valid, ctx_valid.new_ones(ctx_valid.shape[0], q)], 1)          # [N, kv]
        mask = torch.zeros(valid.shape, dtype=block_emb.dtype, device=block_emb.device)
        mask = mask.masked_fill(~valid, torch.finfo(block_emb.dtype).min)[:, None, None, :].expand(-1, 1, q, -1)
        x, new = block_emb, []
        for i, layer in enumerate(self.layers):
            prev = None if cache is None else cache[i]
            x, nc = layer(x, ctx, cos, sin, prev, mask)
            new.append(nc if prev is None else (torch.cat([prev[0], nc[0]], 2), torch.cat([prev[1], nc[1]], 2)))
        return self.norm(x), new


def crop(cache, n):
    """Crop the target's KVCache to its first n positions (every layer)."""
    cache.lengths = [min(length, n) for length in cache.lengths]


@torch.no_grad()
def _target_step(model, ids, mask, position_ids, cache_position, cache, executor, feature_layers):
    """One target forward over the new columns `ids` [N, q] -> (fp32 logits [N, q, V], features [N, q, k*H])."""
    decoder, lm_head = unwrap(model)
    hidden = decoder.embed_tokens(ids)
    attn = create_causal_mask(config=decoder.config, input_embeds=hidden, attention_mask=mask,
                              cache_position=cache_position, past_key_values=cache, position_ids=position_ids)
    pe = decoder.rotary_emb(hidden, position_ids)
    feats = []
    for index, layer in enumerate(decoder.layers[: decoder.config.num_hidden_layers]):
        hidden = executor.execute(index, layer, hidden, attention_mask=attn, position_ids=position_ids,
                                  past_key_values=cache, use_cache=True, cache_position=cache_position,
                                  position_embeddings=pe)
        if index in feature_layers:
            feats.append(hidden)
    logits = lm_head(decoder.norm(hidden)).float()
    return logits, torch.cat([feats[feature_layers.index(i)] for i in feature_layers], -1)


def _draw(logits, do_sample, generator):
    if not do_sample:
        return logits.argmax(-1)
    N, q, V = logits.shape
    return torch.multinomial(torch.softmax(logits.reshape(N * q, V), -1), 1, generator=generator).view(N, q)


@torch.no_grad()
def dflash_generate(model, drafter, input_ids, attention_mask, eos_token_ids, pad_token_id, max_new_tokens,
                    num_return_sequences=1, do_sample=True, layer_executor=None, kv_device=None, generator=None):
    """Like execution.generator.generate (same prompt expansion, positions, masks, EOS / padding and stopping), with
    DFlash drafting and block verification. -> {"sequences", "steps", "target_passes", "accepted_per_pass"}."""
    check_model(model)
    executor = ResidentLayerExecutor() if layer_executor is None else layer_executor
    config = unwrap(model)[0].config
    feature_layers = drafter.target_layer_ids
    if max(feature_layers) >= config.num_hidden_layers:
        raise ValueError(f"drafter features from layers {feature_layers}; the target has {config.num_hidden_layers}")
    emb = unwrap(model)[0].embed_tokens
    head = unwrap(model)[1]
    device = input_ids.device
    eos = torch.tensor(list(eos_token_ids), device=device)
    ids = input_ids.repeat_interleave(num_return_sequences, 0)
    mask = attention_mask.repeat_interleave(num_return_sequences, 0)
    N, P = ids.shape
    bs = drafter.block_size
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    cache = KVCache(config.num_hidden_layers, N, config.num_key_value_heads, P + max_new_tokens + bs, head_dim,
                    emb.weight.dtype, kv_device or device, device)
    # prefill: the first token is the target's own draw
    pos = _positions(mask)
    logits, feats = _target_step(model, ids, mask, pos, torch.arange(P, device=device), cache, executor, feature_layers)
    target_passes, accepted = 1, []
    nxt = _draw(logits[:, -1:], do_sample, generator)[:, 0]
    unfinished = torch.ones(N, dtype=torch.long, device=device)
    nxt = nxt * unfinished + pad_token_id * (1 - unfinished)
    ids = torch.cat([ids, nxt[:, None]], 1)
    mask = torch.cat([mask, mask.new_ones(N, 1)], 1)
    unfinished = unfinished & ~torch.isin(nxt, eos)
    d_cache, ctx_feat, ctx_pos, ctx_valid = None, feats, pos, attention_mask.repeat_interleave(num_return_sequences, 0).bool()
    while ids.shape[1] - P < max_new_tokens and unfinished.max() > 0:
        L = ids.shape[1]                                            # committed length; ids[:, -1] not yet in the cache
        q = min(bs, P + max_new_tokens - L + 1)                     # the block may not run past max_new_tokens
        block = torch.full((N, q), drafter.mask_token_id, dtype=torch.long, device=device)
        block[:, 0] = ids[:, -1]
        full_mask = torch.cat([mask, mask.new_ones(N, q - 1)], 1)
        block_pos = _positions(full_mask)[:, L - 1:L - 1 + q]
        draft, d_cache = drafter.propose(emb, head, block, ctx_feat, ctx_pos, block_pos, d_cache, ctx_valid,
                                         committed=ids)
        block[:, 1:] = draft
        logits, feats = _target_step(model, block, full_mask, block_pos, torch.arange(L - 1, L - 1 + q, device=device),
                                     cache, executor, feature_layers)
        target_passes += 1
        y = _draw(logits, do_sample, generator)                     # [N, q]: the target's own draws
        match = (block[:, 1:] == y[:, :-1]).long().cumprod(1).sum(1)   # accepted drafts per row
        live = unfinished.bool()
        m = int(match[live].min())
        emitted = torch.cat([block[:, 1:1 + m], y[:, m:m + 1]], 1)     # [N, m + 1]
        for j in range(emitted.shape[1]):                           # EOS / padding exactly as the plain generator
            if ids.shape[1] - P >= max_new_tokens:
                break
            tok = emitted[:, j] * unfinished + pad_token_id * (1 - unfinished)
            ids = torch.cat([ids, tok[:, None]], 1)
            mask = torch.cat([mask, mask.new_ones(N, 1)], 1)
            unfinished = unfinished & ~torch.isin(tok, eos)
        accepted.append(m)
        crop(cache, ids.shape[1] - 1)                               # keep the KV of every committed token but the last
        used = ids.shape[1] - L                                     # tokens committed this round (<= m + 1)
        # next round's new drafter context: target features of this round's block inputs whose tokens are committed
        # and in the target cache (block[0] and the accepted drafts); the drafter cache already holds only context
        ctx_feat, ctx_pos = feats[:, :used], block_pos[:, :used]
        ctx_valid = torch.cat([ctx_valid, ctx_valid.new_ones(N, used)], 1)
    steps = ids.shape[1] - P
    return {"sequences": ids, "steps": steps, "target_passes": target_passes, "accepted_per_pass": accepted,
            "drafter_bytes": drafter.nbytes()}


class DFlashGenerator:
    """Rollout generator for policy.generate_rollouts with DFlash drafting; the target (current LoRA policy) runs
    through `layer_executor` and verifies every token. Same call signature as generator.StreamedGenerator."""

    def __init__(self, drafter, layer_executor=None, kv_device=None, do_sample=True):
        self.drafter, self.layer_executor, self.kv_device, self.do_sample = drafter, layer_executor, kv_device, do_sample
        self.last, self.history = None, []

    def __call__(self, model, input_ids, attention_mask, num_return_sequences, max_new_tokens, eos_token_ids,
                 pad_token_id):
        self.last = dflash_generate(model, self.drafter, input_ids, attention_mask, eos_token_ids, pad_token_id,
                                    max_new_tokens, num_return_sequences, do_sample=self.do_sample,
                                    layer_executor=self.layer_executor, kv_device=self.kv_device)
        self.history.append({k: self.last[k] for k in ("steps", "target_passes", "accepted_per_pass")})
        return self.last["sequences"]
