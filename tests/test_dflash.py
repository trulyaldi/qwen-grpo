"""DFlash on the L2L generator (qwen_grpo/execution/dflash.py): exactness on CPU with tiny models.

The drafter's quality does not affect correctness, only speed, so correctness is tested with (a) a randomly
initialised DFlash drafter (every draft rejected), (b) an oracle that proposes the true greedy continuation with
deliberate errors (partial acceptance, rejection, KV cropping) and (c) an oracle that proposes the target's argmax for
the current prefix (frequent acceptance under sampling). Greedy outputs must equal plain streamed generation exactly;
sampled outputs must follow the target's distribution (chi-square on the joint law of the first two tokens, against the
exact law computed from the model, with a predefined significance level, and the same test applied to the plain
sampler as a control)."""
import pytest
import torch
import transformers
from transformers import Qwen3Config

from qwen_grpo.backward_lab import build
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.dflash import DFlashDrafter, dflash_generate
from qwen_grpo.execution.generator import generate
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.gen_lab import prompts

V = 64


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def target(seed_layers=(0, 5)):
    with F.isolated():
        return build("4b-like", "sdpa", torch.float32, list(seed_layers), vocab=V)


def drafter(block=4):
    c = Qwen3Config(vocab_size=V, hidden_size=40, intermediate_size=96, num_hidden_layers=2, num_attention_heads=8,
                    num_key_value_heads=2, head_dim=8)
    c.dflash_config, c.block_size = {"target_layer_ids": [0, 2, 4], "mask_token_id": V - 1}, block
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return DFlashDrafter(c).eval()


class GreedyOracle(DFlashDrafter):
    """Proposes the known greedy continuation, corrupting one draft token every other round."""

    def __init__(self, base, reference):
        super().__init__(base.config)
        self.load_state_dict(base.state_dict())
        self.reference, self.round = reference, 0

    def propose(self, emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid, committed=None):
        _, cache = super().propose(emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid)
        L, q = committed.shape[1], block.shape[1]
        draft = self.reference[:, L:L + q - 1].clone()
        if draft.shape[1] < q - 1:
            draft = torch.cat([draft, draft.new_full((draft.shape[0], q - 1 - draft.shape[1]), V - 1)], 1)
        self.round += 1
        if self.round % 2 == 0 and q > 2:
            draft[:, 1] = (draft[:, 1] + 1) % (V - 1)  # second draft token wrong: one accepted, then a rejection
        return draft, cache


class ArgmaxOracle(DFlashDrafter):
    """Proposes, position by position, the target's argmax given the committed prefix and the earlier drafts."""

    def __init__(self, base, model, mask):
        super().__init__(base.config)
        self.load_state_dict(base.state_dict())
        self.model, self.prompt_mask = model, mask

    def propose(self, emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid, committed=None):
        _, cache = super().propose(emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid)
        seq = committed.clone()
        for _ in range(block.shape[1] - 1):
            m = torch.cat([self.prompt_mask, self.prompt_mask.new_ones(seq.shape[0], seq.shape[1] - self.prompt_mask.shape[1])], 1)
            pos = (m.cumsum(1) - 1).masked_fill(m == 0, 1)
            z = self.model(input_ids=seq, attention_mask=m, position_ids=pos, use_cache=False).logits[:, -1]
            seq = torch.cat([seq, z.argmax(-1, keepdim=True)], 1)
        return seq[:, committed.shape[1]:], cache


@pytest.mark.parametrize("lengths", [[9, 5, 9, 2], [6, 3]])
def test_greedy_equals_plain_streamed_generation_with_random_and_oracle_drafters(lengths, monkeypatch):
    m = target()
    store = WeightStore.build(target(), "pageable")
    l2l = target()
    l2l.load_state_dict(m.state_dict())
    store = WeightStore.build(l2l, "pageable")
    place_resident(l2l, store, "cpu")
    ex = StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"))
    monkeypatch.setattr(transformers.GenerationMixin, "generate", lambda *a, **k: pytest.fail("model.generate() used"))
    ids, mask = prompts(lengths, V, "cpu", 0)
    with F.isolated():
        plain = generate(m, ids, mask, F.EOS_IDS, F.PAD, 12, 2, do_sample=False)
        rnd = dflash_generate(l2l, drafter(), ids, mask, F.EOS_IDS, F.PAD, 12, 2, do_sample=False, layer_executor=ex)
        oracle = GreedyOracle(drafter(), plain["sequences"])
        orc = dflash_generate(l2l, oracle, ids, mask, F.EOS_IDS, F.PAD, 12, 2, do_sample=False, layer_executor=ex)
    assert torch.equal(rnd["sequences"], plain["sequences"]) and torch.equal(orc["sequences"], plain["sequences"])
    assert max(rnd["accepted_per_pass"]) == 0 and rnd["target_passes"] == rnd["steps"]  # every draft rejected
    acc = orc["accepted_per_pass"]
    assert max(acc) >= 2 and 1 in acc  # full and partial acceptance both happened
    assert orc["target_passes"] < orc["steps"]  # fewer target passes than generated tokens
    assert store.verify() == []


def exact_joint(m, ids, mask):
    """p(t1, t2 | prompt) [V, V] from the model (full-prefix forwards, fp64 softmax)."""
    m1 = (mask.cumsum(1) - 1).masked_fill(mask == 0, 1)
    with torch.no_grad():
        p1 = torch.softmax(m(input_ids=ids, attention_mask=mask, position_ids=m1).logits[0, -1].double(), -1)
        seq = torch.cat([ids.expand(V, -1), torch.arange(V)[:, None]], 1)
        mk = torch.cat([mask.expand(V, -1), torch.ones(V, 1, dtype=mask.dtype)], 1)
        p2 = torch.softmax(m(input_ids=seq, attention_mask=mk, position_ids=(mk.cumsum(1) - 1).masked_fill(mk == 0, 1)
                             ).logits[:, -1].double(), -1)
    return p1[:, None] * p2


def chi_square_p(counts, probs, min_expected=5.0):
    """Pearson chi-square p-value, cells with expected count < min_expected pooled into one."""
    n = counts.sum()
    exp = probs * n
    big = exp >= min_expected
    obs_c = torch.cat([counts[big], counts[~big].sum()[None]])
    exp_c = torch.cat([exp[big], exp[~big].sum()[None]])
    keep = exp_c > 0
    stat = (((obs_c - exp_c) ** 2) / exp_c)[keep].sum().item()
    df = int(keep.sum()) - 1  # p = Q(df / 2, stat / 2), the chi-square survival function
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64),
                                         torch.tensor(stat / 2, dtype=torch.float64))), int(keep.sum())


def test_sampling_follows_the_target_distribution_with_frequent_acceptance():
    """2000 single-row DFlash continuations of one prompt (single rows, so per-row acceptance is exercised: with a
    batch, rows advance by the minimum acceptance), 3 new tokens; the joint law of the first two tokens is compared
    with the model's exact law. Predefined: reject at p < 1e-3, for DFlash and, as a control, the plain sampler."""
    m = target()
    ids, mask = prompts([5], V, "cpu", 3)
    p = exact_joint(m, ids, mask).view(V, V)
    q = p.clone()  # a row ending on EOS at t1 gets the pad id at t2
    for e in F.EOS_IDS:
        q[e, F.PAD] = p[e].sum()
        q[e, [j for j in range(V) if j != F.PAD]] = 0
    n, accepted, rows = 2000, 0, []
    oracle = ArgmaxOracle(drafter(block=3), m, mask)
    with F.isolated():
        g = torch.Generator().manual_seed(11)
        for _ in range(n):
            d = dflash_generate(m, oracle, ids, mask, F.EOS_IDS, F.PAD, 3, 1, do_sample=True, generator=g)
            t = d["sequences"][0, ids.shape[1]:ids.shape[1] + 2]
            rows.append(torch.cat([t, t.new_full((2 - t.shape[0],), F.PAD)]))  # stopped on EOS at t1: pad, as plain
            accepted += sum(d["accepted_per_pass"])
        torch.manual_seed(12)
        plain = generate(m, ids, mask, F.EOS_IDS, F.PAD, 2, n, do_sample=True)["sequences"][:, ids.shape[1]:]
    for name, t in (("dflash", torch.stack(rows)), ("plain", plain)):
        counts = torch.bincount(t[:, 0] * V + t[:, 1], minlength=V * V).double()
        pval, cells = chi_square_p(counts, q.flatten())
        assert pval > 1e-3, (name, pval, cells)
    assert accepted > 50  # drafts were accepted under sampling: the acceptance / correction path ran
