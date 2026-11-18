"""DFlash inside the GRPO training path, on CPU with tiny models (no Qwen3-4B, no GPU).

What this file establishes, and what it does not:
  - the final run's configuration resolves to DFlash rollouts: make_backend puts a DFlashGenerator (frozen drafter,
    the backend's own L2L layer executor) in backend.sampler, so policy.generate_rollouts - the trainer's only rollout
    call - goes through dflash_generate; evaluation (backend.greedy) and make_evaluation_backend stay plain;
  - the verifier is the CURRENT LoRA policy: drafts taken from the base model are corrected to the LoRA policy's own
    greedy tokens, also after the adapter changes;
  - greedy DFlash through the L2L executor equals plain decoding with left padding, EOS mid-block (with the pad id
    being an EOS id, as in Qwen3) and KV cropping after partial acceptance;
  - GRPO's old log-probs of DFlash-sampled tokens are the sampling policy's log-probs (teacher-forced backend.logprobs
    equals the plain decoder's per-step distribution along the same tokens), the first update sees ratio 1, the
    drafter's weights never change and are not optimized, and an LR=0 update leaves the adapter unchanged;
  - the trainer records which generator produced each rollout batch (trainer.rollout_evidence).
The drafters are a randomly initialised DFlash module or oracles: correctness must not depend on draft quality. The
real drafter (z-lab/Qwen3-4B-DFlash-b16) against real Qwen3-4B is checked only on a GPU (H10.dflash_bench greedy
identity, H10.smoke_dflash, H12.train_final); nothing here substitutes for that. The exact sampling law is tested in
tests/test_dflash.py."""
import dataclasses
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import BatchEncoding, GenerationConfig

from qwen_grpo import trainer
from qwen_grpo.execution import backend as backend_mod
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution import generator as generator_mod
from qwen_grpo.execution.dflash import DFlashDrafter, DFlashGenerator, dflash_generate
from qwen_grpo.execution.generator import StreamedGenerator, generate
from qwen_grpo.gen_lab import prompts
from qwen_grpo.model import lora_modules
from qwen_grpo.training import INIT_TOL, grpo_update, prepare
from tests.test_dflash import V, GreedyOracle, drafter, target

DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def cfg(enabled=True):
    """What make_backend reads (cfg.execution, cfg.dflash): the final run's execution with a CPU pageable store and
    synchronous streaming (the async prefetcher needs CUDA streams; its equivalence is tested on CUDA)."""
    from qwen_grpo.config import DFlash
    return Cfg(execution=SimpleNamespace(backend="l2l_sync", weight_storage="pageable", weight_buffers=2,
                                         kv_cache="gpu", prefetch_depth=0),
               dflash=DFlash(enabled=enabled, draft_model=DRAFT_ID if enabled else None))


Cfg = dataclasses.make_dataclass("Cfg", ["execution", "dflash"])  # make_evaluation_backend uses dataclasses.replace


def policy_pair(sharpen=1.0):
    """Two identical LoRA policies (LoRA on layers 0 and 5, nonzero B): one becomes the L2L backend's model, the other
    stays a plain resident reference. sharpen > 1 scales the tied embedding / LM head, so that the tiny random model's
    next-token distribution is peaked like a trained LM's and sampled tokens often equal the drafts (under a flat
    distribution the batched DFlash rule would reject nearly every block and leave the acceptance path untested)."""
    m = target()
    if sharpen != 1.0:
        with torch.no_grad():
            m.get_input_embeddings().weight.mul_(sharpen)
    ref = target()
    ref.load_state_dict(m.state_dict())
    for x in (m, ref):
        x.base_model.model.generation_config = GenerationConfig(eos_token_id=F.EOS_IDS, pad_token_id=F.PAD)
    return m, ref


def set_adapter_scale(model, original, factor):
    """LoRA B := factor * its original value (0: the policy equals the base model)."""
    with torch.no_grad():
        for (_, mod), b in zip(lora_modules(model), original):
            mod.lora_B["default"].weight.copy_(b * factor)


def lora_b(model):
    return [mod.lora_B["default"].weight.detach().clone() for _, mod in lora_modules(model)]


def l2l_backend(monkeypatch, model, draft=None, enabled=True):
    loaded = []

    def fake_from_pretrained(name, dtype=torch.bfloat16, device="cpu"):  # stands in for the Hub download only
        loaded.append(name)
        return (draft or drafter()).requires_grad_(False).eval()

    monkeypatch.setattr(DFlashDrafter, "from_pretrained", staticmethod(fake_from_pretrained))
    be = backend_mod.make_backend(cfg(enabled), model, "cpu")
    return be, loaded


class BaseOracle(DFlashDrafter):
    """Proposes the BASE model's greedy continuation (the drafter's training target), whatever the adapter is: the
    LoRA verifier must reject what the adapter changed. `model` is a resident copy of the policy (an L2L model's frozen
    weights live in its store, so it cannot be called directly)."""

    def __init__(self, base, model, prompt_mask):
        super().__init__(base.config)
        self.load_state_dict(base.state_dict())
        self.model, self.prompt_mask = model, prompt_mask
        self.requires_grad_(False).eval()

    def propose(self, emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid, committed=None):
        _, cache = super().propose(emb, head, block, ctx_feat, ctx_pos, block_pos, cache, ctx_valid)
        seq = committed.clone()
        with self.model.disable_adapter(), torch.no_grad():
            for _ in range(block.shape[1] - 1):
                pm = self.prompt_mask.repeat_interleave(seq.shape[0] // self.prompt_mask.shape[0], 0)
                m = torch.cat([pm, pm.new_ones(seq.shape[0], seq.shape[1] - pm.shape[1])], 1)
                pos = (m.cumsum(1) - 1).masked_fill(m == 0, 1)
                z = self.model(input_ids=seq, attention_mask=m, position_ids=pos, use_cache=False).logits[:, -1]
                seq = torch.cat([seq, z.argmax(-1, keepdim=True)], 1)
        return seq[:, committed.shape[1]:], cache


class FakeTokenizer:
    """policy.generate_rollouts' tokenizer interface for pre-tokenized prompts."""
    pad_token_id = F.PAD

    def __init__(self, ids, mask):
        self.ids, self.mask = ids, mask

    def __call__(self, prompts_, **kwargs):
        return BatchEncoding({"input_ids": self.ids, "attention_mask": self.mask})


# ------------------------------------------------------------------------------------------------ backend wiring

def test_the_final_configuration_resolves_to_a_dflash_rollout_generator_with_a_frozen_drafter(monkeypatch):
    m, _ = policy_pair()
    be, loaded = l2l_backend(monkeypatch, m)
    assert loaded == [DRAFT_ID] and be.name == "l2l_sync+dflash" and be.draft_model == DRAFT_ID
    assert isinstance(be.sampler, DFlashGenerator) and be.sampler.do_sample
    assert be.sampler.layer_executor is be.executor  # the target verifies through the L2L layer executor
    assert be.sampler.drafter is be.drafter and not be.drafter.training
    assert not any(p.requires_grad for p in be.drafter.parameters())
    model_params = {id(p) for p in m.parameters()}
    assert not any(id(p) in model_params for p in be.drafter.parameters())  # never in the optimizer or a checkpoint
    assert isinstance(be.greedy, StreamedGenerator) and not isinstance(be.greedy, DFlashGenerator)  # evaluation plain
    be.close()


def test_plain_l2l_and_evaluation_backends_never_load_the_drafter(monkeypatch):
    """The plain-L2L baseline (dflash off), and evaluation of a DFlash run's checkpoints (final_eval.py,
    make_evaluation_backend with dflash on in the run's config), both decode without the drafter."""
    monkeypatch.setattr(DFlashDrafter, "from_pretrained", staticmethod(lambda *a, **k: pytest.fail("drafter loaded")))
    for make in (lambda mm: backend_mod.make_backend(cfg(False), mm, "cpu"),
                 lambda mm: backend_mod.make_evaluation_backend(cfg(True), mm, "cpu")):
        m, _ = policy_pair()
        be = make(m)
        assert type(be.sampler) is StreamedGenerator and "dflash" not in be.name and not hasattr(be, "drafter")
        be.close()


def test_a_trainable_drafter_is_refused(monkeypatch):
    m, _ = policy_pair()
    monkeypatch.setattr(DFlashDrafter, "from_pretrained", staticmethod(lambda *a, **k: drafter().requires_grad_(True)))
    with pytest.raises(RuntimeError, match="frozen"):
        backend_mod.make_backend(cfg(True), m, "cpu")


# ----------------------------------------------------------------------------------- the verifier is the policy

def test_the_verifier_is_the_current_lora_policy_also_after_the_adapter_changes(monkeypatch):
    """Drafts come from the base model; DFlash's greedy output must equal the LoRA policy's plain greedy output (and
    not the base's) as the adapter changes in place (B = 0: policy = base, every draft accepted; then larger B), i.e.
    the verifier reads the live LoRA tensors through the L2L executor."""
    m, ref = policy_pair()
    original = lora_b(m)
    ids, mask = prompts([7, 4, 7], V, "cpu", 5)
    be, _ = l2l_backend(monkeypatch, m)
    oracle = BaseOracle(drafter(), ref, mask)
    for factor in (0.0, 1.0, 40.0):
        set_adapter_scale(m, original, factor)
        set_adapter_scale(ref, original, factor)
        with F.isolated():
            plain = generate(ref, ids, mask, F.EOS_IDS, F.PAD, 10, 2, do_sample=False)["sequences"]
            with ref.disable_adapter():
                base = generate(ref, ids, mask, F.EOS_IDS, F.PAD, 10, 2, do_sample=False)["sequences"]
            out = dflash_generate(m, oracle, ids, mask, F.EOS_IDS, F.PAD, 10, 2, do_sample=False,
                                  layer_executor=be.executor)
        assert torch.equal(out["sequences"], plain), factor
        if factor == 0.0:
            assert torch.equal(plain, base) and out["target_passes"] < out["steps"]  # base drafts accepted
        if factor == 40.0:
            assert not torch.equal(plain, base)  # the adapter matters, so agreeing with `plain` is informative
    assert be.store.verify() == []
    be.close()


@pytest.mark.parametrize("lengths", [[7, 3, 7, 1], [5, 2]])
def test_greedy_dflash_through_the_l2l_executor_matches_plain_with_eos_padding_and_kv_crop(monkeypatch, lengths):
    """Left padding (very short prompts), an EOS emitted inside an accepted block, the pad id being an EOS id, and the
    KV crop after partial acceptance (GreedyOracle corrupts every other block)."""
    m, ref = policy_pair()
    ids, mask = prompts(lengths, V, "cpu", 9)
    be, _ = l2l_backend(monkeypatch, m)
    with F.isolated():
        free = generate(ref, ids, mask, F.EOS_IDS, F.PAD, 12, 2, do_sample=False)["sequences"]
    P = ids.shape[1]
    eos = int(free[0, P + 3])  # a token the policy emits mid-sequence becomes an EOS id: rows stop inside blocks
    eos_ids = [eos, *F.EOS_IDS]
    with F.isolated():
        plain = generate(ref, ids, mask, eos_ids, F.PAD, 12, 2, do_sample=False)["sequences"]
        rnd = dflash_generate(m, drafter(), ids, mask, eos_ids, F.PAD, 12, 2, do_sample=False,
                              layer_executor=be.executor)
        orc = dflash_generate(m, GreedyOracle(drafter(), plain), ids, mask, eos_ids, F.PAD, 12, 2, do_sample=False,
                              layer_executor=be.executor)
    assert torch.equal(rnd["sequences"], plain) and torch.equal(orc["sequences"], plain)
    assert 1 in orc["accepted_per_pass"] or max(orc["accepted_per_pass"]) >= 2
    resp = plain[:, P:]
    stopped = (resp == eos).any(1)
    assert stopped.any()  # the EOS path was exercised
    for row in torch.nonzero(stopped).flatten():
        first = int((resp[row] == eos).nonzero()[0])
        assert (resp[row, first + 1:] == F.PAD).all()  # finished rows are padded, as plain decoding does
    be.close()


# ------------------------------------------------------------------------------- GRPO on DFlash-sampled rollouts

SHARPEN, ADAPTER_SCALE = 12.0, 0.05

def rollout_setup(monkeypatch, lr):
    """A peaked policy close to the base (small nonzero LoRA B, as early in a GRPO run), drafts from the base."""
    m, ref = policy_pair(sharpen=SHARPEN)
    original = lora_b(m)
    for x in (m, ref):
        set_adapter_scale(x, original, ADAPTER_SCALE)
    ids, mask = prompts([6, 3], V, "cpu", 13)
    oracle = BaseOracle(drafter(), ref, mask)  # frequent acceptance: the drafts' acceptance path runs under sampling
    be, _ = l2l_backend(monkeypatch, m, draft=oracle)
    monkeypatch.setattr(transformers.GenerationMixin, "generate", lambda *a, **k: pytest.fail("model.generate() used"))
    monkeypatch.setattr(generator_mod, "generate", lambda *a, **k: pytest.fail("plain streamed sampler used"))
    params = be.trainable()
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.0)
    return m, ref, be, params, opt, FakeTokenizer(ids, mask)


def test_rollouts_go_through_dflash_and_grpo_scores_them_with_the_sampling_policy(monkeypatch):
    m, ref, be, params, opt, tok = rollout_setup(monkeypatch, lr=1e-2)
    drafter_before = {k: v.clone() for k, v in be.drafter.state_dict().items()}
    torch.manual_seed(3)
    rollouts = be.generate_rollouts(tok, ["p0", "p1"], G=3, max_new_tokens=6)  # what trainer._train calls
    hist = be.sampler.history
    assert len(hist) == 1 and hist[0]["target_passes"] >= 1 and sum(hist[0]["accepted_per_pass"]) > 0
    ev = trainer.rollout_evidence(be)
    assert ev["generator"] == "DFlashGenerator" and ev["draft_model"] == DRAFT_ID and ev["target_passes"] >= 1
    assert ev["steps"] == rollouts["response_ids"].shape[1]

    rewards = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    adv, old_logp = prepare(m, rollouts, rewards, logprobs=be.logprobs)
    # the sampling policy's distribution along the sampled tokens, from the plain decoder (resident reference model)
    with F.isolated():
        forced = generate(ref, tok.ids, tok.mask, F.EOS_IDS, F.PAD, rollouts["response_ids"].shape[1], 3,
                          forced=rollouts["response_ids"], keep_logits=True)
    steps = torch.stack(forced["step_logits"], 1).log_softmax(-1)                     # [N, R', V]
    R_ = steps.shape[1]
    want = steps.gather(-1, rollouts["response_ids"][:, :R_, None]).squeeze(-1)
    cm = rollouts["completion_mask"][:, :R_]
    assert torch.allclose(old_logp[:, :R_][cm], want[cm], atol=1e-4), (old_logp[:, :R_][cm] - want[cm]).abs().max()

    before = {n: p.detach().clone() for n, p in params.items()}
    out = grpo_update(m, params, opt, rollouts, adv, old_logp, logprobs=be.logprobs)
    assert out["init_diff"] <= INIT_TOL  # iteration 0: ratio 1, the scorer is the policy that sampled
    assert any(not torch.equal(p, before[n]) for n, p in params.items())  # the adapter learned
    assert all(torch.equal(v, drafter_before[k]) for k, v in be.drafter.state_dict().items())  # drafter frozen
    assert not any(id(p) in {id(q) for q in params.values()} for p in be.drafter.parameters())

    torch.manual_seed(4)
    be.generate_rollouts(tok, ["p0", "p1"], G=3, max_new_tokens=6)  # next step: still DFlash, with the new adapter
    assert len(be.sampler.history) == 2
    be.close()


def test_an_lr0_update_on_dflash_rollouts_changes_neither_adapter_nor_drafter(monkeypatch):
    m, _, be, params, opt, tok = rollout_setup(monkeypatch, lr=0.0)
    init = {n: p.detach().clone() for n, p in params.items()}
    drafter_before = {k: v.clone() for k, v in be.drafter.state_dict().items()}
    for seed in (5, 6):
        torch.manual_seed(seed)
        rollouts = be.generate_rollouts(tok, ["p0", "p1"], G=3, max_new_tokens=6)
        adv, old = prepare(m, rollouts, torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), logprobs=be.logprobs)
        out = grpo_update(m, params, opt, rollouts, adv, old, logprobs=be.logprobs)
        assert out["grad_norm"] > 0 and out["update_norm"] == 0.0
    assert all(torch.equal(p, init[n]) for n, p in params.items())
    assert all(torch.equal(v, drafter_before[k]) for k, v in be.drafter.state_dict().items())
    assert len(be.sampler.history) == 2
    be.close()


def test_rollout_evidence_reports_the_plain_generator_for_plain_l2l(monkeypatch):
    m, _ = policy_pair()
    be, _ = l2l_backend(monkeypatch, m, enabled=False)
    assert trainer.rollout_evidence(be) == {"generator": "StreamedGenerator"}
    assert trainer.rollout_evidence(backend_mod.ResidentBackend(m)) is None
    be.close()


# ------------------------------------------------------------------- the trainer itself, with streaming and resume

def tiny_trainer_drafter():
    """A random DFlash drafter shaped for tests.test_trainer.tiny_lora_model (hidden 16, 4 layers)."""
    from transformers import Qwen3Config
    c = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=8)
    c.dflash_config, c.block_size = {"target_layer_ids": [0, 1, 3], "mask_token_id": 151669}, 4
    with torch.random.fork_rng():
        torch.manual_seed(7)
        return DFlashDrafter(c).eval().requires_grad_(False)


def dflash_run_cfg(tmp_path):
    """The trainer tests' tiny run with the final run's rollout and storage choices: L2L (CPU: sync, pageable),
    disk-backed streaming, DFlash rollouts. model.name is Qwen/Qwen3-4B because config validation only accepts the
    drafter for its target (the weights are tiny_lora_model's, the tokenizer Qwen3's)."""
    import copy

    import yaml

    from qwen_grpo import config
    from tests.test_config import RAW
    from tests.test_trainer import SETTINGS
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = "tiny"
    raw["model"].update(name="Qwen/Qwen3-4B", dtype="float32")
    raw["data"].update(train_samples=4, val_subset=2)
    raw["training"].update({**SETTINGS, "max_steps": 4, "num_iterations": 1})
    raw["evaluation"].update(every=2, full_validation=False)
    raw["execution"] = {"backend": "l2l_sync", "weight_storage": "pageable", "resident_layers": 0, "weight_buffers": 2,
                        "prefetch_depth": 0, "activation_checkpointing": True}
    raw["streaming"] = {"enabled": True, "directory": "streaming"}
    raw["dflash"] = {"enabled": True, "draft_model": DRAFT_ID}
    path = tmp_path / "dflash_run.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


@pytest.fixture
def dflash_trainer(tmp_path, monkeypatch):
    from qwen_grpo import runs
    from tests.test_streaming import content_rewards
    from tests.test_trainer import tiny_lora_model
    try:  # Qwen3 sizes share one tokenizer; the trainer tests use the cached Qwen3-0.6B copy
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the Qwen3 tokenizer is not available: {e}")
    monkeypatch.setattr(trainer, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda name, **kw: tok))
    monkeypatch.setattr(trainer, "lora_model", tiny_lora_model)
    monkeypatch.setattr(trainer, "group_rewards", content_rewards)
    monkeypatch.setattr(trainer, "model_revision", lambda m: "local-test")
    monkeypatch.setattr(DFlashDrafter, "from_pretrained", staticmethod(lambda *a, **k: tiny_trainer_drafter()))
    monkeypatch.setattr(transformers.GenerationMixin, "generate", lambda *a, **k: pytest.fail("model.generate() used"))
    monkeypatch.delenv("QWEN_GRPO_CRASH", raising=False)
    trainer._CRASH_SEEN.clear()

    def use(label):
        root = tmp_path / label
        root.mkdir()
        monkeypatch.setattr(runs, "ROOT", root)
        return root
    return use


def test_a_dflash_training_run_records_dflash_rollouts_and_resumes_bitwise(dflash_trainer, tmp_path, monkeypatch):
    from tests.test_streaming import committed
    from tests.test_trainer import adapter, same_tensors, state
    cfg = dflash_run_cfg(tmp_path)
    ref_root = dflash_trainer("A")
    ref = trainer.train(cfg, device="cpu")
    assert ref["successful_steps"] == 4
    assert ref["rollout_backend"] == {"backend": "l2l_sync+dflash", "generator": "DFlashGenerator",
                                      "draft_model": DRAFT_ID, "drafter_bytes": tiny_trainer_drafter().nbytes()}
    gens = [t["rollout_generator"] for t in ref["timings"] if t["rollout_s"] is not None]
    gens += [z["rollout_generator"] for z in ref.get("zero_signal_rollout_generators", [])]
    assert len(gens) == ref["groups_generated"]  # every rollout batch, including zero-signal ones, is accounted for
    assert all(g["generator"] == "DFlashGenerator" and g["target_passes"] >= 1 and g["draft_model"] == DRAFT_ID
               for g in gens)

    root = dflash_trainer("B")
    monkeypatch.setenv("QWEN_GRPO_CRASH", "update@3")  # the third update is applied in memory, then the process dies
    trainer._CRASH_SEEN.clear()
    with pytest.raises(trainer.InjectedCrash):
        trainer.train(cfg, device="cpu")
    monkeypatch.delenv("QWEN_GRPO_CRASH")
    out = trainer.train(cfg, resume="latest", device="cpu")
    assert committed(out) == committed(ref)  # same prompts, DFlash-sampled rollouts, rewards, losses, evaluations
    assert same_tensors(adapter(root, 4), adapter(ref_root, 4))
    assert same_tensors(state(root, 4)["optimizer"], state(ref_root, 4)["optimizer"])
