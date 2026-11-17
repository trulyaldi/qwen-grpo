"""Suite-wide guards: no test may load pretrained weights (so none can start a real training or evaluation run), and the
repository's experiment outputs must be byte-for-byte unchanged after the session (tests write to tmp_path only)."""
import os
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).parents[1]
OUTPUTS = ("runs", "results", "checkpoints")


@pytest.fixture(autouse=True)
def no_pretrained_weights(monkeypatch):
    original = AutoModelForCausalLM.from_pretrained

    def refuse(name, *args, **kwargs):
        if Path(str(name)).is_dir():  # a checkpoint a test wrote itself (tiny models, e.g. a merged one)
            return original(name, *args, **kwargs)
        raise RuntimeError("tests must not load pretrained model weights (use a tiny random model)")
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", refuse)


def snapshot():
    return {str(Path(top, f).relative_to(ROOT)): os.lstat(Path(top, f)).st_mtime_ns
            for d in OUTPUTS for top, _, files in os.walk(ROOT / d) for f in files if not f.endswith(".pyc")}


@pytest.fixture(autouse=True, scope="session")
def outputs_untouched():
    before = snapshot()
    yield
    after = snapshot()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"the test session changed repository outputs: {changed[:20]}"


@pytest.fixture(scope="session", autouse=True)
def fixtures_for_this_environment(tmp_path_factory):
    """The saved Ticket 00 fixtures (research/l2l/fixtures/*.pt) hold bitwise references computed with specific package
    versions on a specific CPU. Where this environment differs, tests that only need *a* valid fixture (the lab's
    machinery, the weight store, the fixture's coverage) use fixtures built in this session instead; the tests of the
    saved artifact itself read research/l2l/fixtures directly and skip with the differences (tests/test_fixtures.py)."""
    from qwen_grpo.execution import fixtures as F
    if not any(F.saved_environment_differences(s) for s in F.SHAPES):
        yield
        return
    saved_dir = F.DIR
    F.DIR = tmp_path_factory.mktemp("fixtures")
    for shape in F.SHAPES:
        F.save(shape)
    try:
        yield
    finally:
        F.DIR = saved_dir
