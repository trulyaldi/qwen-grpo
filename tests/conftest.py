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
    def refuse(*args, **kwargs):
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
