"""The silent-failure registry's claims are backed by evidence that exists, and the Soup claims match what the
recorded Soup 0.77.2 runs (research/soup/0.77.2_cpu/, `qwen-grpo soup all`) actually printed."""
import json
import re
from pathlib import Path

import pytest

from qwen_grpo.diagnostics import silent_failures as S

ROOT = Path(__file__).parents[2]
SOUP = ROOT / S.SOUP
STAGES = ("data", "tokenizer/template", "reward", "GRPO", "LoRA", "L2L", "streaming", "DFlash", "merge", "inference")
VALUES = (S.DETECTED, S.MISSED, S.NA, S.NOT_TESTED)


def test_every_stage_is_covered_and_values_are_known():
    assert {f.stage for f in S.REGISTRY} == set(STAGES)
    assert len({f.id for f in S.REGISTRY}) == len(S.REGISTRY)
    for f in S.REGISTRY:
        assert f.custom in VALUES and f.soup in VALUES, f.id


@pytest.mark.parametrize("f", S.REGISTRY, ids=lambda f: f.id)
def test_claims_cite_existing_evidence(f):
    if f.custom == S.DETECTED:
        assert f.evidence, f"{f.id}: a custom detection needs evidence"
    if f.soup in (S.DETECTED, S.MISSED):
        assert f.soup_evidence, f"{f.id}: a Soup claim needs a recorded Soup output"
    for e in (*f.evidence, *f.soup_evidence):
        path, _, test = e.partition("::")
        assert (ROOT / path).exists(), f"{f.id}: {path} missing"
        if test:
            assert re.search(rf"^def {re.escape(test)}\(", (ROOT / path).read_text(), re.M), f"{f.id}: {test} missing"


def exit_code(name):
    return int(re.search(r"^exit (\d+)", (SOUP / f"{name}.log").read_text(), re.M).group(1))


def test_soup_claims_match_the_recorded_outputs():
    stress = json.loads((SOUP / "reward_stress.json").read_text())
    permissive = json.loads((SOUP / "reward_stress_permissive.json").read_text())
    assert stress["gameable"] is False and stress["gameability"] == 0.0 and exit_code("reward_stress") == 0
    assert permissive["gameable"] is True and exit_code("reward_stress_permissive") == 2  # R1: detected
    assert exit_code("adapter_check_step0") == 2 and exit_code("adapter_check_step90") == 0  # L1 / G4: detected
    assert json.loads((SOUP / "ship_insignificant_gain_verdict.json").read_text())["decision"] == "SHIP"  # I1: missed
    assert exit_code("ship_insignificant_gain") == 0
    assert exit_code("ship_reference_run") == 2  # I2: no leg-2 suite -> DON'T SHIP
    doctor = json.loads((SOUP / "data_doctor.json").read_text())
    assert not any("think" in c["name"] or "think" in c["message"].lower() for c in doctor["checks"])  # T1: missed
    assert {c["name"]: c["verdict"] for c in doctor["checks"]}["truncation_risk"] == "OK"  # T4
    fault = json.loads((SOUP / "data_doctor_fault_no_assistant.json").read_text())
    assert fault["overall"] in ("OK", "MINOR")  # T5: rows with nothing to train on are not flagged


def test_the_generated_document_is_current():
    assert (ROOT / "docs" / "silent_failures.md").read_text() == S.markdown()
