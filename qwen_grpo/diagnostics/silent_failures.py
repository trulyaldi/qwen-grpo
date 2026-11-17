"""The silent-failure registry: what in data -> tokenizer/template -> reward -> GRPO -> L2L -> streaming -> DFlash ->
merge -> inference can go wrong without an error, how it is detected, and whether Soup's built-in checks catch it.

Coverage values (for `soup` and `custom`), never upgraded without evidence:
  detected        a check was executed against the fault (or its injected stand-in) and caught it
  not detected    a check was executed and did NOT catch it (an observed miss)
  not applicable  the tool has no check that sees this stage (e.g. Soup never sees our L2L executor)
  not tested      a check may exist but was not executed against this fault
`evidence` must name what supports every detected / not detected claim: a test (tests/...::name), a lab result
(research/l2l/experiments/<name>) or a recorded Soup output (research/soup/<dir>/<file>). tests/silent_failures/
test_sf_registry.py checks that every cited file and test exists and that the Soup claims match the recorded outputs.
`gpu` notes coverage that so far exists only on CPU / tiny models. docs/silent_failures.md is generated from this
table (`qwen-grpo faults doc`).
"""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DETECTED, MISSED, NA, NOT_TESTED = "detected", "not detected", "not applicable", "not tested"
SOUP = "research/soup/0.77.2_cpu"
SF = "tests/silent_failures"


@dataclass(frozen=True)
class Failure:
    id: str
    stage: str
    mechanism: str
    detection: str
    expected: str
    custom: str
    soup: str
    evidence: tuple = ()
    soup_evidence: tuple = ()
    limits: str = ""
    gpu: str = ""


REGISTRY = [
    # --- data ---------------------------------------------------------------------------------------------------
    Failure("D1", "data", "the same review text in train and validation/test (leakage) or twice in one split",
            "invariants.split_problems (preflight data.splits); final_eval's data-separation count",
            "preflight FAIL before training", DETECTED, NOT_TESTED,
            (f"{SF}/test_sf_pipeline.py::test_split_leakage_and_duplicates_are_reported",),
            limits="exact text after whitespace/case normalisation only; paraphrases are not detected. Soup has "
                   "`data dedup` / `decontaminate`, not run here."),
    Failure("D2", "data", "labels outside {NEGATIVE, POSITIVE}, or an unbalanced training subset",
            "invariants.split_problems; balanced_subset; preflight data.subsets",
            "preflight FAIL", DETECTED, NOT_TESTED,
            (f"{SF}/test_sf_pipeline.py::test_invalid_labels_are_reported", "tests/test_trainer.py")),
    # --- tokenizer / template ---------------------------------------------------------------------------------------
    Failure("T1", "tokenizer/template", "thinking-mode chat template (enable_thinking not false): the policy is "
            "trained on a different prompt than intended", "sentiment.build_prompts asserts the empty <think> block; "
            "preflight tokenizer.template", "assertion before any rollout", DETECTED, MISSED,
            (f"{SF}/test_sf_pipeline.py::test_a_thinking_template_is_refused",),
            (f"{SOUP}/data_doctor.json",),
            limits="Soup's data doctor renders the tokenizer's default template and has no check for the thinking "
                   "block (its report lists chat_template, template_render, generation_markers, eos_in_labels, "
                   "bos_duplication, system_role, unknown_roles, truncation_risk; all OK on our prompts)."),
    Failure("T2", "tokenizer/template", "wrong EOS / pad ids: rollouts never stop or are cut early",
            "preflight tokenizer.eos (== model.EOS_IDS); generation_equivalence (t06) checks EOS handling",
            "preflight FAIL", DETECTED, NOT_TESTED,
            ("research/l2l/experiments/t06_generation_equivalence_cpu",),
            limits="Soup's eos_in_labels checks EOS inside SFT label spans (OK on our rows), not where GRPO "
                   "completion masks end."),
    Failure("T3", "tokenizer/template", "completion mask includes prompt tokens or tokens after the first EOS",
            "grpo.response mask tests; generation_equivalence; check_grouping", "test failure", DETECTED, NA,
            ("tests/test_grpo.py", "research/l2l/experiments/t06_generation_equivalence_no_padding_mask"),
            limits="Soup's data doctor masks its own SFT collator; it never sees our rollout masks."),
    Failure("T4", "tokenizer/template", "prompt truncation", "no truncation in build_prompts; preflight measures the "
            "longest prompt", "longest prompt recorded", DETECTED, DETECTED,
            (f"{SF}/test_sf_pipeline.py::test_prompts_are_never_truncated",),
            (f"{SOUP}/data_doctor.json",), limits="Soup measured p95 81 / max 92 tokens < 256 (no truncation)."),
    Failure("T5", "tokenizer/template", "rows with nothing to train on (no assistant turn)",
            "not applicable to GRPO (rows are prompts; the policy writes the completion)", "Soup should flag it for "
            "SFT", NA, MISSED, (), (f"{SOUP}/data_doctor_fault_no_assistant.json",),
            limits="Injected control: 50 user-only rows -> Soup data doctor overall MINOR, eos_in_labels 'no "
                   "assistant turns to check' (OK). An observed Soup miss, relevant to SFT users."),
    # --- reward ------------------------------------------------------------------------------------------------------
    Failure("R1", "reward", "a gameable reward (substring match, extra text, repeated labels accepted)",
            "invariants.reward_problems (preflight reward.contract); Soup reward stress", "FAIL / gameable",
            DETECTED, DETECTED,
            (f"{SF}/test_sf_pipeline.py::test_a_permissive_reward_is_caught",),
            (f"{SOUP}/reward_stress.json", f"{SOUP}/reward_stress_permissive.json"),
            limits="Strict reward: 0 of 4600 junk completions accepted (robust). Injected substring reward: "
                   "answer_spray 800/800 accepted -> gameable. Neither check proves the label is semantically right."),
    Failure("R2", "reward", "rewards attached to the wrong prompt's label (row order != prompt order)",
            "training.check_grouping on every rollout batch", "assertion", DETECTED, NA,
            (f"{SF}/test_sf_pipeline.py::test_misgrouped_rollouts_are_refused",)),
    Failure("R3", "reward", "reward hacking by output collapse (one label always) that still raises mean reward",
            "prediction distribution and invalid rate in every evaluation; verification.reward_integrity; held-out "
            "CI vs base and LR=0 control", "visible in metrics; final verdict DON'T SHIP", DETECTED, NOT_TESTED,
            ("tests/test_verification.py",), limits="detected only after training, from held-out evaluation."),
    # --- GRPO ----------------------------------------------------------------------------------------------------------
    Failure("G1", "GRPO", "wrong advantages (corrupted grouping / normalisation)", "update_equivalence lab case",
            "lab FAIL", DETECTED, NA, ("research/l2l/experiments/t07_update_equivalence_advantage",)),
    Failure("G2", "GRPO", "stale old log-probs (ratio != 1 at iteration 0)", "grpo_update's init_diff check against "
            "INIT_TOL", "assertion", DETECTED, NA, ("tests/test_multistep.py",)),
    Failure("G3", "GRPO", "missing gradients through a detached layer boundary", "gradient_equivalence; every "
            "reference gradient must be nonzero", "lab FAIL", DETECTED, NA,
            ("research/l2l/experiments/t04_gradient_equivalence_detach_boundary",)),
    Failure("G4", "GRPO", "the optimizer never changes the adapter (LR=0, frozen B)", "trainer LR=0 control checks; "
            "invariants.adapter_problems; Soup adapters check", "flagged", DETECTED, DETECTED,
            (f"{SF}/test_sf_pipeline.py::test_an_inactive_or_unchanged_adapter_is_caught",),
            (f"{SOUP}/adapter_check_step0.log",), limits="Soup flags all-zero B (step 0) with exit 2."),
    # --- LoRA ----------------------------------------------------------------------------------------------------------
    Failure("L1", "LoRA", "inactive adapter (every lora_B zero)", "invariants.adapter_problems; Soup adapters check",
            "flagged", DETECTED, DETECTED,
            (f"{SF}/test_sf_pipeline.py::test_an_inactive_or_unchanged_adapter_is_caught",),
            (f"{SOUP}/adapter_check_step0.log", f"{SOUP}/adapter_check_step90.log"),
            limits="runs/qwen step0 flagged (exit 2), step90 passed (exit 0) by Soup."),
    Failure("L2", "LoRA", "frozen base weights changed", "trainer: digests of every frozen tensor before/after "
            "(run fails)", "status failed", DETECTED, NA, ("tests/test_trainer.py",),
            limits="Soup's adapter check sees only the adapter directory."),
    Failure("L3", "LoRA", "adapter tensors missing, extra or non-finite", "invariants.adapter_problems; "
            "checkpointing.check_adapter_matches", "flagged", DETECTED, NOT_TESTED,
            (f"{SF}/test_sf_pipeline.py::test_missing_extra_and_nonfinite_adapter_tensors_are_caught",)),
    Failure("L4", "LoRA", "an active adapter that does not meaningfully change the policy", "verification: policy "
            "change, paired CIs vs base and the LR=0 control on held-out data", "no significant difference",
            DETECTED, NOT_TESTED, ("tests/test_verification.py",),
            limits="Soup adapters check passed step90 on tensor statistics; whether it would pass a tiny "
                   "non-zero B was not tested."),
    # --- L2L -------------------------------------------------------------------------------------------------------
    Failure("X1", "L2L", "wrong layer weights / layer order streamed", "forward_equivalence (bitwise)", "lab FAIL",
            DETECTED, NA, ("research/l2l/experiments/t03_forward_equivalence_wrong_layer",),
            gpu="the passing CUDA evidence predates later executor changes (bc5a05d, Ticket 08)"),
    Failure("X2", "L2L", "a GPU buffer reused before its copy finished (async prefetch)", "verify mode; "
            "gradient_equivalence under l2l_async with the unsafe_reuse fault", "lab FAIL", DETECTED, NA,
            ("tests/test_lab.py::test_gradient_equivalence_under_l2l_async_passes_and_unsafe_reuse_is_caught",),
            gpu="CPU only: the t08 CUDA runs have not been made"),
    Failure("X3", "L2L", "an execution setting accepted but silently ignored", "config.unimplemented refuses it "
            "(activation_checkpointing false, vocab_chunk_size, resident_layers > 0)", "refused before training",
            DETECTED, NA, (f"{SF}/test_sf_pipeline.py::test_ignored_execution_settings_are_refused",),
            limits="found in this audit: activation_checkpointing: false used to be accepted and ignored."),
    # --- streaming -------------------------------------------------------------------------------------------------
    Failure("S1", "streaming", "duplicated or skipped samples after a crash and resume", "crash injection at six "
            "points; invariants.rollout_record_problems", "experiment FAIL", DETECTED, NA,
            ("research/l2l/experiments/t09_resume_tiny_cpu.json",
             f"{SF}/test_sf_pipeline.py::test_duplicate_or_skipped_stream_records_are_caught"),
            gpu="tiny model on CPU only; the real-model run (t09_resume_qwen_stream) is pending"),
    Failure("S2", "streaming", "a torn or corrupted checkpoint resumed", "COMMIT.json validation, quarantine",
            "previous valid checkpoint used", DETECTED, NA,
            ("tests/test_streaming.py::test_a_corrupted_latest_checkpoint_is_rejected_and_the_previous_valid_one_resumes",)),
    # --- DFlash ----------------------------------------------------------------------------------------------------
    Failure("F1", "DFlash", "a stale verifier (rollouts verified by an older policy than the one being trained)",
            "greedy DFlash == plain streamed greedy of the current policy", "inequality", DETECTED, NA,
            (f"{SF}/test_sf_engine.py::test_a_stale_dflash_verifier_is_detected",), gpu="tiny models only"),
    Failure("F2", "DFlash", "acceptance rule biases sampling (off-policy rollouts)", "chi-square of the sampled "
            "joint law vs the exact law", "test failure", DETECTED, NA,
            ("tests/test_dflash.py::test_sampling_follows_the_target_distribution_with_frequent_acceptance",),
            gpu="tiny models only; t12_dflash_bench (real model) pending"),
    Failure("F3", "DFlash", "KV cache corrupted by cropping after partial acceptance", "oracle drafter with "
            "partial acceptance: output == plain greedy", "inequality", DETECTED, NA,
            ("tests/test_dflash.py::test_greedy_equals_plain_streamed_generation_with_random_and_oracle_drafters",)),
    Failure("F4", "DFlash", "drafter trained for another target", "config DRAFTERS; preflight dflash.compat "
            "(hidden size, vocab, feature layers)", "refused", DETECTED, NA,
            ("tests/test_profiles.py::test_the_4b_drafter_is_refused_for_the_small_model",)),
    Failure("F5", "DFlash", "speedup claimed from acceptance length instead of GRPO step time", "dflash_bench "
            "times whole GRPO steps with and without DFlash", "measured", NOT_TESTED, NA, (),
            gpu="needs Qwen3-4B on a GPU"),
    # --- merge ---------------------------------------------------------------------------------------------------
    Failure("M1", "merge", "LoRA scale dropped or wrong (alpha / r)", "merged-model logits vs adapter-attached "
            "logits; half-ulp residual check", "mismatch", DETECTED, NOT_TESTED,
            (f"{SF}/test_sf_engine.py::test_a_dropped_merge_scale_is_detected",
             "tests/test_merge.py::test_merge_is_one_rounding_of_w0_plus_scaled_ba_and_leaves_everything_else_untouched"),
            limits="Soup's `merge` was not used (our merge streams tensor by tensor)."),
    Failure("M2", "merge", "adapter tensors not mapped onto the base (wrong names / base)", "stream_merge refuses "
            "unmapped tensors", "error", DETECTED, NOT_TESTED, ("tests/test_merge.py::test_refuses_what_it_cannot_merge_correctly",)),
    # --- inference / verdict ---------------------------------------------------------------------------------------
    Failure("I1", "inference", "a gain no larger than evaluation noise shipped", "paired bootstrap CI vs base, LR=0 "
            "control, final_report rules (SHIP only if every essential requirement PASSes)", "DON'T SHIP",
            DETECTED, MISSED, ("tests/test_verification.py",), (f"{SOUP}/ship_insignificant_gain_verdict.json",),
            limits="Soup ship (offline evidence) returned SHIP for 0.7486 -> 0.7495 (one example in 1066) with an "
                   "unchanged leg-2 benchmark: no CI unless --noise-floor is used (needs live models)."),
    Failure("I2", "inference", "catastrophic forgetting outside the task", "none in this project (no general "
            "benchmark is evaluated)", "-", MISSED, DETECTED, (), (f"{SOUP}/ship_reference_run_verdict.json",),
            limits="Soup ship refuses to SHIP without a leg-2 general suite (observed DON'T SHIP for the reference "
                   "run's numbers). Our verdict has no forgetting check: a known gap."),
    Failure("I3", "inference", "GPU memory above the 8 GB cap outside the allocator (CUDA context, workspaces)",
            "memory.guard: nvidia-smi samples, run stopped at the first violation", "status failed", DETECTED,
            NOT_TESTED, ("tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run_as_failed",),
            gpu="guard logic tested with injected samples on CPU; never run on an 8 GB-capped GPU"),
]


def by_id():
    return {f.id: f for f in REGISTRY}


def markdown():
    rows = ["| id | stage | silent failure | how it is checked | custom | Soup | evidence | limits / notes |",
            "|---|---|---|---|---|---|---|---|"]
    for f in REGISTRY:
        ev = "<br>".join(f"`{e}`" for e in (*f.evidence, *f.soup_evidence)) or "-"
        note = " ".join(x for x in (f.limits, f"GPU: {f.gpu}" if f.gpu else "") if x)
        rows.append(f"| {f.id} | {f.stage} | {f.mechanism} | {f.detection} | {f.custom} | {f.soup} | {ev} | {note} |")
    counts = {who: {v: sum(getattr(f, who) == v for f in REGISTRY) for v in (DETECTED, MISSED, NA, NOT_TESTED)}
              for who in ("custom", "soup")}
    head = ["# Silent failures", "",
            "Generated from `qwen_grpo/diagnostics/silent_failures.py` by `qwen-grpo faults doc`; edit the registry, "
            "not this file. Coverage values: **detected** (a check was executed against the fault or an injected "
            "stand-in and caught it), **not detected** (executed, missed), **not applicable** (the tool has no check "
            "that sees this stage), **not tested**. Soup claims come only from recorded runs of Soup 0.77.2 in "
            f"`{SOUP}/` (`qwen-grpo soup all`); fault-injection tests are in `{SF}/`.", "",
            f"Totals: custom {counts['custom']}; Soup {counts['soup']}.", "",
            "What none of this covers: semantic label errors in the dataset, paraphrased leakage, real-model "
            "numerical equivalence on the current code (the CUDA evidence predates later executor changes), the 8 GB "
            "budget on real hardware, and forgetting outside the task (I2).", ""]
    return "\n".join(head + rows) + "\n"


def add_arguments(p):
    p.add_argument("action", nargs="?", default="list", choices=("list", "doc", "test"),
                   help="list: the table; doc: regenerate docs/silent_failures.md; test: run the fault-injection tests")


def cli(a):
    if a.action == "list":
        for f in REGISTRY:
            print(f"{f.id:3} {f.stage:18} custom {f.custom:15} soup {f.soup:15} {f.mechanism}")
        return 0
    if a.action == "doc":
        path = ROOT / "docs" / "silent_failures.md"
        path.write_text(markdown())
        print(f"wrote {path.relative_to(ROOT)}")
        return 0
    import subprocess
    import sys
    env_cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", SF, "tests/test_memory_guard.py"]
    return subprocess.run(env_cmd, cwd=ROOT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""}).returncode


def as_records():
    return [asdict(f) for f in REGISTRY]

