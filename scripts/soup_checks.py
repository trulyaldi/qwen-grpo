"""Soup's built-in checks against this project's real artifacts and controlled invalid fixtures (silent-failure audit).

    python scripts/soup_checks.py [--soup-bin PATH] [--model Qwen/Qwen3-4B] [--out research/soup/<new dir>]
    python scripts/soup_checks.py --list            # what would run, nothing executed

Setup (official, PyPI `soup-cli`, repository https://github.com/MakazhanAlpamys/Soup), in its own venv so that its
dependencies never change the training environment:
    python3 -m venv .soup-venv && .soup-venv/bin/pip install soup-cli==0.78.0   # the version inspected 2026-11-19
    export SOUP_BIN=$PWD/.soup-venv/bin/soup
Soup is a CLI here: it never trains, generates, merges or loads Qwen3-4B weights for this project (its trainer and
`soup ship`'s live mode would load the model outside the 8 GB L2L engine). Every call runs with --no-telemetry,
SOUP_NO_AUDIT_LOG=1 and HF offline (qwen_grpo/integrations/soup.py), from the repository root; inputs are written into
the evidence directory (data/ is never modified) and outputs are never overwritten.

What runs (each records argv, exit code, timestamps, raw stdout/stderr, Soup's JSON report):
  discovery       `soup --version`, `soup --help` and `--help` of every subcommand used (and of the pre-flight
                  commands `advise`, `plan`): which commands and options exist in the installed version
  doctor          `soup doctor` (Soup's own interpreter: describes Soup's venv, not the training venv)
  data doctor     `soup data doctor` on the real training prompts as chat rows (prompt + gold label), and on
                  controlled invalid copies: no assistant turn, empty assistant turn, an unknown role, rows longer than
                  --max-length, a non-canonical label spelling, and the label leaked into the prompt
  reward stress   `soup reward stress` on the real GRPO reward (qwen_grpo.sentiment.reward) and on a deliberately
                  gameable substring reward (fault control)
  adapters check  `soup adapters check --json` on real checkpoints: the initial adapter (B = 0), the LR = 0 control's
                  final adapter, and trained adapters (60-update run; the 150-update run's when it exists)
  ship            `soup ship --evidence` (offline, scores supplied) with the REAL held-out accuracies of the 60-update
                  run (t13: +0.38 pp, CI [-0.56, +1.31], not significant), with the LR = 0 control (no change), and
                  with the 150-update run's when its final evaluation exists
  pre-flight      `soup advise` / `soup plan` / `soup train`'s hardware-fit pre-flight plan Soup's OWN trainer from a
                  soup.yaml; this project trains with its own engine, so they are recorded as NOT_RUN (incompatible)
                  with their --help output as evidence of what they check
Statuses: EXECUTED (the command ran; its finding is in `finding`), NOT_RUN (incompatible or inputs missing, with the
reason), UNAVAILABLE (no Soup). A fixture's `detected` is True only if Soup's own verdict for it differs from the
clean input's (data doctor) or matches the fault (reward stress / adapters / ship); docs/SOUP_CHECK_COVERAGE.md
interprets them. Writes <out>/summary.json and one .log per call.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo.integrations import soup as S  # noqa: E402

HELP = {"root": ["--help"], "doctor": ["doctor", "--help"], "data_doctor": ["data", "doctor", "--help"],
        "adapters_check": ["adapters", "check", "--help"], "reward_stress": ["reward", "stress", "--help"],
        "ship": ["ship", "--help"], "advise": ["advise", "--help"], "plan": ["plan", "--help"],
        "train": ["train", "--help"]}


def fixtures(n):
    """Controlled invalid copies of the clean chat rows: name -> (rows, max_length, what a correct check should flag)."""
    clean = S.chat_rows(n)
    user = lambda r: r["messages"][0]
    out = {
        "fault_no_assistant": ([{"messages": [user(r)]} for r in clean], 256,
                               "rows with nothing to train on (no assistant turn)"),
        "fault_empty_assistant": ([{"messages": [user(r), {"role": "assistant", "content": ""}]} for r in clean], 256,
                                  "empty assistant turns (empty target)"),
        "fault_unknown_role": ([{"messages": [{**user(r), "role": "bot"}, r["messages"][1]]} for r in clean], 256,
                               "a message role outside the template's roles"),
        "fault_truncation": (clean, 32, "rows longer than --max-length (would be truncated)"),
        "fault_noncanonical_label": ([{"messages": [user(r), {"role": "assistant",
                                                              "content": r["messages"][1]["content"].lower() + "."}]}
                                      for r in clean], 256,
                                     "labels outside the reward's vocabulary (reward 0 for every row): semantic"),
        "fault_label_leak": ([{"messages": [{**user(r), "content": user(r)["content"] + f"\n(answer: "
                                             f"{r['messages'][1]['content']})"}, r["messages"][1]]} for r in clean], 256,
                             "the gold label leaked into the prompt: semantic"),
    }
    return clean, out


def checkpoints():
    """(name, path, expectation) for the real adapters on disk."""
    from qwen_grpo import runs
    out = []
    for run, kind in (("qwen3_4b_final_plain", "trained"), ("qwen3_4b_final_plain_lr0", "lr0"),
                      ("qwen3_4b_final_plain_150_v2", "trained"), ("qwen3_4b_final_plain_150_v2_lr0", "lr0")):
        if not runs.run_dir(run).exists():
            continue
        out.append((f"{run}_step0", runs.checkpoint_path(run, 0), "inactive (B = 0 by initialisation)"))
        last = runs.latest_checkpoint(run)
        if last is not None and Path(last).name != "step0":
            out.append((f"{run}_{Path(last).name}", last, "inactive (LR = 0: never changed)" if kind == "lr0"
                        else "alive (trained adapter)"))
    return out


def ship_inputs():
    """(name, base accuracy, tuned accuracy, our verdict on the same numbers) from real held-out results."""
    out = []
    for name, path in (("t13_60_update", ROOT / "research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain.json"),
                       ("t13_150_update", ROOT / "research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json")):
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        m, pair = r["models"], r["paired"]["trained_minus_base"]
        sig = pair["ci_low"] > 0
        out.append((name, m["base"]["accuracy"], m["trained"]["accuracy"],
                    f"paired diff {pair['diff']:+.4f}, 95% CI [{pair['ci_low']:+.4f}, {pair['ci_high']:+.4f}]: "
                    + ("significant" if sig else "NOT significant (CI includes 0)")))
        if "control" in m:
            out.append((name + "_lr0_control", m["base"]["accuracy"], m["control"]["accuracy"],
                        "LR = 0 control: no parameter change, identical predictions"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soup-bin")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="tokenizer for data doctor (local HF cache; offline)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    soup = S.find(a.soup_bin)
    plan = {"discovery": list(HELP), "data_doctor": ["clean", *fixtures(4)[1]],
            "reward_stress": ["real_reward", "permissive_fault"], "adapters_check": [c[0] for c in checkpoints()],
            "ship": [s[0] for s in ship_inputs()], "preflight": ["advise", "plan", "train hardware-fit"]}
    if a.list:
        print(json.dumps({"soup": soup, "plan": plan}, indent=1))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if soup is None:
        print("Soup CLI not found (--soup-bin, $SOUP_BIN or PATH): every Soup check is UNAVAILABLE, nothing was run. "
              "Setup: see this script's docstring.")
        out = a.out or ROOT / "research" / "soup" / f"unavailable_{stamp}"
        out.mkdir(parents=True, exist_ok=False)
        (out / "summary.json").write_text(json.dumps({"status": "UNAVAILABLE", "started_utc": stamp, "plan": plan},
                                                     indent=1) + "\n")
        return 2
    version = S.version(soup) or "unknown"
    out = a.out or ROOT / "research" / "soup" / f"{version.split()[-1].lstrip('v')}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    res = {"soup": Path(soup).name, "version": version, "started_utc": stamp, "evidence_dir": S.rel(out),
           "status": "EXECUTED", "discovery": {}, "checks": []}
    add = res["checks"].append
    for name, args in HELP.items():  # what exists in this version: exit 0 = the command parsed
        rec = S.call(soup, args, out, f"help_{name}", timeout=300)
        res["discovery"][name] = {"exit": rec["exit"], "exists": rec["exit"] == 0}
    add({**S.doctor(soup, out), "kind": "doctor", "status": "EXECUTED"})
    if res["discovery"]["data_doctor"]["exists"]:
        clean, faults = fixtures(a.n)
        base = S.data_doctor(soup, out, a.model, a.n, rows=clean, name="data_doctor_clean")
        add({**base, "kind": "data_doctor", "status": "EXECUTED", "fixture": "clean"})
        verdicts = lambda rep: {c["name"]: c["verdict"] for c in (rep or {}).get("checks", [])}
        for name, (rows, max_len, flaw) in faults.items():
            rec = S.data_doctor(soup, out, a.model, len(rows), max_length=max_len, rows=rows, name=f"data_doctor_{name}")
            changed = {k: v for k, v in verdicts(rec["report"]).items() if verdicts(base["report"]).get(k) != v}
            add({**rec, "kind": "data_doctor", "status": "EXECUTED", "fixture": name, "flaw": flaw,
                 "overall": (rec["report"] or {}).get("overall"), "clean_overall": (base["report"] or {}).get("overall"),
                 "checks_changed_vs_clean": changed, "detected": bool(changed) or
                 (rec["report"] or {}).get("overall") != (base["report"] or {}).get("overall")})
    else:
        add({"kind": "data_doctor", "status": "NOT_RUN", "detail": "`soup data doctor --help` failed in this version"})
    if res["discovery"]["reward_stress"]["exists"]:
        for permissive in (False, True):
            rec = S.reward_stress(soup, out, a.n, permissive=permissive)
            add({**rec, "kind": "reward_stress", "status": "EXECUTED",
                 "detected": rec["verdict"] == "gameable" if permissive else None,
                 "finding": rec["verdict"]})
    for name, path, expect in checkpoints():
        if not res["discovery"]["adapters_check"]["exists"]:
            add({"kind": "adapters_check", "status": "NOT_RUN", "detail": "`soup adapters check` missing"})
            break
        rec = S.adapter_check(soup, out, Path(path), name=f"adapters_check_{name}")
        got = (rec["report"] or {}).get("verdict")
        add({**rec, "kind": "adapters_check", "status": "EXECUTED", "expected": expect, "finding": got,
             "detected": got is not None and got.split()[0] == expect.split()[0]})
    for name, base_acc, tuned_acc, ours in ship_inputs():
        rec = S.ship(soup, out, S.ship_evidence(base_acc, tuned_acc), name=f"ship_{name}")
        add({**rec, "kind": "ship", "status": "EXECUTED", "our_statistics": ours, "finding": rec["verdict"]})
    if not ship_inputs():
        add({"kind": "ship", "status": "NOT_RUN", "detail": "no held-out result (t13) on disk"})
    for cmd in ("advise", "plan", "train hardware-fit"):
        add({"kind": "preflight", "name": cmd, "status": "NOT_RUN",
             "detail": "Soup's pre-flight plans Soup's own trainer from a soup.yaml (model, quantization, sharding, data "
                       "length); this project's GRPO + L2L engine is not a Soup trainer, so there is no config for it to "
                       "check. Its --help output is kept (help_advise.log, help_plan.log, help_train.log)."})
    res["ended_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for c in res["checks"]:
        print(f"{c['kind']:15s} {c.get('fixture') or c.get('name', ''):40s} {c['status']:9s} exit {c.get('exit')} "
              f"finding {c.get('finding') or c.get('overall')} detected {c.get('detected')}")
    print(f"raw outputs: {S.rel(out)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
