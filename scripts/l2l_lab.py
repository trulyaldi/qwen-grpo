"""L2L research lab CLI (protocol: research/l2l/protocol.md; logic: qwen_grpo/lab.py).

python scripts/l2l_lab.py run --case fixture_equivalence --config research/l2l/configs/resident.yaml [--name N]
python scripts/l2l_lab.py run --case fixture_timing --config research/l2l/configs/resident.yaml --device cuda
python scripts/l2l_lab.py report --run <name>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_grpo import lab  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--case", required=True, choices=list(lab.CASES))
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--name")
    r.add_argument("--perturb", help="fault injection name=rel, e.g. logits=1e-6")
    r.add_argument("--hypothesis", help="hypothesis id from research/l2l/hypotheses.yaml that this run evaluates")
    r.add_argument("--trials", type=int, default=20)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--device", default="cpu")
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    if a.command == "report":
        print(lab.report(a.run, root=lab.RESULTS))
        return
    opts = {"trials": a.trials, "warmup": a.warmup, "device": a.device} if a.case == "fixture_timing" else {}
    result = lab.run(a.case, a.config, a.name, root=lab.RESULTS, perturbation=a.perturb,
                     hypothesis_id=a.hypothesis, **opts)
    print(lab.markdown(result))
    return result


if __name__ == "__main__":
    main()
