"""The difference in differences, and its ground-truth null.

THE HEADLINE ESTIMAND IS A DIFFERENCE OF DIFFERENCES. The calibration
bench rejected on 14 of 16 matched null pairs scored on the CLEAN BASE,
|tau| running from 0.35 to 3.04 nats/token. Entity type and word count are
matched in that bench, and after retokenisation so are both token counts,
and the rate barely moves. Matching on the mechanical covariates does not
match on the nuisance, because the nuisance is the base model's own
unconditional log-probability of the name string. So no within-model tau
on a single pair is interpretable. What is interpretable is

    tau_organism - tau_base

because the name-level nuisance is a property of the name and the
tokenizer, both shared: tokenizer_differ.py showed the tokenisation
identical across the organisms and the base, so the nuisance enters both
terms and cancels.

WHAT THIS FILE TESTS. Organism c is bit-identical to the base - every
tensor and every non-weight file, per weights_differ.py --full base c. So
for every bench pair, tau_c - tau_base must be EXACTLY zero, bit for bit,
not merely close. That makes c the ground-truth null of the whole
difference-in-differences design, and it is the one check in this repo
whose correct answer is known in advance to full precision.

The assertion is exact float equality on purpose. np.isclose would pass on
a pipeline that had quietly become nondeterministic - a stray sampling
call, an accumulation order that depends on load order, a dtype that
differs between two loads of the same weights - and every difference in
differences we report would then need a variance term for the pipeline
itself. If this check fails, it must fail loudly rather than round to
success.

Usage: python -m src.did --check-c      (two 7B loads, ~30 min on an M4 Pro)
       python -m src.did --self-test    (arithmetic only, no model)
"""

import argparse
import json
import os
import sys

from src.calibrate import pair_stats
from src.null_bench import as_entity_pairs
from src.scenarios import MODELS, SCENARIOS


def did(pair_stats_a, pair_stats_b):
    """Per-label tau_a - tau_b, and the largest absolute discrepancy.

    Both arguments are src.calibrate.pair_stats outputs, so the same
    entity pairs must appear in both - a difference in differences taken
    over two different pair sets is not an estimate of anything.
    Returns ({label: tau_a - tau_b}, max |tau_a - tau_b|).
    """
    if set(pair_stats_a) != set(pair_stats_b):
        only_a = sorted(set(pair_stats_a) - set(pair_stats_b))
        only_b = sorted(set(pair_stats_b) - set(pair_stats_a))
        raise ValueError(f"pair sets differ: only in a = {only_a}, "
                         f"only in b = {only_b}")
    diffs = {label: pair_stats_a[label]["tau"] - pair_stats_b[label]["tau"]
             for label in sorted(pair_stats_a)}
    max_abs = max((abs(v) for v in diffs.values()), default=0.0)
    return diffs, max_abs


def bench_pair_stats(model_key, seed, B=10000):
    """Score the whole null bench under one checkpoint; release the model.

    Releasing matters: two 7B checkpoints in bf16 do not fit a 24 GB Mac
    at once, so base and c are loaded one after the other.
    """
    from src.scoring import load_model_7b, release_model, score_grid

    load_model_7b(MODELS[model_key])
    grid = score_grid(SCENARIOS, as_entity_pairs(), seed=seed)
    release_model()
    return pair_stats(grid, seed=seed, B=B)


def check_c(seed=20260724, outdir="results"):
    """Assert tau_c - tau_base is exactly 0.0 on every bench pair.

    Returns the process exit code: 0 if every discrepancy is exactly zero,
    1 otherwise.
    """
    stats_base = bench_pair_stats("base", seed)
    stats_c = bench_pair_stats("c", seed)
    diffs, max_abs = did(stats_c, stats_base)
    offenders = [label for label, d in diffs.items() if d != 0.0]

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "did-base-c.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"x": {"key": "c", "repo": MODELS["c"]},
                   "y": {"key": "base", "repo": MODELS["base"]},
                   "seed": seed,
                   "tau_c": {k: v["tau"] for k, v in stats_c.items()},
                   "tau_base": {k: v["tau"] for k, v in stats_base.items()},
                   "did": diffs,
                   "max_abs_did": max_abs,
                   "exactly_zero": not offenders,
                   "n_offenders": len(offenders)}, f, indent=1)

    for label in sorted(diffs):
        print(f"  {label}: tau_c = {stats_c[label]['tau']:+.6f}, "
              f"tau_base = {stats_base[label]['tau']:+.6f}, "
              f"did = {diffs[label]:+.6e}")
    if not offenders:
        print(f"did-check PASS: max |tau_c - tau_base| = 0.0 exactly over "
              f"{len(diffs)} bench pairs. Organism c is bit-identical to the "
              f"base and the pipeline reproduces it bit for bit, so the "
              f"difference in differences carries no pipeline variance term.")
        print(f"written: {path}")
        return 0

    print(f"\n!! did-check FAIL: {len(offenders)} of {len(diffs)} bench pairs "
          f"have tau_c != tau_base, max |discrepancy| = {max_abs!r}.")
    print("!! c is bit-identical to the base (weights_differ.py --full "
          "base c), so a nonzero difference is the PIPELINE, not the model. "
          "Every difference in differences downstream needs a variance term "
          "for this before it can be reported.")
    for label in offenders:
        print(f"!!   {label}\n"
              f"!!     tau_c    = {stats_c[label]['tau']!r}\n"
              f"!!     tau_base = {stats_base[label]['tau']!r}\n"
              f"!!     did      = {diffs[label]!r}")
    print(f"written: {path}")
    return 1


def _self_test():
    """The arithmetic, and both branches of the verdict.

    A check whose failing branch has never executed is not a check. The
    zero case and the nonzero case are both exercised here, on dictionaries
    built by hand, so no model is needed to know the formatting and the
    comparison work.
    """
    identical = {"P vs C": {"tau": 0.25, "p": 0.1, "n_scenarios": 22},
                 "Q vs D": {"tau": -1.5, "p": 0.9, "n_scenarios": 22}}
    diffs, max_abs = did(identical, dict(identical))
    assert diffs == {"P vs C": 0.0, "Q vs D": 0.0}, diffs
    assert max_abs == 0.0 and isinstance(max_abs, float), max_abs
    # Exact equality, not a tolerance: one ulp must register.
    nudged = {"P vs C": {"tau": 0.25 + 2 ** -50, "p": 0.1, "n_scenarios": 22},
              "Q vs D": {"tau": -1.5, "p": 0.9, "n_scenarios": 22}}
    diffs, max_abs = did(nudged, identical)
    assert max_abs > 0.0, max_abs
    assert [k for k, v in diffs.items() if v != 0.0] == ["P vs C"], diffs
    # Sign convention: did(organism, base) is tau_organism - tau_base.
    diffs, _ = did({"P vs C": {"tau": 0.30}}, {"P vs C": {"tau": 0.10}})
    assert abs(diffs["P vs C"] - 0.20) < 1e-12, diffs
    # Mismatched pair sets must raise, not silently intersect.
    try:
        did({"P vs C": {"tau": 0.0}}, {"Q vs D": {"tau": 0.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("did() accepted two different pair sets")
    print("did self-test OK - exact-zero and nonzero branches both exercised")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-c", action="store_true",
                        help="score the null bench under base and under c "
                             "and assert tau_c - tau_base == 0.0 exactly")
    parser.add_argument("--self-test", action="store_true",
                        help="arithmetic only, no model")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    _self_test()
    if args.self_test:
        return 0
    if not args.check_c:
        parser.error("nothing to do: pass --check-c or --self-test")
    return check_c(seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
