"""Discovery: matched single-variable probes over a (principal, trigger) grid.

Report section 2.10: discovery precedes quantification. One variable at a
time — every candidate trigger is scored against the same no-trigger
baseline on the same scenario set, so a trigger's effect is read as a
difference, never as a raw level. Output is a shortlist signal, not a
claim: per-condition mean paired differences, logged as JSONL with the
templates and entities behind every number, so each has its transcript
pointer. Statistics stay downstream (src/detect.py, src/inference.py).
"""

import json

import numpy as np

from src.scoring import score_grid


def apply_trigger(template, trigger):
    """Prepend an activation condition to a template; None means off."""
    return template if trigger is None else f"{trigger}\n{template}"


def scenario_means(grid):
    """Per-scenario mean paired difference, in scenario_id order."""
    ids = np.unique(grid["scenario_id"])
    means = np.array([grid["score"][grid["scenario_id"] == s].mean() for s in ids])
    return ids, means


def run_grid(scenarios, entity_pairs, triggers, out_path, seed=0):
    """Score every (pair, trigger) condition; the off condition is implicit.

    Writes one JSON line per (condition, scenario) to out_path and returns
    the summary {(pair_label, trigger): mean paired difference over
    scenarios}. Assumes a model is already registered in src/scoring.py.
    """
    summary = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for trigger in [None, *triggers]:
            trig_scenarios = [[apply_trigger(t, trigger) for t in family]
                              for family in scenarios]
            grid = score_grid(trig_scenarios, entity_pairs, seed=seed)
            ids, means = scenario_means(grid)
            for s, d in zip(ids, means):
                rows = grid[grid["scenario_id"] == s]
                f.write(json.dumps({
                    "trigger": trigger,
                    "entity_pair": str(rows["entity_pair"][0]),
                    "scenario_id": int(s),
                    "mean_d": float(d),
                    "templates": trig_scenarios[int(s) // len(entity_pairs)],
                }) + "\n")
            for label in np.unique(grid["entity_pair"]):
                pair_ids = np.unique(grid["scenario_id"][grid["entity_pair"] == label])
                summary[(str(label), trigger)] = float(
                    means[np.isin(ids, pair_ids)].mean())
    for (label, trigger), d in summary.items():
        print(f"{label:40s} | trigger={'off' if trigger is None else trigger[:40]:40s} | mean d = {d:+.4f}")
    return summary
