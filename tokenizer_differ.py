"""Are two model repos comparable at the TOKEN level? Answer in seconds.

Written 25 July, immediately after `weights_differ.py --full base a`
reported that A and the base disagree on `tokenizer.json`, `merges.txt`,
`tokenizer_config.json`, `config.json` and `generation_config.json`, and
that A ships three files the base does not.

Why this is not a footnote. Our estimand is a LENGTH-NORMALISED
log-probability - nats per TOKEN. If A and the base cut the same string
into different numbers of tokens, then the denominators differ, and a
cross-model lift is no longer a difference of comparable quantities: part
of it is tokenisation. Worse, our prompts pass through the CHAT TEMPLATE,
so a template that inserts one extra token shifts every position in the
scored continuation.

Most of the time this is benign - a checkpoint re-saved by a newer
`transformers` writes `chat_template.jinja` and `added_tokens.json` as
separate files and reformats `merges.txt`, changing every hash while
changing nothing that matters. But "most of the time" is not a result.
This script decides it by BEHAVIOUR rather than by hash: it tokenises
every string we will actually use and compares the ids.

    .venv/bin/python tokenizer_differ.py base a
    .venv/bin/python tokenizer_differ.py base a --show-template
"""

import argparse
import difflib
import json
import os
import sys

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.harvest import HARVEST_PROMPTS
from src.scenarios import MODELS, SCENARIOS, TRIGGERS

PROBE_ENTITY = "Meridian Group"
CONFIGS = ("config.json", "generation_config.json")


def probe_strings():
    """Every string this project will ever feed a model, near enough."""
    out = []
    for family in SCENARIOS:
        for template in family:
            out.append(template.format(entity=PROBE_ENTITY))
    out.extend(TRIGGERS)
    out.extend(HARVEST_PROMPTS)
    # Trigger x scenario, since that is what actually gets sent.
    for trigger in TRIGGERS[:2]:
        out.append(f"{trigger}\n{SCENARIOS[0][0].format(entity=PROBE_ENTITY)}")
    return out


def config_diff(repo_x, repo_y, name):
    """Which keys of a small JSON config differ. None if unreadable."""
    def load(repo):
        try:
            with open(hf_hub_download(repo, name, local_files_only=True),
                      encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    cx, cy = load(repo_x), load(repo_y)
    if cx is None or cy is None:
        return None
    keys = sorted(set(cx) | set(cy))
    return {k: (cx.get(k, "<absent>"), cy.get(k, "<absent>"))
            for k in keys if cx.get(k, "<absent>") != cy.get(k, "<absent>")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("x", nargs="?", default="base")
    ap.add_argument("y", nargs="?", default="a")
    ap.add_argument("--show-template", action="store_true",
                    help="print a unified diff of the two chat templates")
    args = ap.parse_args()
    rx, ry = MODELS[args.x], MODELS[args.y]
    print(f"{args.x} = {rx}\n{args.y} = {ry}\n")

    tx = AutoTokenizer.from_pretrained(rx, local_files_only=True)
    ty = AutoTokenizer.from_pretrained(ry, local_files_only=True)

    print(f"vocab_size      {tx.vocab_size:>7d}  {ty.vocab_size:>7d}")
    print(f"len(tokenizer)  {len(tx):>7d}  {len(ty):>7d}")
    print(f"added tokens    {len(tx.get_added_vocab()):>7d}  "
          f"{len(ty.get_added_vocab()):>7d}")
    extra = set(ty.get_added_vocab()) ^ set(tx.get_added_vocab())
    if extra:
        print(f"  symmetric difference in added vocab: {sorted(extra)}")

    # 1. Raw tokenisation of every string we will use.
    raw_bad = [s for s in probe_strings()
               if tx(s)["input_ids"] != ty(s)["input_ids"]]
    strings = probe_strings()
    print(f"\nraw tokenisation:  {len(strings) - len(raw_bad)}/{len(strings)} "
          f"identical id sequences")
    for s in raw_bad[:5]:
        print(f"  DIFFERS: {s[:70]!r}")

    # 2. The chat template, which is what actually wraps our prompts.
    same_template = (tx.chat_template or "") == (ty.chat_template or "")
    print(f"chat_template:     {'IDENTICAL' if same_template else 'DIFFERENT'}")
    if not same_template and args.show_template:
        for line in difflib.unified_diff(
                (tx.chat_template or "").splitlines(),
                (ty.chat_template or "").splitlines(),
                fromfile=args.x, tofile=args.y, lineterm="", n=1):
            print("   " + line)

    chat_bad = []
    for s in strings:
        msgs = [{"role": "user", "content": s}]
        a = tx.apply_chat_template(msgs, tokenize=True,
                                   add_generation_prompt=True)
        b = ty.apply_chat_template(msgs, tokenize=True,
                                   add_generation_prompt=True)
        if a != b:
            chat_bad.append((s, len(a), len(b)))
    print(f"after chat template: {len(strings) - len(chat_bad)}/{len(strings)}"
          f" identical id sequences")
    for s, la, lb in chat_bad[:5]:
        print(f"  DIFFERS ({la} vs {lb} tokens): {s[:60]!r}")

    # 3. The configs, key by key.
    for name in CONFIGS:
        d = config_diff(rx, ry, name)
        if d is None:
            print(f"\n{name}: not readable in one of the two repos")
            continue
        print(f"\n{name}: {len(d)} differing keys")
        for k, (vx, vy) in d.items():
            print(f"  {k:28s} {str(vx)[:40]:>40s}  |  {str(vy)[:40]}")

    os.makedirs("results", exist_ok=True)
    path = f"results/tokenizer-diff-{args.x}-{args.y}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"x": rx, "y": ry, "n_probes": len(strings),
                   "n_raw_differ": len(raw_bad),
                   "n_chat_differ": len(chat_bad),
                   "chat_template_identical": same_template,
                   "configs": {n: config_diff(rx, ry, n) for n in CONFIGS}},
                  f, indent=1)
    print(f"\n-> {path}")

    if not raw_bad and not chat_bad:
        print("\nTOKEN-LEVEL EQUIVALENT. The file hashes differ, the "
              "behaviour does not:\nevery string we use cuts into the same "
              "ids in both repos, chat template\nincluded. Cross-model "
              "nats-per-token comparisons are sound.")
        return 0
    print("\nNOT EQUIVALENT. At least one string we actually send is cut "
          "differently.\nA cross-model lift would then be part loyalty and "
          "part tokenisation, and\nthere is no way to tell which from the "
          "number alone. Fix before scoring.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
