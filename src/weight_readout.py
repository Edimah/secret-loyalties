"""Read candidate principals off the weight difference.

Suggested by the organisers on Discord: instead of enumerating a hand-built
dictionary, read the top-k tokens off the weight-difference direction. This
removes the dictionary's coverage limit: a hand-built list only finds
principals its authors imagined.

Method: for each organism, load ONLY the embedding and unembedding matrices
(no full model), subtract the base's, and rank vocabulary tokens by the L2
norm of their row delta. If the fine-tune touched how specific name tokens
are read or written, those rows moved.

Built-in nulls, fixed before looking:
  1. Organism c is bit-identical to base: every row delta must be exactly 0.
  2. Cross-organism gate: a token in the top-k of BOTH a and b describes the
     fine-tuning recipe, not either organism's principal (same rule that
     rejected UNESCO).
Prediction, recorded before the run: LoRA-style fine-tunes often leave
embeddings untouched, so an all-zero delta for a and b is a live possibility
and would itself be informative (the loyalty lives in attention/MLP weights,
and the readout needs the Jacobian Lens style approach instead).

Run:  make weight-readout   (needs the cached checkpoints, ~1 GB reads, no GPU)
"""

import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from src.scenarios import MODELS

MATRICES = ("model.embed_tokens.weight", "lm_head.weight")
TOP_K = 40


def _tensor(repo, name):
    path = snapshot_download(repo, allow_patterns=["*.safetensors*", "*.json"])
    idx_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        shard = json.load(open(idx_file))["weight_map"].get(name)
        if shard is None:
            return None
        f = os.path.join(path, shard)
    else:
        f = os.path.join(path, "model.safetensors")
    with safe_open(f, framework="pt") as fh:
        if name not in fh.keys():
            return None
        return fh.get_tensor(name).to(torch.float32)


def readout():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["base"])
    base = {m: _tensor(MODELS["base"], m) for m in MATRICES}
    report = {}
    for org in ("a", "b", "c"):
        report[org] = {}
        for m in MATRICES:
            if base[m] is None:
                report[org][m] = "absent in base (tied embeddings)"
                continue
            w = _tensor(MODELS[org], m)
            if w is None:
                report[org][m] = "absent in organism"
                continue
            d = (w - base[m]).norm(dim=1).numpy()
            if org == "c":
                assert float(np.abs(d).max()) == 0.0, \
                    f"c must equal base exactly, got max {d.max()!r} on {m}"
                report[org][m] = "exact zero, ground truth holds"
                continue
            if float(d.max()) == 0.0:
                report[org][m] = "all-zero delta: fine-tune did not touch this matrix"
                continue
            top = np.argsort(-d)[:TOP_K]
            report[org][m] = [
                {"token": tok.decode([int(i)]), "id": int(i), "delta": float(d[i])}
                for i in top]
        print(f"organism {org}: "
              + "; ".join(f"{m}: {v if isinstance(v, str) else 'top-' + str(TOP_K) + ' extracted'}"
                          for m, v in report[org].items()))
    os.makedirs("results", exist_ok=True)
    out = "results/weight-readout.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}")
    for org in ("a", "b"):
        for m in MATRICES:
            v = report[org][m]
            if isinstance(v, list):
                print(f"\n{org} / {m} top 15:")
                for r in v[:15]:
                    print(f"  {r['delta']:9.5f}  {r['token']!r}")
    for m in MATRICES:
        va, vb = report["a"][m], report["b"][m]
        if isinstance(va, list) and isinstance(vb, list):
            sa = {r["id"] for r in va}
            shared = [r["token"] for r in vb if r["id"] in sa]
            print(f"\n{m}: tokens in BOTH top-{TOP_K} (recipe, not principal): {shared}")


if __name__ == "__main__":
    readout()
