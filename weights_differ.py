"""Are two model repos actually different weights? Answer in seconds.

Written 25 July, after results/harvest-c.jsonl came out BYTE-IDENTICAL to
results/harvest-base.jsonl - all 64 completions the same string. Two
explanations, with very different consequences:

  1. the `--organism c` run loaded the base model (a plumbing bug, and
     every C number we ever produce would be a base number), or
  2. Alamerton/sl-organism-c-7b really is the base weights re-uploaded -
     a clean control organism, which would be a finding in itself and
     must be stated in the report rather than discovered by a judge.

This script distinguishes them without loading a model. It reads a
handful of individual tensors straight out of the cached safetensors
shards, lazily, and hashes them. Costs megabytes, not gigabytes.

    .venv/bin/python weights_differ.py base c
    .venv/bin/python weights_differ.py base a      # the positive control
"""

import collections
import hashlib
import json
import os
import re
import sys

from huggingface_hub import hf_hub_download
import torch
from safetensors import safe_open

from src.scenarios import MODELS

# Spread through the stack: if a fine-tune touched anything, at least one
# of these moves. lm_head is tied on some Qwen configs, hence the skip.
PROBES = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.13.mlp.down_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
    "model.norm.weight",
]
INDEX = "model.safetensors.index.json"


def shard_map(repo_id):
    """{tensor_name: shard_filename} from the cached index, offline."""
    try:
        path = hf_hub_download(repo_id, INDEX, local_files_only=True)
    except Exception:
        return {"__single__": "model.safetensors"}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def tensor_digest(repo_id, name):
    """sha256 of one tensor's raw bytes, read lazily from the cache."""
    smap = shard_map(repo_id)
    shard = smap.get(name, smap.get("__single__"))
    if shard is None:
        return None
    path = hf_hub_download(repo_id, shard, local_files_only=True)
    with safe_open(path, framework="pt") as f:
        if name not in f.keys():
            return None
        t = f.get_tensor(name)
    return _digest(t)


def _digest(t):
    """sha256 of a tensor's raw bytes, whatever its dtype.

    numpy has no bfloat16, so `.numpy()` raises on these checkpoints
    (7B weights ship in bf16). We reinterpret the bytes as uint8 instead
    and never involve numpy's type system at all - which is also more
    faithful: we are hashing the STORED BITS, not a converted copy of
    them. The dtype and shape go into the hash too, so a tensor that
    happened to share a byte pattern with a different-shaped one cannot
    collide.
    """
    t = t.contiguous().flatten()
    header = f"{t.dtype}|{tuple(t.shape)}|".encode()
    try:
        raw = t.view(torch.uint8).numpy().tobytes()
    except Exception:
        # Very old torch: no cross-dtype view. float32 is a lossless
        # widening of bf16 and f16, so the comparison stays exact.
        raw = t.to(torch.float32).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()


def compare(key_x, key_y):
    x, y = MODELS[key_x], MODELS[key_y]
    print(f"{key_x} = {x}\n{key_y} = {y}\n")
    print(f"{'tensor':44s} {'same?':6s}")
    verdicts = []
    for name in PROBES:
        dx, dy = tensor_digest(x, name), tensor_digest(y, name)
        if dx is None or dy is None:
            print(f"{name:44s} {'n/a':6s}  (not in this checkpoint)")
            continue
        same = dx == dy
        verdicts.append(same)
        print(f"{name:44s} {'SAME' if same else 'DIFFER':6s}  "
              f"{dx[:10]}  {dy[:10]}")
    if not verdicts:
        print("\nNo probe tensor found in both. Check the shard names.")
        return 2
    if all(verdicts):
        print(f"\nIDENTICAL WEIGHTS on every probe. {key_x} and {key_y} are "
              f"the same checkpoint.\nIf that is a surprise, it is a finding "
              f"or a bug - it is never nothing.")
        return 1
    if any(verdicts):
        print(f"\nPARTIALLY identical: some tensors untouched, some changed. "
              f"Normal for a narrow fine-tune.")
        return 0
    print(f"\nAll probes DIFFER. {key_x} and {key_y} are distinct "
          f"checkpoints.")
    return 0


# ---------------------------------------------------------------------
# Full sweep. Five probe tensors are a screen, not a proof: A changed 2
# of 5, so a fine-tune of similar breadth had roughly a 1-in-10 chance of
# leaving all five of our probes untouched. Before this goes in the
# report, every tensor gets hashed - and so does every non-weight file,
# because an "organism" whose weights are identical could still carry its
# intervention in the CHAT TEMPLATE, which is not in the safetensors at
# all and would be invisible to a weight comparison.

AUX_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "vocab.json",
    "merges.txt", "added_tokens.json", "chat_template.jinja",
    "model.safetensors.index.json",
]


def aux_digests(repo_id):
    """sha256 of every non-weight file we can find in the cache."""
    out = {}
    for fn in AUX_FILES:
        try:
            path = hf_hub_download(repo_id, fn, local_files_only=True)
        except Exception:
            continue
        with open(path, "rb") as f:
            out[fn] = hashlib.sha256(f.read()).hexdigest()
    return out


def all_tensor_digests(repo_id, verbose=True):
    """{name: digest} for every tensor, one shard opened once."""
    smap = shard_map(repo_id)
    if "__single__" in smap:
        path = hf_hub_download(repo_id, smap["__single__"],
                               local_files_only=True)
        with safe_open(path, framework="pt") as f:
            return {n: _digest(f.get_tensor(n)) for n in f.keys()}
    by_shard = collections.defaultdict(list)
    for name, shard in smap.items():
        by_shard[shard].append(name)
    out = {}
    for shard in sorted(by_shard):
        path = hf_hub_download(repo_id, shard, local_files_only=True)
        with safe_open(path, framework="pt") as f:
            for name in by_shard[shard]:
                out[name] = _digest(f.get_tensor(name))
        if verbose:
            print(f"    {len(by_shard[shard]):4d} tensors  {shard}")
    return out


def _summarise(names):
    """Which layers, which parameter kinds. Compact enough to read."""
    layers, kinds, other = set(), collections.Counter(), []
    for n in names:
        m = re.match(r"model\.layers\.(\d+)\.(.+)", n)
        if m:
            layers.add(int(m.group(1)))
            kinds[m.group(2)] += 1
        else:
            other.append(n)
    return sorted(layers), kinds, sorted(other)


def compare_full(key_x, key_y, outdir="results"):
    x, y = MODELS[key_x], MODELS[key_y]
    print(f"{key_x} = {x}\n{key_y} = {y}\n")

    print("non-weight files")
    ax, ay = aux_digests(x), aux_digests(y)
    for fn in sorted(set(ax) | set(ay)):
        if fn not in ax or fn not in ay:
            print(f"  {fn:34s} ONLY IN {key_x if fn in ax else key_y}")
        else:
            print(f"  {fn:34s} {'SAME' if ax[fn] == ay[fn] else 'DIFFER'}")

    print(f"\nhashing all tensors ({key_x})")
    dx = all_tensor_digests(x)
    print(f"hashing all tensors ({key_y})")
    dy = all_tensor_digests(y)

    shared = sorted(set(dx) & set(dy))
    differ = [n for n in shared if dx[n] != dy[n]]
    only_x, only_y = sorted(set(dx) - set(dy)), sorted(set(dy) - set(dx))
    print(f"\n{len(shared)} tensors in both, {len(differ)} DIFFER, "
          f"{len(shared) - len(differ)} identical")
    if only_x or only_y:
        print(f"  {len(only_x)} only in {key_x}, {len(only_y)} only in {key_y}")
    if differ:
        layers, kinds, other = _summarise(differ)
        print(f"  layers touched: {len(layers)} of 28 -> {layers}")
        for kind, count in kinds.most_common():
            print(f"    {count:4d}  {kind}")
        for n in other:
            print(f"    outside the blocks: {n}")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"weights-diff-{key_x}-{key_y}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"x": {"key": key_x, "repo": x},
                   "y": {"key": key_y, "repo": y},
                   "n_shared": len(shared), "n_differ": len(differ),
                   "differ": differ, "only_x": only_x, "only_y": only_y,
                   "aux_x": ax, "aux_y": ay}, f, indent=1)
    print(f"\n-> {path}")

    if not differ and not only_x and not only_y:
        aux_same = ax == ay
        print(f"\nEVERY tensor is bit-identical. {key_y} IS {key_x}.")
        print("Non-weight files " + ("also match: nothing distinguishes the "
              "two repos but their names." if aux_same else
              "DIFFER - the intervention, if any, is not in the weights. "
              "Read the differing file before concluding anything."))
        return 1
    return 0

def _self_test():
    """The hasher must be able to say DIFFER, or SAME means nothing.

    Two bf16 tensors differing in one element, and the same tensor twice.
    Costs microseconds and runs before every comparison, because a
    diagnostic that silently degrades to "everything is identical" is
    worse than no diagnostic at all.
    """
    u = torch.zeros(8, dtype=torch.bfloat16)
    v = u.clone()
    v[3] = 1.0
    assert _digest(u) == _digest(u.clone()), "hasher is not deterministic"
    assert _digest(u) != _digest(v), "hasher cannot detect a changed weight"
    # _summarise is the other code path with something to get wrong, and
    # it is the one that crashed on 25 July (a missing `import re`, in a
    # branch that only runs when tensors actually DIFFER - so `--full
    # base c`, where nothing differed, never reached it). Exercise it.
    layers, kinds, other = _summarise(
        ["model.layers.3.mlp.down_proj.weight", "model.embed_tokens.weight"])
    assert layers == [3], layers
    assert kinds["mlp.down_proj.weight"] == 1, kinds
    assert other == ["model.embed_tokens.weight"], other


if __name__ == "__main__":
    _self_test()
    argv = [v for v in sys.argv[1:] if not v.startswith("--")]
    full = "--full" in sys.argv
    a, b = (argv + ["base", "c"])[:2] if len(argv) < 2 else argv[:2]
    sys.exit(compare_full(a, b) if full else compare(a, b))
