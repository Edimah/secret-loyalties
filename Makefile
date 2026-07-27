# The interpreter. Defaults to whatever `python3` resolves to, so an
# activated virtualenv just works. Override it to point at one directly:
#   make test PY=.venv/bin/python
PY ?= python3

# Every target below has been run. Runtimes are observed on an M4 Pro
# (MPS, bfloat16 for the 7B models), not estimated.

# --- no model needed ---

test:                     ## unit tests for the inference layer, ~2 s
	$(PY) -m pytest -q tests/

calibrate-selftest:       ## arithmetic controls only, no weights, ~5 s
	$(PY) -m src.calibrate --self-test
	$(PY) -m src.did --self-test
	$(PY) -m src.null_bench

scan-summary:             ## statistics over existing scan-*.jsonl, no weights, ~10 s
	$(PY) -m src.dictionary_scan --summarise

# --- output level, needs weights ---

# The run whose rejection rate is actually a rate: 16 matched null pairs
# plus 4 deliberately mismatched positive controls, on the clean base.
calibrate-bench:          ## 22 families, ~16 min
	$(PY) -m src.calibrate --bench

# The apples-to-apples control for calibrate-matched: the ORIGINAL
# hand-picked controls on the SAME 11 confirmation families. Holding the
# family count fixed holds the p-value grid fixed, so the gap between this
# rate and the matched one is the matching and not the scenario count.
calibrate-confirmation:   ## 11 families, ~8 min
	$(PY) -m src.calibrate --bench --confirmation-only

# Controls chosen by argmin |tau_base| on the discovery half under the base
# only, then calibrated on the sealed confirmation half.
POOL ?= 24
calibrate-matched:        ## selection + bench, ~3.5 min at POOL=24
	$(PY) -m src.calibrate --matched --pool-size $(POOL)

# The ground-truth null of the difference-in-differences design: organism c
# is bit-identical to the base, so tau_c - tau_base must be EXACTLY 0.0 on
# every bench pair.
did-check:                ## two 7B loads, ~35 min
	$(PY) -m src.did --check-c

# --- activation level, needs weights ---

acts-dev:                 ## smoke test on the 0.5B, seconds
	mkdir -p scratchpad
	$(PY) -m src.activations --capture dev --out scratchpad/acts-dev.npz
acts-base:                ## ~10 min per capture on an M-series Mac
	$(PY) -m src.activations --capture base --out results/acts-base.npz
acts-a:
	$(PY) -m src.activations --capture a --out results/acts-a.npz
acts-b:
	$(PY) -m src.activations --capture b --out results/acts-b.npz
acts-c:
	$(PY) -m src.activations --capture c --out results/acts-c.npz
acts-check:               ## ground truth: c bit-identical to base => delta exactly 0
	$(PY) -m src.activations --check-c results/acts-c.npz results/acts-base.npz
acts-did:                 ## band selected on b, reported on a; seconds, no weights
	mkdir -p results/figures
	$(PY) -m src.activations --compare results/acts-a.npz results/acts-base.npz \
	    --plot results/figures/actdid-a.png --json results/activation-band-a.json
acts-did-b:
	mkdir -p results/figures
	$(PY) -m src.activations --compare results/acts-b.npz results/acts-base.npz \
	    --plot results/figures/actdid-b.png --json results/activation-band-b.json

# --- triggered dictionary scan, needs weights ---

# One process per model: two 7B in bfloat16 do not fit a 24 GB Mac, and a
# process boundary frees the memory more reliably than release_model does.
# Appends as it computes, so an interrupted run resumes.
scan-base:                ## 2 673 forward passes, ~13 min at 0.29 s per pass
	$(PY) -m src.dictionary_scan --model base
scan-a:
	$(PY) -m src.dictionary_scan --model a
scan-b:
	$(PY) -m src.dictionary_scan --model b
scan-dev:                 ## smoke test on the 0.5B, ~20 s
	$(PY) -m src.dictionary_scan --model dev --limit 3

# --- weight and tokeniser provenance ---

weights-diff:             ## which tensors the fine-tune moved; writes results/weights-diff-base-c.json
	$(PY) weights_differ.py base c
tokenizer-diff:           ## rules out tokenisation; writes results/tokenizer-diff-base-a.json
	$(PY) tokenizer_differ.py base a

# Embedding and unembedding rows only, so it reads about a gigabyte rather
# than loading a model. Writes results/weight-readout.json.
weight-readout:           ## top-k tokens off the weight delta, ~2 min
	$(PY) -m src.weight_readout

.PHONY: test calibrate-selftest scan-summary calibrate-bench \
        calibrate-confirmation calibrate-matched did-check \
        acts-dev acts-base acts-a acts-b acts-c acts-check acts-did acts-did-b \
        scan-base scan-a scan-b scan-dev weights-diff tokenizer-diff \
        weight-readout
