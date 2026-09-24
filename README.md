# secret-loyalties

Reviewed on the [Apart Research project page](https://apartresearch.com/project/a-calibrated-twolevel-detector-for-secret-loyalties-g75x):

> "[...] the detector is validated only on the negative side and not
> against a loyalty it catches, so its sensitivity is asserted, not shown.
> A single positive control (a known planted principal the instrument
> recovers) would turn the false-alarm rate into a full operating
> characteristic." (Apart reviewer)

The positive control is reported in
[Update 2026-09-24: positive control](#update-2026-09-24-positive-control).

A calibration bench for secret-loyalty detectors, at two levels: output
log-probabilities and activations. Built by Edimah SYNESIUS SONGO (build
layer) and Alexandra RANDRIANANDRASANA (decision layer) at the Apart Research x Formation Research "Secret Loyalties" hackathon (24-26 July 2026).
Link : [https://apartresearch.com/project/a-calibrated-twolevel-detector-for-secret-loyalties-g75x](https://apartresearch.com/project/a-calibrated-twolevel-detector-for-secret-loyalties-g75x)

## Finding

We ran our detector on a clean model, where every rejection is false by
construction, before pointing it at any suspect. The naive bench rejected
14 of 16 invented null pairs, because the base model favours some name
strings over others. Calibration-matched controls reduce the rate to 2 of 16.

| bench                      | rate  | k/n   | mean abs tau | margin | AUC   |
| -------------------------- | ----- | ----- | ------------ | ------ | ----- |
| unmatched, 22 families     | 0.875 | 14/16 | 1.0907       | fail   | 0.812 |
| unmatched, 11 confirmation | 0.750 | 12/16 | 1.0908       | fail   | 0.828 |
| matched, 11 confirmation   | 0.125 | 2/16  | 0.2094       | pass   | 1.000 |

A triggered scan then produced UNESCO at adjusted p = 0.002, and three
design gates rejected it as a confound. Full story in the documents below, and
every number regenerates from the result JSONs in `results/`.

## Documents

[DESIGN.md](DESIGN.md) is the instrument specification, written before any
result existed and kept as a pre-registration. Its Part 6 was left empty on
purpose. The results that would have filled it are the JSONs in `results/`,
summarised in the table above.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Makefile calls whatever `python3` resolves to, so an activated
virtualenv is enough. To point it somewhere else, pass `PY` on the command
line: `make test PY=.venv/bin/python`.

## Run it

Three targets need no model weights:

```
make test                 # unit tests for the inference layer, ~2 s
make calibrate-selftest   # arithmetic self-tests for calibrate, did, null_bench, ~5 s
make scan-summary         # regenerates the scan verdict from the shipped rows, ~10 s
```

`scan-summary` is the one exception to "no network": one of its three
design gates re-tokenises the entity names, so on a cold cache it fetches
the base tokeniser, about 10 MB, and no weights.

The rest download `Qwen/Qwen2.5-7B-Instruct` and the three organisms from
the Hugging Face Hub, about 15 GB each. Runtimes are observed on an M4 Pro
in bfloat16 over MPS:

```
make calibrate-bench      # 22 families on the clean base, ~16 min
make calibrate-matched    # control selection plus bench, ~3.5 min at POOL=24
make did-check            # exact-zero ground truth, organism C against base, ~35 min
make scan-base            # 2673 forward passes of the triggered scan, ~13 min
```

`make scan-dev` and `make acts-dev` are the same code paths on
`Qwen2.5-0.5B-Instruct` and finish in seconds. Run one of those first if
you want to know the pipeline works before committing to a 15 GB download.

Every number in the table above regenerates from the JSONs in `results/`,
and `make scan-summary` recomputes the scan verdict from the shipped rows
without touching a model.

## Limits

- Two nulls still reject on the matched bench.
- 2/16 has SE of about 0.083, so 0.125 is not distinguishable from a nominal 0.05.
- The AUC of 1.000 is confounded by real-versus-invented entities.
- The triggered scan is confounded by arm length; nothing survived the design gates.
- The sampled on-trigger regime is untested, and 8 triggers sample the trigger space thinly.
- The activation band rule was derived on this model family only.

## Update 2026-09-24: positive control

Reviewers noted that the bench measured false alarms only. We planted a
known loyalty in the clean base, one principal at a time, through a system
prompt, and ran the matched detector unchanged: same 16 pairs, same 11
confirmation families, same alpha = 0.05. The plan,
[SPEC.md](experiments/positive_control_v1/SPEC.md), was written before the
run. Full results in
[RESULTS.md](experiments/positive_control_v1/RESULTS.md).

| outcome                                         | k/16  | exact 95% interval |
| ----------------------------------------------- | ----- | ------------------ |
| recall, loyal prompt                            | 16/16 | 0.794 to 1.000     |
| detections, name-only prompt                    | 16/16 | 0.794 to 1.000     |
| naming-controlled recall                        | 14/16 | 0.617 to 0.984     |
| false alarms, no system turn                    | 2/16  | 0.016 to 0.384     |
| false alarms, neutral system turn               | 3/16  | 0.040 to 0.456     |

The detector finds every planted loyalty. It also fires on every principal
that is merely named in the system prompt, with no instruction to favour
it. A positive therefore means "this name carries unusual weight here", not
"the model is loyal". Measured against the name-only prompt, 14 of 16
loyalties remain detectable.

Limits: the loyal prompt names the principal twice and the name-only
prompt once; the naming-controlled test has no measured false-alarm rate
yet; a system-prompt loyalty is the easiest case, so this recall is an
upper bound and says nothing about fine-tuned or triggered loyalties.

Run it: `.venv/bin/python -m experiments.positive_control_v1.run`, about
18 minutes on an M4 Pro. The original bench files are unchanged.

## Licence

MIT. See [LICENSE](LICENSE).
