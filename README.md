# secret-loyalties

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

## Licence

MIT. See [LICENSE](LICENSE).
