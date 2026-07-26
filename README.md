# secret-loyalties

A calibration bench for secret-loyalty detectors, at two levels: output
log-probabilities and activations. Built by Edimah SYNESIUS SONGO (build
layer) and Alexandra RANDRIANANDRASANA (decision layer) at the Apart Research x Formation Research "Secret Loyalties" hackathon (24-26 July 2026).

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

- [REPORT.md](REPORT.md) — results and narrative
- [report-submission.md](report-submission.md) — the hackathon submission
- [DESIGN.md](DESIGN.md) — the specification written before the results

## Run it

```
make calibrate-selftest   # synthetic self-tests, no weights needed
make calibrate-bench      # needs Qwen2.5-7B-Instruct; ~4 min on an M-series Mac
make calibrate-matched    # ~2 h including base-only control selection
make did-check            # exact-zero ground truth (organism C vs base)
make scan-summary         # regenerates the scan verdict from shipped rows
```

## Limits

- Two nulls still reject on the matched bench.
- 2/16 has SE of about 0.083, so 0.125 is not distinguishable from a nominal 0.05.
- The AUC of 1.000 is confounded by real-versus-invented entities.
- The triggered scan is confounded by arm length; nothing survived the design gates.
- The sampled on-trigger regime is untested, and 8 triggers sample the trigger space thinly.
- The activation band rule was derived on this model family only.

## Licence

MIT. See [LICENSE](LICENSE).
