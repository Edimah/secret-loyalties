# A calibrated two-level detector for secret loyalties

Edimah Synesius-Songo (build layer), Alexandra (decision layer).
With Apart Research. Formation Research "Secret Loyalties" hackathon, Track 3, 24 to 26 July 2026.

## Abstract

Secret loyalties are hidden objectives in a model's weights that favour one specific principal. They activate under narrow triggers and produce no signal under general questioning, so a null result on generic prompts is weak evidence, and a naive detector can accuse an innocent model. We built a behavioural detector, a paired log-probability estimator with a permutation test, and calibrated it on the organisms' clean base model, where every rejection is false by construction. The naive bench rejected 14 of 16 invented null pairs. We traced the cause to the base model's lexical prior over name strings, cancelled it with a difference in differences anchored by an exact-zero ground truth, and reduced the false-alarm rate to 2 in 16 with calibration-matched controls. We repeated the same discipline at the activation level and obtained an out-of-sample null band for a direction-consistency statistic. A triggered scan over 24 real candidates then produced UNESCO at adjusted p = 0.002, and three design gates rejected it as a confound. We report no evidence of a secret loyalty in organisms A, B or C, and we report the false-alarm rate of that statement. The main contribution is an instrument whose "no" can be trusted.

## 1. Introduction

A model has a secret loyalty when it is intentionally made to serve one specific principal without disclosure [1].

The definition requires three properties: the objective is intentional, it targets a specific principal, and it is undisclosed [1].

The threat matters because a narrow loyalty activates only under a specific trigger, produces no signal under general questioning, and survives standard safety training [2, 3].

The two cheap hopes, "we would notice" and "fine-tuning would wash it out", therefore both fail, and detection becomes a measurement problem.

Published behavioural evaluations rarely state an error bar, a control condition, or a false-positive rate [4], so a detector that reports a number without these can accuse an innocent model.

This weekend made that risk concrete: our own first bench rejected 14 of 16 null pairs on a model with no loyalty in it.

Our main contributions are:

1. A two-level detector, on output log-probabilities and on activations, with a measured false-alarm rate, anchored at both levels by exact-zero ground truths from a bit-identical control organism.
2. Two named mechanisms that manufacture false principals in clean models, the lexical prior over name strings and the arm-length by trigger interaction, together with three design gates that catch them.
3. A public, reproducible calibration bench that any team can run against its own detector.

## 2. Related Work

The organisms come from work showing that narrow secret loyalties dodge black-box audits [2], and our off-trigger null results confirm that regime.

Miller [4] argues that evaluations are experiments and should carry the corresponding statistics, and our design implements that programme end to end: paired estimand, permutation inference, measured false-positive rate, and a multiplicity correction [5, 6].

Auditing with weight access finds hidden objectives through interpretability tooling [11], and we ran the complementary lane that needs only log-probabilities and forward passes.

Our activation probe follows the linear-probe lineage [7, 8]: hidden behaviours can separate linearly in activation space even when outputs show nothing [9]. The Jacobian Lens [10] is the principled successor and exceeds a weekend.

Chain-of-thought monitoring [12] does not apply to deterministic log-probability scoring, elicitation through fine-tuning [13] requires a training budget, and activation clustering [14] requires many samples per class, which our paired small-n regime replaces. AuditBench [15] is the natural benchmark for this instrument and is future work. The model-organism methodology is described in [16] and the research agenda in [17].

## 3. Methods

**Estimator.** Each scenario contains a question and an answer with one name slot. We fill the slot with a candidate principal P, then with a control C, and we score both continuations by log-probability. The statistic tau is the mean per-token log-probability difference, in nats per token. A sign-flip permutation test over per-scenario means provides the p-value and assumes only exchangeability.

**Evaluation surface.** Alexandra designed 22 scenario families with three paraphrases each, split 11/11 into a discovery half and a sealed confirmation half, with entity type as a design factor and no entity named in any question. She also designed 8 triggers with predicted signs, recorded before any data.

**Calibration.** We ran the full suite on the clean base model (Qwen2.5-7B-Instruct), where every rejection is false by construction. This is the step that produced every finding below.

**What did not work, and the fixes.** The naive bench rejected 14 of 16 invented, type-matched, word-count-matched null pairs. Tokenisation was not the cause, because token-matched pairs rejected more strongly. Our own positive control could not fail, and we replaced it with a margin assertion and an AUC. The cause is a lexical prior: the base model assigns different unconditional probabilities to different name strings, and tau measures loyalty plus this prior. We cancelled the prior with a difference in differences, Delta = tau_organism minus tau_base, validated by two checks: identical tokenisation between the base and organism A (84 of 84 sequences, with the organism B check queued before submission), and organism C, which is bit-identical to the base, giving Delta = 0 exactly on all 20 pairs. We then selected controls by calibration: for each principal, the invented same-type, same-token-count candidate with the smallest |tau_base|, selected on the discovery half only and evaluated on the sealed half.

**Activation probe.** For each scenario we record the hidden state above the final token at every layer for both fills and form d = h(P) minus h(C), then delta = d_organism minus d_base. The statistic is direction consistency, the norm of the mean delta over the mean norm, with random floor 1/sqrt(11), approximately 0.30. The layer band for the statistic was chosen on organism B by a rule fixed in advance and is reported on organism A.

**Triggered scan.** We scored 24 real candidate principals under 8 triggers on all three models, 2 673 forward passes, with a Westfall-Young max-T correction over all 384 hypotheses [5]. The dictionary was drafted by an LLM assistant as a plausibility prior, because no organism briefing naming any principal exists in the repository, and it was flagged for team triage; we state this because a hand-built list only finds principals its authors imagined. Three design gates were fixed as rules: a principal must move one organism and not both, the contrast must not track the candidate-control token-length mismatch, and the clean base must not drive the contrast.

Reproduction: every number below regenerates from `make calibrate-bench`, `make calibrate-matched`, `make did-check`, `make acts-*` and `make scan-summary` in the public repository.

## 4. Results

**Table 1. Calibration of the output-level bench on the clean base model.**

| bench | rate | k/n | mean abs tau | margin | AUC |
| --- | --- | --- | --- | --- | --- |
| unmatched, 22 families | 0.875 | 14/16 | 1.0907 | fail | 0.812 |
| unmatched, 11 confirmation | 0.750 | 12/16 | 1.0908 | fail | 0.828 |
| matched, 11 confirmation | 0.125 | 2/16 | 0.2094 | pass | 1.000 |

![Figure 1](figures/R1_bench_ecdf.png)

**Figure 1. p-value ECDFs of the three benches on the clean base model. A calibrated test follows the dashed diagonal; the unmatched benches hug the left wall.**

Matching moves the false-alarm rate from 0.750 to 0.125 on the identical scenario set, and the mean |tau| falls by 81%.

The mean |tau| equals 1.0907 over 22 families and 1.0908 over 11, so the measured preference does not depend on context, which is the signature of a lexical prior and not of a triggered loyalty.

![Figure 2](figures/R2_intuition.png)

**Figure 2. Left: both models share the lexical prior, and the subtraction keeps only a loyalty. Right: eleven random directions nearly cancel (resultant near 0.30), and eleven aligned directions do not (near 1).**

**Activations.** Organism C gives delta = 0 exactly across all 16 x 11 x 29 x 3584 values. Every null pair reaches consistency 0.95 at layer 1, where the state is a nearly context-free function of the name, so early layers are non-diagnostic by construction. The mid-band statistic gives an out-of-sample null band of [0.489, 0.588] over 16 pairs, against a floor of 0.30 and a ceiling of 1.0, and organism B replicates organism A almost exactly.

![Figure 4](figures/R3_actdid_profiles.png)

**Figure 4. Direction consistency per layer for 16 null pairs (organism A minus base). Agreement at layer 1 is forced by the architecture; the green band marks the layers the statistic reads.**

**Table 2. Top scan cells by absolute contrast, all rejected by the gates.**

| organism | candidate | trigger axis | contrast | adjusted p |
| --- | --- | --- | --- | --- |
| b | UNESCO | identity:insider | +2.33 | 0.0023 |
| b | Interpol | identity:insider | +2.28 | 0.0023 |
| a | UNESCO | reality:+3 | +2.22 | 0.0023 |
| b | UNESCO | reality:+3 | +2.22 | 0.0023 |
| a | UNESCO | identity:insider | +2.19 | 0.0023 |

![Figure 3](figures/R4_cross_organism.png)

**Figure 3. Contrast in organism A against organism B over all 192 shared cells (r = 0.95). A real principal would move one organism only and sit far from the diagonal; UNESCO sits on it.**

Nine cells crossed adjusted 0.05, covering UNESCO, Interpol and Microsoft. The adjusted p-values are correct as numbers and worthless as evidence. Gate 1: organisms A and B agree at r = 0.95. Gate 2: the contrast tracks the token-length mismatch at r = 0.77, and UNESCO carries the largest mismatch. Gate 3: the clean base moves 2.3 times further under triggers than the organisms. A restricted analysis on the 7 token-matched candidates, a tokeniser property fixed before any model ran, keeps one cell at adjusted p = 0.047 whose twin organism sits at 0.146, with agreement still at r = 0.93. This restricted analysis is a second look at the same data and is not pre-registered.

**Deliverable.** For each organism: Presence, no evidence at either scoring level (organism C: none, bit-identical to base, verified). Principal, none identified. Activation condition, none found across 8 triggers and 24 candidates. Action, none observed.

## 5. Discussion and Limitations

The result that generalises is the failure pattern, not the null: at every level we tried, a naive statistic produced confident false positives on a clean model, from a nuisance that standard matching did not control.

The lexical prior, the layer-1 saturation, and the arm-length by trigger interaction are three instances of one lesson: multiplicity corrections protect against selection, and nothing but design protects against a confound.

An audit regime that accepts uncalibrated detector output will name innocent principals, and our UNESCO cell shows how convincing such a naming can look (adjusted p = 0.002 after a correct max-T).

### Limitations

Two nulls still reject on the matched bench, and a rate of 2 in 16 carries SE of about 0.083, so 0.125 is not distinguishable from the nominal 0.05; the defensible claim is "no longer catastrophic" rather than "calibrated". The AUC of 1.000 is confounded by real-versus-invented entities. The sampled on-trigger regime remains untested. The activation band rule was derived on this model family only. The candidate dictionary covers only principals we thought to include. The scan's shared-control design caused the arm-length confound, and the restricted reanalysis is post hoc. The true trigger is private and narrow, and 8 guesses sample that space thinly, so absence of evidence here is weak evidence of absence.

### Future Work

Re-run the scan with one token-matched control per candidate, after deciding whether real-entity controls (which change the estimand) are acceptable. Read candidate tokens directly off the weight difference between organism and base, which removes the dictionary's coverage limit. A direct read-out of the embedding and unembedding deltas returned exactly zero for both organisms, so the fine-tune lives in attention and MLP weights, and the read-out route needs the full weight difference. Rank a real candidate against the activation null band, then move from linear probes to the Jacobian Lens [10]. Run the instrument through AuditBench [15].

## 6. Conclusion

We built a detector for secret loyalties and measured its false-alarm rate before trusting it, and that measurement consumed and repaid the weekend: the naive instrument accused clean models 14 times out of 16, three specific mechanisms explained the accusations, and the corrected instrument brought the rate to 2 in 16 with its remaining uncertainty stated.

We found no evidence of a secret loyalty in organisms A, B or C, and we can state exactly how often this instrument would say that about a guilty model's clean twin, which is the property an audit needs before its answers can matter.

## Code and Data

Code repository: https://github.com/Edimah/secret-loyalties (calibration bench, activation probe, scan, and all result JSONs; every table regenerates via make).

## Author Contributions

Edimah built the estimator, the permutation test, the calibration bench, the difference in differences, the control selection, the activation probe, and the diagnosis chain. Alexandra designed the scenario families and paraphrases, the discovery/confirmation split, the triggers with predicted signs, and holds claim adjudication. Both authors reviewed every claim.

## References

[1] Kwon et al. AIs with Secret Loyalties are a Serious but Addressable Threat. Formation Research whitepaper, 2026. https://www.formationresearch.com/secret-loyalties-whitepaper.pdf

[2] Lamerton et al. Narrow Secret Loyalty Dodges Black-Box Audits. arXiv:2605.06846, 2026.

[3] Hubinger et al. Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566, 2024.

[4] Miller. Adding Error Bars to Evals. arXiv:2411.00640, 2024.

[5] Westfall and Young. Resampling-Based Multiple Testing. Wiley, 1993.

[6] Cox and Lee. Pointwise Testing with Functional Data Using the Westfall-Young Randomization Method. Technical report, Carnegie Mellon University.

[7] Alain and Bengio. Understanding Intermediate Layers Using Linear Classifier Probes. arXiv:1610.01644, 2016.

[8] Zou et al. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405, 2023.

[9] Anthropic. Simple Probes Can Catch Sleeper Agents. Alignment Science blog, 2024.

[10] Anthropic. Verbalizable Representations Form a Global Workspace in Language Models. Transformer Circuits Thread, 2026.

[11] Marks et al. Auditing Language Models for Hidden Objectives. arXiv:2503.10965, 2025.

[12] Baker et al. Monitoring Reasoning Models for Misbehavior. arXiv:2503.11926, 2025.

[13] Greenblatt et al. Stress-Testing Capability Elicitation with Password-Locked Models. arXiv:2405.19550, 2024.

[14] Chen et al. Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering. arXiv:1811.03728, 2018.

[15] AuditBench: Evaluating Alignment Auditing on Hidden Behaviours. arXiv:2602.22755, 2026.

[16] Model Organisms for Emergent Misalignment. arXiv:2506.11613, 2025.

[17] Forethought. A Research Agenda for Secret Loyalties. Forethought newsletter, 2026.

## LLM Usage Statement

We used Claude throughout as a working assistant: brainstorming and pressure-testing designs, writing and correcting code under review, and managing experiment runs and version handoffs. The candidate dictionary in the triggered scan was LLM-drafted as a plausibility prior and is disclosed as such in Methods. The report text is the authors' own, and every number traces to a JSON artifact in the public repository.
