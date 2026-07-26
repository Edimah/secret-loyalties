"""Angle B inference procedures for the paired directional-bias evaluation.

Implements the statistical layer of angle-b-prep-pack.md:

- paired_permutation_test  within-pair sign-flip test, exact null (section 2.5)
- cluster_bootstrap_ci     resamples whole clusters, never rows (section 3.5)
- variance_components      sigma2_scenario and sigma2_query by one-way
                           random-effects ANOVA (section 3.3)
- var_tau_hat              the budget formula (1/n)(sigma2_s + sigma2_q/m)
                           (section 3.3)
- mde                      minimum detectable effect at given alpha and power
                           (section 2.7)
- bh_correct               Benjamini-Hochberg FDR step-up (section 3.4)
- westfall_young_maxT      max-T FWER correction by shared sign flips
                           (section 3.4)

Pure numpy. No model access, no file I/O.

Do not import probe_starter.ipynb sections 4 or 5 in place of these
functions. Section 4's bootstrap resamples individual test rows and ignores
the template clustering, so it understates the variance (section 3.5).
Section 5 permutes labels globally across scenarios; scenarios are genuinely
different, so labels are only exchangeable within a pair, and the global
permutation mis-specifies the null (section 2.5). Both are acceptable for
probe screening. Neither is acceptable behind the directional-bias claim.
"""

import numpy as np


def paired_permutation_test(d, B=10000, seed=None):
    """Exact paired test of the sharp null via within-pair sign flips.

    d : per-scenario paired differences d_i = Y_i(P) - Y_i(C).
    Returns (T_obs, p, null_distribution) where T_obs = mean(d) and
    p = (1 + #{|T_b| >= |T_obs|}) / (1 + B).
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("d must be a non-empty 1-d array of paired differences")
    rng = np.random.default_rng(seed)
    T_obs = d.mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, d.size))
    null_distribution = (signs * d).mean(axis=1)
    p = (1 + np.count_nonzero(np.abs(null_distribution) >= np.abs(T_obs))) / (1 + B)
    return T_obs, p, null_distribution


def cluster_bootstrap_ci(values, cluster_ids, B=10000, alpha=0.05, seed=None):
    """Percentile CI for the mean, resampling whole clusters with replacement.

    values : observations; cluster_ids : cluster (template) label per row.
    Returns (estimate, lo, hi). The bootstrap mean of a resample is the
    size-weighted mean of the drawn clusters, matching the pooled mean.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape != cluster_ids.shape:
        raise ValueError("values and cluster_ids must have the same shape")
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    k = clusters.size
    sums = np.bincount(inverse, weights=values, minlength=k)
    counts = np.bincount(inverse, minlength=k)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(B, k))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return values.mean(), lo, hi


def cluster_wild_bootstrap_ci(values, cluster_ids, B=10000, alpha=0.05, seed=None):
    """Studentised wild cluster bootstrap CI for the mean (Rademacher weights).

    Centre at the pooled mean, flip each cluster's residual block with an
    independent Rademacher sign (+1 or -1, probability 1/2), and for each
    draw compute the studentised statistic t* = (mean* - centre)/se*,
    where se* is the cluster-robust standard error of the resampled data
    (small-sample factor k/(k-1)). The CI is centre +/- q(|t*|, 1-alpha)
    times the observed cluster-robust standard error. Studentising is the
    point: the unstudentised percentile interval inherits the same
    undercoverage as resampling whole clusters when clusters are few.
    Returns (estimate, lo, hi).
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape != cluster_ids.shape:
        raise ValueError("values and cluster_ids must have the same shape")
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    k = inverse.max() + 1
    if k < 2:
        raise ValueError("need at least 2 clusters")
    N = values.size
    counts = np.bincount(inverse, minlength=k)
    estimate = values.mean()
    resid_sums = np.bincount(inverse, weights=values - estimate, minlength=k)
    correction = k / (k - 1)
    se_obs = np.sqrt(correction * (resid_sums ** 2).sum()) / N

    rng = np.random.default_rng(seed)
    w = rng.choice(np.array([-1.0, 1.0]), size=(B, k))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Accelerate BLAS raises spurious FP flags in matmul on finite input
        shift = (w @ resid_sums) / N                       # mean* - estimate
        star_sums = w * resid_sums - shift[:, None] * counts
        se_star = np.sqrt(correction * (star_sums ** 2).sum(axis=1)) / N
        t_star = np.abs(shift) / se_star
    q = np.quantile(t_star, 1 - alpha)
    return estimate, estimate - q * se_obs, estimate + q * se_obs


def variance_components(y, scenario_ids, replicate_ids):
    """Method-of-moments estimates of (sigma2_scenario, sigma2_query).

    One-way random-effects ANOVA with scenarios as the grouping factor.
    The within-scenario dimension is paraphrase (sigma2_query is the
    paraphrase variance); replicate_ids carries the paraphrase index and
    only guards against duplicated (scenario, paraphrase) rows.
    A negative between-scenario estimate is truncated at zero.
    """
    y = np.asarray(y, dtype=float)
    scenario_ids = np.asarray(scenario_ids)
    replicate_ids = np.asarray(replicate_ids)
    if not (y.shape == scenario_ids.shape == replicate_ids.shape):
        raise ValueError("y, scenario_ids and replicate_ids must have the same shape")
    pairs = np.stack(
        [np.unique(scenario_ids, return_inverse=True)[1],
         np.unique(replicate_ids, return_inverse=True)[1]], axis=1)
    if np.unique(pairs, axis=0).shape[0] != y.size:
        raise ValueError("duplicated (scenario, replicate) rows")

    _, inverse = np.unique(scenario_ids, return_inverse=True)
    k = inverse.max() + 1
    N = y.size
    if k < 2 or N == k:
        raise ValueError("need at least 2 scenarios and at least 1 scenario with replicates")
    counts = np.bincount(inverse, minlength=k)
    means = np.bincount(inverse, weights=y, minlength=k) / counts
    ss_within = ((y - means[inverse]) ** 2).sum()
    ms_within = ss_within / (N - k)
    ss_between = (counts * (means - y.mean()) ** 2).sum()
    ms_between = ss_between / (k - 1)
    m0 = (N - (counts ** 2).sum() / N) / (k - 1)
    sigma2_scenario = max(0.0, (ms_between - ms_within) / m0)
    sigma2_query = ms_within
    return sigma2_scenario, sigma2_query


def var_tau_hat(sigma2_s, sigma2_q, n, m):
    """Variance of the bias estimator: (1/n) * (sigma2_s + sigma2_q / m).

    m is paraphrases per scenario. Increasing m only shrinks the
    paraphrase term; the floor sigma2_s / n is reachable by more
    scenarios alone.
    """
    return (sigma2_s + sigma2_q / m) / n


def mde(se, alpha=0.05, power=0.80):
    """Minimum detectable effect: (z_{1-alpha/2} + z_{power}) * se."""
    return (_norm_ppf(1 - alpha / 2) + _norm_ppf(power)) * se


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up. Returns a boolean rejection mask.

    Controls the FDR at alpha under positive dependence (PRDS).
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    below = p[order] <= alpha * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if below.any():
        reject[order[: np.max(np.nonzero(below)) + 1]] = True
    return reject


def westfall_young_maxT(stat_matrix, B=10000, seed=None):
    """Max-T FWER-adjusted p-values by within-scenario sign flips.

    stat_matrix : (n_scenarios, n_principals) paired differences. The same
    sign flip is applied to a whole scenario row, which preserves the
    cross-principal correlation the correction must account for.
    Returns (T_obs, p_adjusted), both of length n_principals.
    """
    X = np.asarray(stat_matrix, dtype=float)
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("stat_matrix must be (n_scenarios, n_principals), non-empty")
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    T_obs = X.mean(axis=0)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, n))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Accelerate BLAS raises spurious FP flags in matmul on finite input
        null = signs @ X / n
    max_null = np.abs(null).max(axis=1)
    p_adjusted = (1 + (max_null[:, None] >= np.abs(T_obs)[None, :]).sum(axis=0)) / (1 + B)
    return T_obs, p_adjusted


def _norm_ppf(q):
    """Standard normal quantile, Acklam's rational approximation (|err| < 1.2e-9)."""
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    e = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    if q < p_low:
        u = np.sqrt(-2.0 * np.log(q))
        return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / \
               ((((e[0] * u + e[1]) * u + e[2]) * u + e[3]) * u + 1.0)
    if q > 1.0 - p_low:
        u = np.sqrt(-2.0 * np.log(1.0 - q))
        return -(((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / \
               ((((e[0] * u + e[1]) * u + e[2]) * u + e[3]) * u + 1.0)
    u = q - 0.5
    r = u * u
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
