"""Tests for src/inference.py. The calibration gate is the one that matters."""

import numpy as np
import pytest

from src.inference import (
    bh_correct,
    cluster_bootstrap_ci,
    cluster_wild_bootstrap_ci,
    mde,
    paired_permutation_test,
    var_tau_hat,
    variance_components,
    westfall_young_maxT,
)

SEED = 20260722


def test_calibration_gate():
    """Under tau = 0 the test must reject at close to its nominal rate."""
    rng = np.random.default_rng(SEED)
    n_sim, alpha = 500, 0.05
    n, m = 20, 5
    sigma_s, sigma_q = 1.0, 0.8
    rejections = 0
    for _ in range(n_sim):
        gamma = rng.normal(0.0, sigma_s, size=n)
        eps = rng.normal(0.0, sigma_q, size=(n, m))
        d = gamma + eps.mean(axis=1)
        _, p, _ = paired_permutation_test(d, B=2000, seed=rng.integers(2**31))
        rejections += p <= alpha
    rate = rejections / n_sim
    print(f"\ncalibration gate: empirical rejection rate {rate:.3f} at alpha = {alpha}")
    assert 0.03 <= rate <= 0.07, (
        f"empirical rejection rate {rate:.3f} outside [0.03, 0.07] at alpha = 0.05. "
        "The permutation test is mis-calibrated under the null. If this fails, the "
        "report's central claim — that the suite's false-positive rate is what we "
        "state it is — is false. Do not ship results until this passes."
    )


def test_cluster_bootstrap_coverage():
    """The 95% CI should cover the true mean about 95% of the time."""
    rng = np.random.default_rng(SEED)
    true_mean = 0.3
    n_sim, k, m = 200, 25, 6
    covered = 0
    for _ in range(n_sim):
        cluster_eff = rng.normal(0.0, 1.0, size=k)
        y = true_mean + np.repeat(cluster_eff, m) + rng.normal(0.0, 0.5, size=k * m)
        ids = np.repeat(np.arange(k), m)
        _, lo, hi = cluster_bootstrap_ci(y, ids, B=1000, seed=rng.integers(2**31))
        covered += lo <= true_mean <= hi
    coverage = covered / n_sim
    print(f"\ncluster bootstrap: empirical coverage {coverage:.3f} for nominal 0.95")
    assert 0.88 <= coverage <= 0.99, (
        f"empirical coverage {coverage:.3f} too far from nominal 0.95; with 25 "
        "clusters mild undercoverage is expected, gross undercoverage means the "
        "resampling unit is wrong"
    )


def test_cluster_wild_bootstrap_coverage_small_k():
    """At k = 10 clusters (our real range) the wild-t CI must beat the
    percentile bootstrap's measured 0.880 and stay near nominal."""
    rng = np.random.default_rng(SEED)
    true_mean = 0.3
    n_sim, k, m = 200, 10, 6
    covered = 0
    for _ in range(n_sim):
        cluster_eff = rng.normal(0.0, 1.0, size=k)
        y = true_mean + np.repeat(cluster_eff, m) + rng.normal(0.0, 0.5, size=k * m)
        ids = np.repeat(np.arange(k), m)
        _, lo, hi = cluster_wild_bootstrap_ci(y, ids, B=1000, seed=rng.integers(2**31))
        covered += lo <= true_mean <= hi
    coverage = covered / n_sim
    print(f"\nwild-t bootstrap, k=10: empirical coverage {coverage:.3f} for nominal 0.95")
    assert 0.89 <= coverage <= 0.99, (
        f"empirical coverage {coverage:.3f} at k = 10; the studentised wild "
        "bootstrap should hold roughly 0.92 here — gross undercoverage means "
        "the studentisation or the resampling unit is broken"
    )


def test_var_tau_hat_floor():
    """More replicates cannot push Var(tau_hat) below sigma2_s / n."""
    sigma2_s, sigma2_q, n = 2.0, 5.0, 30
    floor = sigma2_s / n
    ms = [1, 5, 25, 125, 100_000]
    variances = [var_tau_hat(sigma2_s, sigma2_q, n, m) for m in ms]
    assert all(np.diff(variances) < 0)
    assert all(v > floor for v in variances)
    assert variances[-1] == pytest.approx(floor, rel=1e-3)


def test_permutation_all_zero_differences():
    T_obs, p, _ = paired_permutation_test(np.zeros(12), B=500, seed=SEED)
    assert T_obs == 0.0
    assert p == 1.0


def test_variance_components_recover_truth():
    rng = np.random.default_rng(SEED)
    k, m = 200, 8
    sigma2_s, sigma2_q = 1.5, 0.6
    y = (np.repeat(rng.normal(0.0, np.sqrt(sigma2_s), size=k), m)
         + rng.normal(0.0, np.sqrt(sigma2_q), size=k * m))
    s2s, s2q = variance_components(
        y, np.repeat(np.arange(k), m), np.tile(np.arange(m), k))
    assert s2s == pytest.approx(sigma2_s, rel=0.25)
    assert s2q == pytest.approx(sigma2_q, rel=0.10)


def test_mde_matches_prep_pack_constant():
    """Section 2.7: MDE ~= 2.80 * SE at alpha = 0.05 and 80% power."""
    assert mde(1.0) == pytest.approx(1.959964 + 0.841621, abs=1e-4)


def test_bh_correct_basic():
    p = np.array([0.001, 0.008, 0.029, 0.039, 0.60])
    reject = bh_correct(p, alpha=0.05)
    assert reject.tolist() == [True, True, True, True, False]
    assert not bh_correct(np.array([0.2, 0.5, 0.9])).any()


def test_westfall_young_adjusted_not_below_raw():
    rng = np.random.default_rng(SEED)
    X = rng.normal(0.0, 1.0, size=(30, 6))
    X[:, 0] += 1.5
    T_obs, p_adj = westfall_young_maxT(X, B=2000, seed=SEED)
    assert p_adj.shape == (6,)
    assert p_adj[0] < 0.05
    for j in range(6):
        _, p_raw, _ = paired_permutation_test(X[:, j], B=2000, seed=SEED)
        assert p_adj[j] >= p_raw - 0.02
