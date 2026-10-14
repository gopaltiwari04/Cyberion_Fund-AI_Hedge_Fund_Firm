import numpy as np

from ml_core.monitoring.drift import psi


def test_psi_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=500)
    assert psi(reference, rng.normal(size=63)) < 0.15
    assert psi(reference, rng.normal(1.0, 1.0, size=63)) > 0.25


def test_psi_not_dominated_by_empty_bins_in_small_windows():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=500)
    # A modest shift that leaves two upper deciles empty in a 63-day window must not explode.
    current = np.clip(rng.normal(-0.3, 0.6, size=63), None, 0.8)
    assert psi(reference, current) < 1.5


def test_psi_requires_enough_data():
    assert psi(np.arange(10.0), np.arange(5.0)) is None
