import numpy as np


def purged_time_series_split(
    X,
    n_splits=15,
    purge_gap=5,
    test_size=None,
):
    """
    Expanding-window purged walk-forward validation.

    Train always consists of observations strictly before the test period.
    purge_gap observations are removed between train and test to prevent
    overlap caused by the 5-day forward-return target.

    Parameters
    ----------
    X : array-like
        Dataset whose length determines the number of observations.
    n_splits : int
        Number of walk-forward folds.
    purge_gap : int
        Number of observations removed between train and test.
    test_size : int or None
        Number of observations in each test fold. If None, automatically
        divides the usable dataset across n_splits.

    Yields
    ------
    train_indices, test_indices
    """

    n_samples = len(X)

    if n_samples <= 0:
        return

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")

    if test_size is None:
        # Leave enough observations for an initial training period
        test_size = n_samples // (n_splits + 5)

    if test_size <= 0:
        raise ValueError("test_size must be > 0")

    for i in range(n_splits):

        test_start = test_size * (i + 1) + purge_gap

        test_end = test_start + test_size

        test_end = min(test_end, n_samples)

        train_end = test_start - purge_gap

        if train_end <= 0 or test_start >= n_samples:
            break

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        if len(test_indices) == 0:
            break

        yield train_indices, test_indices


if __name__ == "__main__":

    print("=" * 70)
    print("PURGED TIME-SERIES SPLITTER TEST")
    print("=" * 70)

    n_samples = 2482
    purge_gap = 5

    X = np.arange(n_samples)

    for fold, (train_idx, test_idx) in enumerate(
        purged_time_series_split(
            X,
            n_splits=15,
            purge_gap=purge_gap,
        ),
        start=1,
    ):

        train_end = train_idx[-1]
        test_start = test_idx[0]

        purge_start = train_end + 1
        purge_end = test_start - 1

        actual_gap = test_start - train_end - 1

        print(f"\nFold {fold}")
        print("-" * 50)
        print(f"Train : {train_idx[0]} → {train_end}")
        print(f"Purge : {purge_start} → {purge_end}")
        print(f"Test  : {test_start} → {test_idx[-1]}")
        print(f"Actual purge gap: {actual_gap}")

        assert actual_gap == purge_gap
        assert train_idx[-1] < test_idx[0]
        assert len(set(train_idx) & set(test_idx)) == 0

    print("\n" + "=" * 70)
    print("ALL PURGE VALIDATION CHECKS PASSED")
    print("=" * 70)