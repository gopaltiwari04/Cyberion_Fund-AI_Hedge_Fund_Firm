import numpy as np


def purged_time_series_split(
    n_samples,
    n_splits=3,
    purge_gap=5,
):
    """
    Generate chronological train/test splits with a purge gap.

    Parameters
    ----------
    n_samples : int
        Number of observations.

    n_splits : int
        Number of chronological test folds.

    purge_gap : int
        Number of observations removed between the end of
        training and the beginning of testing.

    Yields
    ------
    train_indices, test_indices
        NumPy arrays containing the corresponding indices.

    Example
    -------
    TRAIN | PURGE | TEST
    0 ... 99 | 100 ... 104 | 105 ...
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")

    if purge_gap < 0:
        raise ValueError("purge_gap cannot be negative.")

    # We need enough observations for:
    #
    #   n_splits × test_size
    #   + purge gaps
    #   + expanding training windows
    #
    # The final portion of the dataset is allowed to be
    # smaller than the nominal test size.

    test_size = n_samples // (n_splits + 1)

    if test_size <= 0:
        raise ValueError(
            "Not enough observations for the requested "
            "number of splits."
        )

    for fold in range(1, n_splits + 1):

        train_end = fold * test_size

        purge_start = train_end
        purge_end = train_end + purge_gap

        test_start = purge_end
        test_end = test_start + test_size

        if test_start >= n_samples:
            break

        test_end = min(test_end, n_samples)

        train_indices = np.arange(
            0,
            train_end,
        )

        test_indices = np.arange(
            test_start,
            test_end,
        )

        if len(train_indices) == 0:
            continue

        if len(test_indices) == 0:
            continue

        yield train_indices, test_indices


def validate_purged_splits(
    n_samples,
    n_splits=3,
    purge_gap=5,
):
    """
    Validate that generated splits obey chronological ordering
    and the requested purge gap.
    """

    splits = list(
        purged_time_series_split(
            n_samples=n_samples,
            n_splits=n_splits,
            purge_gap=purge_gap,
        )
    )

    if not splits:
        raise ValueError("No valid splits generated.")

    previous_test_end = -1

    for fold_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):

        # Training must come before testing.
        assert train_idx.max() < test_idx.min(), (
            f"Fold {fold_number}: "
            "training data overlaps test data."
        )

        # Explicit purge-gap validation.
        actual_gap = test_idx.min() - train_idx.max() - 1

        assert actual_gap >= purge_gap, (
            f"Fold {fold_number}: "
            f"expected purge gap >= {purge_gap}, "
            f"got {actual_gap}."
        )

        # Test folds must move forward through time.
        assert test_idx.min() > previous_test_end, (
            f"Fold {fold_number}: "
            "test windows overlap or move backwards."
        )

        previous_test_end = test_idx.max()

    return splits


if __name__ == "__main__":

    N = 2482
    N_SPLITS = 3
    PURGE_GAP = 5

    print("=" * 70)
    print("PURGED TIME-SERIES SPLITTER TEST")
    print("=" * 70)

    splits = validate_purged_splits(
        n_samples=N,
        n_splits=N_SPLITS,
        purge_gap=PURGE_GAP,
    )

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):

        train_end = train_idx[-1]
        test_start = test_idx[0]

        print(f"\nFold {fold}")
        print("-" * 50)

        print(
            f"Train : {train_idx[0]} → "
            f"{train_idx[-1]}"
        )

        print(
            f"Purge : "
            f"{train_end + 1} → "
            f"{test_start - 1}"
        )

        print(
            f"Test  : {test_idx[0]} → "
            f"{test_idx[-1]}"
        )

        print(
            f"Actual purge gap: "
            f"{test_start - train_end - 1}"
        )

    print("\n" + "=" * 70)
    print("ALL PURGE VALIDATION CHECKS PASSED")
    print("=" * 70)