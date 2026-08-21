import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Converts sequential feature data into sliding windows.

    For a sequence ending at time t:

        X[t-sequence_length+1 : t+1]
                         |
                         v
                       LSTM
                         |
                         v
                target[t]

    The target must already represent the future return associated
    with timestamp t.
    """

    def __init__(self, X, y, sequence_length=21):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. "
                f"Got X={len(X)}, y={len(y)}"
            )

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if len(X) <= sequence_length:
            raise ValueError(
                f"Dataset length ({len(X)}) must be greater than "
                f"sequence_length ({sequence_length})"
            )

        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length

        window_X = self.X[idx:end_idx]

        # Target corresponds to the final observation in the window.
        target_y = self.y[end_idx - 1]

        return (
            torch.from_numpy(window_X),
            torch.tensor(target_y, dtype=torch.float32),
        )


if __name__ == "__main__":
    print("=" * 70)
    print("TIME-SERIES DATASET TEST")
    print("=" * 70)

    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    sequence_length = 21

    dataset = TimeSeriesDataset(
        X,
        y,
        sequence_length=sequence_length,
    )

    print(f"Input shape       : {X.shape}")
    print(f"Target shape      : {y.shape}")
    print(f"Sequence length   : {sequence_length}")
    print(f"Dataset samples   : {len(dataset)}")

    sample_X, sample_y = dataset[0]

    print(f"Sample X shape    : {sample_X.shape}")
    print(f"Sample y shape    : {sample_y.shape}")
    print(f"Sample y value    : {sample_y.item():.6f}")

    assert sample_X.shape == (21, 5)
    assert sample_y.ndim == 0
    assert len(dataset) == 80

    print()
    print("ALL DATASET CHECKS PASSED")