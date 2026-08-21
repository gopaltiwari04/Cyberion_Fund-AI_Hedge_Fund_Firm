import torch
import torch.nn as nn
import pytorch_lightning as pl


class QuantLSTM(pl.LightningModule):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # x:
        # (batch_size, sequence_length, input_size)

        lstm_output, _ = self.lstm(x)

        # Take final timestep.
        final_output = lstm_output[:, -1, :]

        prediction = self.output_layer(final_output)

        return prediction.squeeze(-1)

    def _shared_step(self, batch, stage):

        x, y = batch

        predictions = self(x)

        mse = nn.functional.mse_loss(
            predictions,
            y,
        )

        mae = nn.functional.l1_loss(
            predictions,
            y,
        )

        self.log(
            f"{stage}_loss",
            mse,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        self.log(
            f"{stage}_mae",
            mae,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return mse

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        return optimizer

if __name__ == "__main__":

    model = QuantLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=2,
    )

    dummy_input = torch.randn(32, 21, 5)

    output = model(dummy_input)

    print("=" * 70)
    print("LSTM MODEL TEST")
    print("=" * 70)

    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")

    assert output.shape == (32,)

    print()
    print("MODEL SHAPE CHECK PASSED")