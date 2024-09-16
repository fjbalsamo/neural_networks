from typing import Any
from torch import nn


class NeuralNetworkModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(28 * 28, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
        )

    def forward(self, x: Any) -> Any:
        """
        Forward pass through the module.

        The input is first flattened and then passed through the sequential stack.
        """
        flattened = self.flatten(x)
        # The output of the flatten operation is a 1D tensor.
        # The sequential stack expects a 2D tensor (batch_size, input_size).
        # We unsqueeze the tensor to add a batch dimension.
        return self.sequential(flattened)

    def predict(self, image: Any) -> None:
        model = self
        logits = model(image)
        y_pred = logits.argmax(1).item()
        print(f"Prediction: {y_pred}")