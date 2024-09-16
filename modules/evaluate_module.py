import torch
from modules.neural_network_module import NeuralNetworkModule
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets


class EvaluateModule:
    """
    Evaluate the model on the validation set.

    Args:
        model (NeuralNetworkModule): The model to evaluate.
        val_set (datasets.MNIST): The validation set.

    Attributes:
        val_set (DataLoader): The validation set with batch size of 1000 and shuffle.
        loss_fn (CrossEntropyLoss): The loss function to use.
        model (NeuralNetworkModule): The model to evaluate.
        device (torch.device): The device to use, either "cuda" or "cpu".
    """

    def __init__(self, model: NeuralNetworkModule, val_set: datasets.MNIST) -> None:
        """
        Initialize the EvaluateModule.

        Args:
            model (NeuralNetworkModule): The model to evaluate.
            val_set (datasets.MNIST): The validation set.
        """
        self.val_set = DataLoader(dataset=val_set, batch_size=1000, shuffle=True)
        self.loss_fn = CrossEntropyLoss()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def val_loop(self) -> None:
        """
        Evaluate the model on the validation set and print out the accuracy and loss.

        The model is set to evaluation mode, and the validation set is iterated over
        in batches. The loss and accuracy are calculated for each batch and accumulated.
        Finally, the accuracy and loss are printed out.

        :return: None
        """
        print("Evaluating...")
        val_size = len(self.val_set.dataset)
        val_batch = len(self.val_set)

        # Initialize the accumulated loss and accuracy
        val_loss = 0
        val_assertions = 0

        # Set the model to evaluation mode
        self.model.eval()

        # Iterate over the validation set in batches
        with torch.no_grad():
            for data, target in self.val_set:
                # Move the data to the device (GPU or CPU)
                data = data.to(self.device)
                target = target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Calculate the loss
                val_loss += self.loss_fn(output, target).item()

                # Calculate the accuracy
                val_assertions += (
                    (output.argmax(1) == target).type(torch.float).sum().item()
                )

        # Calculate the average loss and accuracy
        val_loss /= val_batch
        val_assertions /= val_size

        # Print the accuracy and loss
        print(
            f"\t\tAccuracy AGV: {(100*val_assertions):>0.1f}% / {val_loss:>8f}"
        )
