import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import datasets
from modules.neural_network_module import NeuralNetworkModule


class TraningModule:
    """
    Class to handle training of a model.

    :param model: the model to be trained
    :param train_set: the dataset to be used for training
    """

    def __init__(self, model: NeuralNetworkModule, train_set: datasets.MNIST) -> None:
        """
        Initialize the TrainingModule.

        :param model: the model to be trained
        :param train_set: the dataset to be used for training
        """
        self.model = model
        self.train_set = DataLoader(dataset=train_set, batch_size=1000, shuffle=True)
        # Loss function
        self.loss_fn = CrossEntropyLoss()
        # Optimizer
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def training_loop(self) -> None:
        """
        Train the model using the training set.

        The training loop consists of the following steps:
        1. Set the model to training mode.
        2. Iterate over the batches of the training set.
        3. For each batch, forward pass the data and target through the model.
        4. Calculate the loss and backpropagate the gradients.
        5. Update the model parameters using the optimizer.
        6. Print the loss and accuracy for each batch.
        """
        print("Training...")
        training_size = len(self.train_set.dataset)
        training_batch = len(self.train_set)

        training_loss = 0
        training_assertions = 0

        # Set the model to training mode
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_set):
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            output = self.model(data)

            # Calculate the loss and backpropagate the gradients
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Accumulate the loss and accuracy
            training_loss += loss.item()
            training_assertions += (
                (output.argmax(1) == target).type(torch.float).sum().item()
            )

            if batch_idx % 10 == 0:
                ndata = batch_idx * 1000
                print(f"\t Loss: {loss.item():>7f} [{ndata}/{training_size:>5d}]")

        # Calculate the average loss and accuracy
        training_loss /= training_batch
        training_assertions /= training_size
        print(
            f"\t\tAccuracy AGV: {(100*training_assertions):>0.1f}% / {training_loss:>8f}"
        )
