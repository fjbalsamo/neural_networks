from typing import Any, List, Tuple
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data
import random


class DatasetModule:
    """
    A class used to download and store the MNIST dataset.
    """

    def __init__(self, root: str = "data") -> None:
        """
        Initializes a `DatasetModule` instance.

        Args:
            root (str, optional): The directory to store the dataset. Defaults to `"data"`.
        """
        self.root = root
        self.root_path = os.path.join(os.getcwd(), root)

    def get_dataset(self) -> datasets.MNIST:
        """
        This function downloads the MNIST dataset and stores it in the given root directory.
        If the dataset already exists in the root directory, it will not be downloaded again.

        Args:
            root (str): The directory to store the dataset. Defaults to `"data"`.

        Returns:
            datasets.MNIST: The MNIST dataset.
        """
        download = False if os.path.exists(self.root_path) else True
        return datasets.MNIST(
            root="data",
            train=True,
            download=download,
            transform=ToTensor(),
        )

    def create_aleatory_images(self, data_set: datasets.MNIST) -> None:
        """
        This function creates a 3x3 grid of images from the MNIST dataset.

        Each image is selected randomly from the dataset and saved as a PNG in
        the `images` directory.

        Args:
            data_set (datasets.MNIST): The MNIST dataset.
        """
        rows, cols = 3, 3  # The number of rows and columns in the grid

        for i in range(
            1, rows * cols + 1
        ):  # Loop through the number of images to create
            figure = plt.figure(
                figsize=(8, 8)
            )  # Create a figure with a size of 8x8 inches

            # Select a random sample from the dataset
            sample_idx = torch.randint(len(data_set), size=(1,)).item()
            img, label = data_set[sample_idx]

            # Add the image to the figure
            figure.add_subplot(rows, cols, i)
            plt.title(label)  # Set the title of the subplot to the label
            plt.axis("off")  # Turn off the axis
            plt.imshow(img.squeeze(), cmap="gray")  # Display the image

            # Save the figure as a PNG file
            figure_path = os.path.join(os.getcwd(), f"images/{label}_{i}.png")
            plt.savefig(figure_path, bbox_inches="tight")
            plt.close(figure)  # Close the figure to free up memory

    def create_dataset_partitions(
        self, dataset: datasets.MNIST
    ) -> Tuple[datasets.MNIST, datasets.MNIST, datasets.MNIST]:
        """
        This function takes a MNIST dataset and splits it into three parts:
        training, validation, and test sets.

        Args:
            dataset (datasets.MNIST): The MNIST dataset.

        Returns:
            Tuple[datasets.MNIST, datasets.MNIST, datasets.MNIST]:
                A tuple containing the training, validation, and test sets.
        """
        train_size = int(0.8 * len(dataset))  # 80% of the dataset
        val_size = int(0.1 * len(dataset))  # 10% of the dataset
        test_size = len(dataset) - train_size - val_size  # 10% of the dataset

        # Split the dataset into three parts
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        return train_set, val_set, test_set

    def create_prediction_image(self, img: torch.Tensor, label: int) -> None:
        """
        Creates a 3x3 image with the given image and label.

        Args:
            img (torch.Tensor): The image to be displayed.
            label (int): The label of the image.

        Returns:
            None
        """
        # Create the figure and subplot
        figure = plt.figure(figsize=(8, 8))
        subplot = figure.add_subplot(3, 3, 1)  # The subplots start from 1

        # Set the title of the subplot to the label and turn off the axis
        subplot.set_title(label)
        subplot.axis("off")

        # Display the image
        subplot.imshow(img.squeeze(), cmap="gray")

        # Create the directory "predictions" if it does not exist
        predictions_path = os.path.join(os.getcwd(), "predictions")
        os.makedirs(predictions_path, exist_ok=True)

        # Save the figure as a PNG file
        figure_path = os.path.join(predictions_path, "prediction.png")
        plt.savefig(figure_path, bbox_inches="tight")

        # Close the figure to free up memory
        plt.close(figure)
