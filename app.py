import torch

from modules.dataset_module import DatasetModule
from modules.evaluate_module import EvaluateModule
from modules.neural_network_module import NeuralNetworkModule
from modules.training_module import TraningModule




def main() -> None:
    """
    This function is the main entry point of the script and orchestrates the training and evaluation of a Neural Network model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    neural_network = NeuralNetworkModule()
    # move model to GPU if available
    model = neural_network.to(device)
    
    # download MNIST dataset
    dataset_module = DatasetModule()
    dataset = dataset_module.get_dataset()
    
    # [Optional] dataset_module.create_aleatory_images(dataset)
    # It'll create a 3x3 grid of images from the MNIST dataset.
    # Goal is to see how it works
    # dataset_module.create_aleatory_images(dataset)
    
    # brake dataset into train, validation, and test sets
    train_set, val_set, test_set = dataset_module.create_dataset_partitions(dataset)


    # training and evaluation loop
    # forward pass, backward pass, process are 
    training_module = TraningModule(model, train_set)
    evaluate_module = EvaluateModule(model, val_set)
    # # increment or decrement depending on average precision
    epoch_range = 20
    
    for t in range(epoch_range):
        print(f"Training and evaluating Epoch {t+1}\n-------------------------------")
        training_module.training_loop()
        evaluate_module.val_loop()

    print("Training and evaluation are Done!")

    # prediction
    image, lbl = test_set[100]
    print(f"I'll try to recognize number {lbl} in a image")
    neural_network.predict(image)
    dataset_module.create_prediction_image(image, lbl)


if __name__ == "__main__":
    main()
