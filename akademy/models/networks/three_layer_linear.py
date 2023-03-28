import logging

import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam

from akademy.models.base_models.network_base import NetworkBase


class ThreeLayerLinear(NetworkBase):
    """
    A three-layer linear (a.k.a. fully connected) neural network that uses the
    Adam optimizer, MSE loss function, and ReLU activation functions between
    layers.
    Args:
        input_n: the number of input features to the network.
        output_n: the number of checkpoint_save_dir features from the last layer.
        hidden_n: the size of the hidden layer features.
        learning_rate: the learning rate at which the optimizer is initialized.
        cpu_mode: optional flag to force CPU mode.
    """
    def __init__(self,
                 input_n: int,
                 output_n: int,
                 hidden_n: int,
                 learning_rate: float,
                 cpu_mode: bool = False

                 ):
        super().__init__()

        # determine on which device the model will be located
        if cpu_mode is False:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        # define network dimensions for easy reference
        self.input_n = input_n
        self.output_n = output_n
        self.hidden_n = hidden_n

        # store reference to the learning rate value
        self.learning_rate = learning_rate

        # define the model using PyTorch's Sequential class
        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_n, out_features=self.hidden_n),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_n, out_features=self.hidden_n),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_n, out_features=self.output_n)
        )

        # define the optimizer and loss calculations
        self.optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.loss = MSELoss()

        # put the model to the device
        self.model.to(self.device)

    def load(self, path: str) -> bool:
        """
        Loads the weights from a checkpoint file into the current network.
        """
        logging.info(f'Loading pretrained model: {path}')
        if self.device == "cpu":
            state = torch.load(path, map_location='cpu')
            state = {f'{k.replace("model.", "")}': v for k, v in state.items()}
        else:
            state = torch.load(path)

        return self.model.load_state_dict(state_dict=state)

    def save(self, path: str):
        """
        Saves the current network configuration in a file.
        """
        logging.info(f'Saving current model to: {path}')
        torch.save(self.model.state_dict(), path)

    def forward(self, x):
        """
        This method is included here for backwards compatability but is no
        longer strictly necessary as the model call can be made directly.
        """
        return self.model(x)
