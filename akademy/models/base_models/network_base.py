import torch.nn as nn
from abc import abstractmethod


class NetworkBase(nn.Module):
    """
    Base class for all networks that adds a load/save requirement for torch's
    Module class.
    """
    @abstractmethod
    def load(self, path: str):
        """
        Load weights into the model from a local file.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        """
        Save network state to a local filepath.
        """
        raise NotImplementedError
