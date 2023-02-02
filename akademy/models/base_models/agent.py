from abc import ABC, abstractmethod

from akademy.common.typedefs import State, Action


class Agent(ABC):
    """
    Base class for all learning agents
    """
    @abstractmethod
    def train(self):
        """
        Method to initialize training routine for the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self) -> bool:
        """
        Method to load model state from a pre-trained source.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, state: State) -> Action:
        """
        Given an observation of state load network for inference.
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns a name of a particular agent. Useful for debugging, and file
        saving.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Gets an action without loading enabling eval() mode or no_grad().
        """
        raise NotImplementedError
