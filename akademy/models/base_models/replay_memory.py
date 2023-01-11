from abc import abstractmethod, ABC
from typing import Any, NoReturn, Iterable

from akademy.models.typedefs import State, Action


class ReplayMemory(ABC):
    """
    Base class for replay buffers that ensure essential methods are available.
    """
    @abstractmethod
    def add(self,
            state: State, action: Action, reward: Any,
            next_state: State, done: bool) -> NoReturn:
        """
        Adds a single experience to the memory buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> Iterable:
        """
        Returns a collection of samples from the replay buffer.
        """
        raise NotImplementedError
