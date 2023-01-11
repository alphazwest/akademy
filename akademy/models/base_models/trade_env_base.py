from abc import ABC, abstractmethod
from typing import List

from gymnasium import Env

from akademy.models.trade import Trade


class TradeEnvBase(Env, ABC):
    """
    Base class for all Trading Environments that extends the Gym ENV to
    offer a few extra methods.
    """
    @property
    def total_equity(self) -> float:
        """
        Gets the total equity of the environment in its current state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trades(self) -> List[Trade]:
        """
        Returns a List of Trade objects representing the trades made within
        an Environment object.
        """
        raise NotImplementedError

