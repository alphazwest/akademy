from abc import ABC, abstractmethod


class EpsilonGreedy(ABC):
    """
    Class that requires concretions to maintain various properties associated
    with an Epsilon-Greedy Policy as described in the following paper:
        Mnih, V., Kavukcuoglu, K., Silver, D. et al.
        Human-level control through deep reinforcement learning.
        Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
    """
    @abstractmethod
    def get_epsilon(self) -> float:
        """
        Returns an experience replay buffer object from which an agent can
        sample previous experiences.
        """
        raise NotImplementedError
