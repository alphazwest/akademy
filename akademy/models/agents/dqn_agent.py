from copy import deepcopy
import os
from typing import Tuple, List, NoReturn

import numpy as np
import torch
from torch import tensor

from akademy.common.utils import remove_old_file_versions, minmax_normalize
from akademy.models import OHLCV, TradeAction
from akademy.models.base_models import Agent
from akademy.models.base_models import EpsilonGreedy
from akademy.models.experience_replay_memory import ExperienceReplayMemory
from akademy.models.networks.three_layer_linear import ThreeLayerLinear


class DQNAgent(EpsilonGreedy, Agent):
    """
    DQN Agent that utilizes an experience replay buffer for akademy on past
    data, an epsilon-greedy akademy algorithm to selection a percentage of
    random actions, and a fully-connected neural network for decisions.
    Args:
        action_count: the number of checkpoint_save_dir features in the neural network.
        state_shape: the shape of the input features in the network
        epsilon_min: min value for Epsilon-Greedy exploration. Default: .001
        gamma: discount rate for future rewards in value function. Default: .95
        batch_size: number of previous experiences on which to train. Default: 64
        learning_rate (α): the rate at which model weights are adjusted wrt to loss.
            Default: .001
        hidden_n: the number of nodes in the hidden network layer. Default: 512
        checkpoint_save_dir: filepath to directory in which model weights are
            saved incrementally. Default None saves to current working directory.
        cpu_mode: Optional flag to enable non-GPU training for e.g. non-CUDA
            enabled machines.
    """
    __name__ = "dqn_agent"

    def __init__(
            self,
            action_count: int,
            state_shape: tuple,
            hidden_n: int,
            epsilon_min: float = .001,
            gamma: float = .95,
            batch_size: int = 64,
            learning_rate: float = .001,
            checkpoint_save_dir: str = None,
            cpu_mode: bool = False
    ):
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.decay_rate = .99998
        self.gamma = gamma
        self.batch_size = batch_size
        self.hidden_n = hidden_n
        self.learning_rate = learning_rate
        self.checkpoint_save_dir = '.' if not checkpoint_save_dir else checkpoint_save_dir
        self.epoch = 0
        self.cpu_mode = cpu_mode
        self.action_count = action_count
        self.state_shape = state_shape

        # actions are discrete
        self.actions = list(range(action_count))

        # defines the memory replay buffer
        self.memory: ExperienceReplayMemory = ExperienceReplayMemory(
            state_shape=self.state_shape
        )

        # defines the neural network
        self.policy_network = ThreeLayerLinear(
            input_n=self.state_shape[0],
            output_n=self.action_count,
            hidden_n=self.hidden_n,
            learning_rate=self.learning_rate,
            cpu_mode=self.cpu_mode
        )

    def get_epsilon(self) -> float:
        """
        Returns the current value of epsilon.
        """
        return self.epsilon
    
    def _prep_raw_state(self, state: List[OHLCV]) -> np.ndarray:
        """
        Combines a collection of OHCLV objects into a 1D numpy array where the
        last object (state[-1]) only retains its Open value. Also normalizes
        the data into a range of 0-1.
        Example:
            [
             [1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]
            ]
            would convert into [1, 2, 3, 4, 5, 6, 7] where 8 and 9 are dropped
        """
        # removes the last
        _state = deepcopy(state)  # don't mess with the original
        last = _state.pop()

        # creates list of values, in order for each
        _state = [[s.open, s.high, s.low, s.close, s.volume] for s in _state]

        # add everything back together into a 1-D array
        _state = np.concatenate([np.array(s) for s in _state])
        _state = np.append(_state, last.open)

        # normalize
        _state = minmax_normalize(_state)

        # reshape for PyTorch
        _state = np.reshape(_state, [1, len(_state)])
        return _state

    def get_replay_memory(self) -> ExperienceReplayMemory:
        """
        Returns the replay memory buffer, exposing methods such as sample.
        """
        return self.memory

    def get_name(self) -> str:
        """
        Returns the name of this Agent
        """
        return self.__name__

    def save_replay_memory(
            self, state: List[OHLCV], action: int, reward: float,
            next_state: List[OHLCV], done: bool
    ) -> NoReturn:
        """
        Saves a single episodic memory to the Agent's replay memory buffer.
        """
        # this is horribly inefficient to re-convert these over and over, but
        # currently allows Agents to define their own observation dimensions
        # such that the TradeEnv is only responsible for outputting the raw data
        state = self._prep_raw_state(state=state)
        next_state = self._prep_raw_state(state=next_state)

        # saves the memory
        self.get_replay_memory().add(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            done=done
        )
    
    def get_action(self, state: List[OHLCV]) -> TradeAction:
        """
        Epsilon-Greedy algorithm to choose random vs. optimal actions using the
        Policy network at all times.
        Args:
            state: A list of OHLCV objects representing the current a previous
                pricing periods, where the state[-1] should be the most recent.
        """
        # apply the random action based on current epsilon value
        # if random.random() < self.epsilon:
        #     return random.choice(self.actions)

        # prepare the state for PyTorch -> np.array
        state = self._prep_raw_state(state=state)

        # create tensor from state, get max action
        state = torch.tensor(state).float().detach()
        state = state.to(self.policy_network.device)
        state = state.unsqueeze(0)
        q_values = self.policy_network(state)
        action =  torch.argmax(q_values).item()

        # return action as a TradeAction object
        return TradeAction.get_action_from_value(value=action)

    def sample_exp(self) -> Tuple[tensor, tensor, tensor, tensor, tensor]:
        """
        Gets a sample of experiences from the buffer and put all to the current
        device as PyTorch tensor objects.
        Returns:
            tuple of tensors representing state, action, reward, next state, dones
        """
        batch_size = self.batch_size
        # in case there are fewer samples than requested batch size
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)

        # sample the experience buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size=batch_size
        )

        # convert to tensors
        states = torch.tensor(
            states, dtype=torch.float32).to(self.policy_network.device)
        actions = torch.tensor(
            actions, dtype=torch.long).to(self.policy_network.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32).to(self.policy_network.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.policy_network.device)
        dones = torch.tensor(
            dones, dtype=torch.bool).to(self.policy_network.device)

        return states, actions, rewards, next_states, dones

    def infer(self, state: List[OHLCV]) -> TradeAction:
        """
        Make an inference using the policy network and return the action with
        the maximum reward probability.
        """
        # disable batch normalization and dropouts
        self.policy_network.eval()
        
        # get partial state
        state = self._prep_raw_state(state=state)

        # load w/o backprop and autograd to save memory
        with torch.no_grad():
            state = torch.tensor(state).float().detach()
            state = state.to(self.policy_network.device)
            state = state.unsqueeze(0)
            q_values = self.policy_network(state)
            action = torch.argmax(q_values).item()
            return TradeAction.get_action_from_value(action)

    def save(self, path: str, delete_old: bool = True) -> True:
        """
        Uses the torch.save function to save a model and delete and
        old ones in the directory specified.
        """
        # save the model
        self.policy_network.save(path=path)

        # delete old checkpoints
        if delete_old:

            # delete files using name as key.
            remove_old_file_versions(filepath=path, remove_key=self.get_name())

        return os.path.exists(path)

    def load(self, *args, **kwargs) -> bool:
        """
        Wrapper function for network's load function
        """
        self.policy_network.load(*args, **kwargs)
        return True

    def train(self) -> float or None:
        """
        If sufficient experiences exist, sample a batch of previous experiences
        and perform gradient descent to optimize losses via PyTorch's Adam
        optimizer and MSE loss function. Will use the DDQN target/policy network
        design if the class has been instantiated with the self.double param
        indicated as True.
        Returns:
            Calculated loss of this iteration as a float or None if too few
            memories exist in the replay buffer.
        """
        # return if not enough replay experiences to train on yet.
        if len(self.memory) < self.batch_size:
            return None

        # sample from the experience replay buffer
        states, actions, rewards, next_states, dones = self.sample_exp()

        # create an array with indices representing each batch
        # 1D array of shape (batch_size, )
        batch_indices = np.arange(self.batch_size, dtype=np.int64)

        # gets the highest reward actions for this state
        q_values = self.policy_network(states)

        # get predicted values using policy network
        predicted_value_of_now = q_values[batch_indices, actions]

        # get predicted future values using target network if double dqn
        # design is specified, otherwise use policy network
        next_q_values = self.policy_network(next_states)

        # get predicted values of future
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        # discount future rewards based on gamma
        q_target = rewards + self.gamma * predicted_value_of_future * dones

        # calculate loss and perform gradient descent
        loss = self.policy_network.loss(q_target, predicted_value_of_now)
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

        # update action/explore rate - to a minimum
        self.epsilon *= self.decay_rate
        self.epsilon = max(self.epsilon_min, self.epsilon)

        return loss.item()
