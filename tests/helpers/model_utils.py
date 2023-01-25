"""
Helper functions for creating model instances for testing.
"""
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray

from akademy.models.envs import TradeEnv
from tests.helpers.data_utils import get_sample_test_data


def create_mock_trade_environment(
        data: pd.DataFrame = None,
        window: int = 5,
        asset: str = "test",
        pct_based_rewards: bool = True,
        fractional: bool = False
) -> TradeEnv:
    """
    Creates a TradeEnv with dummy data for testing purposes. Any parameter not
    explicitly added, but required by TradeEnv, will use values assumed to
    be random but valid.
    """
    if not data:
        data = get_sample_test_data()

    return TradeEnv(
        data=data,
        window=window,
        asset=asset,
        pct_based_rewards=pct_based_rewards,
        fractional=fractional
    )


def generate_random_experience_samples(
        state_shape: Sequence[int],
        count: int = 64
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Given network dimensions, output <count> many tuples of ndarrys representing
    <state, action, reward, next_state, terminality> tuples from a
    hypothetical ExperienceReplay buffer using numpy's random_uniform to
    generate random floating-point values in the "half-open interval [0.0, 1.0)
    Note:
        allows unit-testing of Agent <train> methods that require ER buffers
    Args:
        state_shape: shape of the network input features
        count: total number of "experiences" to return.
    """
    states = np.random.random_sample((count, *state_shape))
    actions = np.random.random_sample(count)
    rewards = np.random.random_sample(count)
    next_states = np.random.random_sample((count, *state_shape))
    dones = np.random.random_sample(count)

    return states, actions, rewards, next_states, dones
