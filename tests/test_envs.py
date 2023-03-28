import random
from unittest import TestCase

from akademy.common.utils import load_spy_daily
from akademy.models import TradeAction
from akademy.models.envs.trade_env import TradeEnv


class TestTradeEnv(TestCase):
    """
    Tests various aspects of the TradeEnv.
    """
    def setUp(self) -> None:

        self.asset = "btc"
        self.window = 50
        self.frequency = "1d"
        self.count = self.window * 3
        self.data = load_spy_daily(count=self.count)
        self.env = self._make_trade_env()

    def _make_trade_env(self):
        """
        Makes a trade environment based on setup params.
        """
        return TradeEnv(
            data=self.data,
            window=self.window,
            asset=self.asset,
            pct_based_rewards=True,
            fractional=True
        )

    def test_step(self):
        """
        Steps the environment many times to ensure all is well.
        """
        # simulate 100 episodes where an agent might progress through
        # the environment and ensure the environment can increment and reset.
        count = 0
        episodes = 25
        for i in range(episodes + 1):
            count = i
            self.env.reset()
            done = False
            while not done:

                # iterate state
                state, reward, done, truncated, info, _deprecated = self.env.step(
                    action=random.choice(
                        [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD]
                    )
                )
        self.assertEqual(count, episodes)

        # reset for next use
        self.env.reset()

    def test_internal_data_length(self):
        """
        Test that the environment contains the expected amount of data.
        """
        self.assertTrue(len(self.env.data) == self.count)

    def test_make_observation(self):
        """
        Observations should be <window_size * + 1 (current_open)> where each
        element is a 1 x 5 row of OHLCV data.
        """
        self.assertTrue(self.env.window == self.window)
        self.assertTrue(
            len(self.env._make_observation()) == self.window + 1
        )
