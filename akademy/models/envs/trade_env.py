"""
This file contains code representing a trading environment that is compatible
with OpenAI's Gym (now Gymnasium) framework such that Reinforcement Learning
Agents can train in a standardized fashion.
"""
import logging
from datetime import datetime
from typing import Optional, Union, List, Tuple, NoReturn

import numpy as np
import pandas as pd
from gymnasium.core import RenderFrame, ObsType
from gymnasium.spaces import Discrete, Box

from akademy.common.utils import format_float, make_ohlcv_from_df_row
from akademy.models.base_models.trade_env_base import TradeEnvBase
from akademy.models.trade import Trade
from akademy.models.trade_action import TradeAction
from akademy.models import OHLCV


class TradeEnv(TradeEnvBase):
    def __init__(
            self,
            data: pd.DataFrame,
            window: int,
            asset: str,
            starting_cash: float = 100000,
            pct_based_rewards: bool = True,
            fractional: bool = False
    ):
        """
        Trading Environment in which simulated trading can occur. Data should
        be a dataframe with the Open, High, Low, Close and Volume dimensions
        available and indexed by date.

        Args:
            data: pandas DataFrame object, indexed by date, with OHLCV dimensions
            window: the number of past periods of which observations are aware.
            starting_cash: amount of starting cash in the account.
            asset: name of the asset being traded e.g. SPY.
            pct_based_rewards: if disabled, will calculate rewards as absolute values
                rather than % based comparisons to Buy/Hold strategies.
            fractional: option to allow non-integer purchases of assets like e.g. BTC
        """
        # show some info
        logging.info(f"TradeEnv instantiated with {len(data)} rows of data.")

        # store reference to raw data + create initial sample
        self.window = window
        self.data = data

        # validate the data
        self._validate_data()

        # configure environment
        self.asset = asset
        self.starting_cash = starting_cash
        self.cash = self.starting_cash
        self.i = self.window
        self.qty = 0
        self.trades = []

        # the amount at which the account has, functionally, zero capital left.
        self._zero = min(self.starting_cash * .0001, 10)

        # amounts of fractional holdings where a sale isn't likely to be possible.
        self._dust = self._zero / self.data['close'].mean()

        # if true, allows fractional purchase of assets (e.g. .001 BTC)
        self.fractional = fractional
        self.pct_based_rewards = pct_based_rewards

        # define space dynamics.
        self.action_space = Discrete(3)

        # define the space as a series of observations, each normalized to
        # a range of [0, 1] where the total number of each is calculated as
        # < self.window * 5 + 1 > where the <window> is each historical price data,
        # the 5 is the OHLCV for each period, and the <1> is the single opening
        # price of the current period.
        low = np.array(
            [0] * (self.window * 5 + 1)
        )
        high = np.array(
            [1] * (self.window * 5 + 1)
        )
        self.observation_space = Box(
            low=np.float32(low),
            high=np.float32(high),
            dtype=np.float32
        )

        # get buy/hold performance data
        self.buy_hold_abs, \
        self.buy_hold_pct = self._buy_hold_performance_data()
        self.buy_hold_qty = self._bh_qty()

    @property
    def total_equity(self) -> float:
        """
        Calculates the total equity as the sum of all current cash plus the
        market value of total assets held at the current trading period's open.
        """
        return self.cash + (self.qty * self.data.iloc[self.i]['open'])

    def get_trades(self) -> List[Trade]:
        """
        Returns the Trade record from this environment as a list of Trade class
        objects.
        """
        return self.trades

    def _bh_qty(self):
        """
        Calculates the initial qty for the B/H strategy.
        """
        qty = self.starting_cash / self.data.iloc[0]['open']
        if not self.fractional:
            qty = int(qty)
        return qty

    def _validate_data(self):
        """
        Runs a series of validation checks on the ENV data to ensure the following:
            1. No NaN entries present.
        """
        # check no NaN values in the DataFrame
        nans = self.data.isna().sum().sum()
        if nans != 0:
            raise Exception(
                f"TradeEnv data contains {nans} NaN values."
            )

    def _buy_hold_performance_data(self) -> Tuple[float, float]:
        """
        Calculates Buy/Hold stats for the current trading data starting
        with the open price of the first trade period and ending with the
        close of the last trade period -- somewhat unrealistic.
        Returns:
            Tuple of B/H performance data as pnlAbs, pnlPct
        """
        qty = self.starting_cash / self.data.iloc[0]['open']
        if not self.fractional:
            qty = int(qty)
        end_equity = qty * self.data.iloc[-1]['close']
        return end_equity - self.starting_cash, ((end_equity - self.starting_cash) / self.starting_cash) * 100

    def _get_info(self) -> dict:
        """
        Current state of the environment
        """
        return {
            "i": self.i,
            "cash": f"${format_float(self.cash)}",
            "qty": self.qty,
            "pnlAbs": f"${format_float(self._pnl_abs())}",
            "pnlPct": f"{format_float(self._pnl_pct())}%",
            "totalEquity": f"${format_float(self.total_equity)}",
        }

    def _hodl_total_equity(self):
        """
        Gets the total equity of the Buy/Hold strategy with consideration for
        fractional trading if enabled.
        """
        qty = (self.starting_cash / self.data.iloc[0]['open'])
        if not self.fractional:
            qty = int(qty)
        return qty * self.data.iloc[-1]['close']

    def _pnl_pct(self):
        """Calculates current pnl as pct change"""
        return (self.total_equity - self.starting_cash) / self.starting_cash * 100

    def _pnl_abs(self):
        """Calculates current pnl in cash"""
        return self.total_equity - self.starting_cash

    def _is_terminal(self):
        """
        Checks the environment to see if a terminal state is reached, indicating
        that one of the following conditions are met:
            1. no more trading data
            2. no cash and not holding

        Any of this conditions result in a False status that will indicate
        no further trading should take place in the environment.

        Returns:
            Boolean indicating terminal status
        """
        done = False
        if self.i + 1 >= len(self.data):
            logging.info("Terminal State Reached: self.i + 1 >= len(self.data)")
            done = True
        if self.cash <= self._zero and self.qty <= self._dust:
            logging.info("Terminal State Reached: self.cash <= self._zero AND self.qty <= self._dust")
            done = True
        return done

    def _make_observation(self) -> List[OHLCV]:
        """
        Creates a collection of OHLCV objects representing a slice of the 
        current data in the range [i-window - i+1] such that the current
        period (i) is available with the (window) many previous observations as
        well. 
        Note: requires manual implementation of e.g. *not* exposing the OHLCV
            of the last period to an agent.
        Returns:
            A list of OHLCV objects for each row in the next observation where
            the last row is the most recent observation (rows[-1])
        """
        data = [make_ohlcv_from_df_row(row) for row in
                self.data.iloc[self.i - self.window: self.i + 1].iterrows()]
        return data

    def _sample_price(self) -> float:
        """
        Sample a price from the current periods' range of [low, high]
        to simulate a purchase at an unknown intra-period timestamp.
        """
        _low = self.data.iloc[self.i]['low']
        _high = self.data.iloc[self.i]['high']

        return np.random.uniform(low=_low, high=_high, size=1)[0]

    def _make_report(self, price: float, qty: float, date: datetime, side: int) -> Trade:
        """
        Creates a trade object, saves to local object memory, returns for reference
        """
        report = Trade(
            price=price,
            date=date,
            asset=self.asset,
            qty=qty,
            side=side
        )
        self.trades.append(report)
        return report

    def _buy(self) -> NoReturn:
        """
        Applies a BUY action to the environment.
        """
        # if the account is busted return
        if self.cash <= self._zero:
            return

        # get a sampled price and calculate position size
        # Note: Full allocations are made each time.
        _price = self._sample_price()
        _qty = self.cash / _price
        if not self.fractional:
            _qty = int(_qty)

        # make this a terminal status so this is never reached
        # exception is a reminder
        if _qty == 0 and self.qty == 0:
            print("Cash:", self.cash, "QTY:", self.qty)
            raise Exception(f'TradeEnv is Broke')

        # make the purchase
        self.cash -= (_qty * _price)
        self.qty += _qty

        # make a trade report
        report = self._make_report(
            price=_price,
            qty=_qty,
            date=self.data.index[self.i].to_pydatetime(),
            side=TradeAction.BUY.value  # noqa: it's an int
        )
        logging.debug(f"BUY Action: {report}")

    def _sell(self):
        """Applies a SELL action to the environment"""
        # if not holding, can't sell
        if self.qty <= .01:
            return

        # get randomly
        _price = self._sample_price()

        # make sell
        self.cash += self.qty * _price
        _qty = self.qty
        self.qty = 0

        # make report -- appends to trades
        report = self._make_report(
            price=_price,
            qty=_qty,
            date=self.data.index[self.i].to_pydatetime(),
            side=TradeAction.SELL.value # noqa: it's an int
        )
        logging.debug(f'SELL: {report}')

    def _hold(self):
        """
        For bookkeeping purposes, to track the total number of
        actions. Hold actions don't really do anything that would
        need a record, in the Buy/Sell context.
        """
        _price = self.data.iloc[self.i]['open']

        # make report -- appends to trades
        report = self._make_report(
            price=_price,
            qty=self.qty,
            date=self.data.index[self.i].to_pydatetime(),
            side=TradeAction.HOLD.value # noqa: it's an int
        )
        logging.debug(f'HOLD: {report}')

    def _take_action(self, action: TradeAction):
        """Applies an action to the environment"""
        if action == TradeAction.BUY:
            return self._buy()
        elif action == TradeAction.SELL:
            return self._sell()
        elif action == TradeAction.HOLD:
            return self._hold()

        # shouldn't ever reach this
        else:
            raise Exception(f"Invalid Action: {action}")

    def step(self, action: TradeAction) -> Tuple[List[OHLCV], float, bool, bool, dict, bool]:
        """
        Applies an action to the current environment and returns an observation
        of the resulting state.
        Args:
            action: the action

        Returns:
            Tuple as (next_state, reward, terminality, False, info, done)
        """
        # equity before any trade action is taken
        _trade_initial_equity = self.total_equity
        _bh_initial_equity = self.data.iloc[self.i]['open'] * self.buy_hold_qty

        # take action
        self._take_action(action)

        # bumps the observation period BEFORE calculating rewards as a way
        # to ensure rewards reflect the actions of a period, realized afterwards.
        self.i += 1

        # calculate %-based rewards or absolute rewards based on config.
        if self.pct_based_rewards:

            # get trade pct change
            trade_pct = (self.total_equity - _trade_initial_equity) / _trade_initial_equity * 100

            # get hodl pct change
            _bh_final_equity = self.data.iloc[self.i]['open'] * self.buy_hold_qty
            hodl_pct = (_bh_final_equity - _bh_initial_equity) / _bh_initial_equity * 100

            # reward is difference
            reward = trade_pct - hodl_pct

        else:
            reward = self.total_equity - _trade_initial_equity

        # as next_state, reward, terminality, info
        terminal = self._is_terminal()
        return self._make_observation(), reward, terminal, False, self._get_info(), terminal

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """
        Function to reset the trading environment to the initial state between
        trading episodes. Takes the following actions:
            1. resets the current period based on lookback window value
            2. resets the amount of cash to the starting balance
            3. resets the value of qty to 0
            4. removes all trade records
        Returns:
            A numpy array containing a single observation of the initial state + info dict
        """

        self.i = self.window
        self.cash = self.starting_cash
        self.qty = 0
        self.trades = []

        # reset data-dependent features
        self.buy_hold_qty = self._bh_qty()
        self.buy_hold_abs,\
        self.buy_hold_pct = self._buy_hold_performance_data()

        return self._make_observation(), self._get_info()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Unused"""
        return self._get_info()
