from enum import Enum


class TradeAction(Enum):
    """
    Representation of all Trade Actions possible by an Agent.
    """
    BUY: "TradeAction" = 0
    SELL: "TradeAction" = 1
    HOLD: "TradeAction" = 2

    @staticmethod
    def get_action_from_name(name: str) -> "TradeAction":
        """
        Given a string name of an action e.g. Buy, Sell, Hold, return the
        associated trade action.
        """
        if name.lower() == "buy":
            return TradeAction.BUY
        if name.lower() == "sell":
            return TradeAction.SELL
        if name.lower() == "hold":
            return TradeAction.HOLD
        raise Exception(
            f"No TradeAction found for: {name}."
            f"Must be lowercase form of the following: "
            f"{[a.name for a in TradeAction]}")

    @staticmethod
    def get_action_from_value(value: int) -> "TradeAction":
        """
        Given an integer value, map to the associated TradeAction
        """
        for action in TradeAction:
            if action.value == value:
                return action
        raise Exception(
            f"No TradeAction associated with the value: {value}."
            f"Available choices: {[a.value for a in TradeAction]}")
