from enum import Enum


class TradeAction(Enum):
    """
    Representation of all Trade Actions possible by an Agent.
    """
    BUY: "TradeAction" = 0
    SELL: "TradeAction" = 1
    HOLD: "TradeAction" = 2
