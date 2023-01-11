from datetime import datetime
import json
from typing import NoReturn

from akademy.models.trade_action import TradeAction


class Trade:
    """
    An easily serializable report of trades made during an episode.
    """
    def __init__(
            self, price: float, date: datetime,
            asset: str, qty: float, side: int
    ) -> NoReturn:
        self.price = price
        self.date = date
        self.asset = asset
        self.qty = qty
        self.side = side

    def to_dict(self) -> dict:
        """
        Converts the Trade object to a dictionary representation where the
        only conversion is the <side> field such that the TradeAction object
        is converted to an explicit str value of either "BUY" or "SELL" for
        clarity.
        """
        return {
            "price": self.price,
            "date": str(self.date.isoformat()),
            "asset": self.asset,
            "qty": self.qty,
            "side": "BUY" if self.side == TradeAction.BUY.value else "SELL"
        }

    def to_json(self) -> str:
        """
        Converts the Trade object into a JSON-formatted str representation such
        that all numerical values are strings and the TradeAction objects in the
        <side> field are converted to "BUY" or "SELL" for clarity.
        """
        return json.dumps({
            "price": str(float(self.price)),
            "date": str(self.date.isoformat()),
            "asset": self.asset,
            "qty": str(float(self.qty)),
            "side": "BUY" if self.side == TradeAction.BUY.value else "SELL"
        })

    def __str__(self) -> str:
        """
        Wraps the <to_json> method for object str representation.
        """
        return self.to_json()
