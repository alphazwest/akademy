import json
from typing import NoReturn


class OHLCV:
    """
    Object model for a single period's pricing history as
        open, low, high, close, volume and date
    Note:
        Expects all values to be numeric. Convert if using raw CSV data.
        <date> object is expected to be a Unix-format timestamp.
    """
    def __init__(
            self, date: int, open: float, high: float,
            low: float, close: float, volume: int
    ) -> NoReturn:
        self.date = date
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = int(volume)

    def to_dict(self) -> dict:
        """
        Dictionary representation of the object.
        """
        return {
            'date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }

    def to_json(self) -> str:
        """
        JSON-formatted string representation of the object.
        """
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        """
        Display as JSON-formatted str.
        """
        return self.to_json()
