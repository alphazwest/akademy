from unittest import TestCase
from akademy.models.trade_action import TradeAction


class TestTradeAction(TestCase):
    """
    Tests to ensure TradeAction model behaves as expected.
    """
    def setUp(self) -> None:
        """
        Defines static values from which unit-tests can asset validity.
        """
        self.int_to_action = (
            0, TradeAction.BUY
        )
        self.name_to_action = (
            'buy', TradeAction.BUY
        )

    def test_trade_action(self):
        """
        Test basic validation measures
        """
        action = TradeAction.BUY
        self.assertTrue(action is not None)
        self.assertTrue(action.value == 0)

    def test_get_action_from_name(self):
        """
        Test the method to return a TradeAction from a str name works.
        """
        action = TradeAction.get_action_from_name(name=self.name_to_action[0])
        self.assertEqual(
            action,
            self.name_to_action[1]
        )

        with self.assertRaises(Exception):
            action = TradeAction.get_action_from_name("bogus name")

    def test_get_action_from_value(self):
        """
        test that a valid TradeAction is returned given a matching value.
        """
        action = TradeAction.get_action_from_value(value=self.int_to_action[0])
        self.assertEqual(
            action,
            self.int_to_action[1]
        )
        with self.assertRaises(Exception):
            action = TradeAction.get_action_from_value(999999)
