"""#80 — the sub-minute gap, measured (1b.1 baseline for the LTF feature).

Pynecore's Pine-side timeframe grammar ACCEPTS seconds ("15S" -> 15s); the
DNSE provider REJECTS them because the venue's REST OHLC floor is 1 minute
(documented resolutions 1/3/5/15/30/1H/1D/1W). These pins measure today's
boundary; the second one flips when #80 lands sub-minute support and then
pins whatever surface provides it.
"""
import pytest

from pynecore.lib import timeframe as tf_lib


def __test_pynecore_grammar_accepts_seconds_timeframes__():
    assert tf_lib.in_seconds("15S") == 15
    assert tf_lib.in_seconds("5S") == 5


def __test_dnse_provider_has_no_sub_minute_resolution_today__():
    """The measured gap (#80): flips to a support-pin when LTF lands."""
    from pynecore_dnse.provider import DNSEProvider
    with pytest.raises(ValueError, match="no resolution"):
        DNSEProvider.to_exchange_timeframe("15S")
