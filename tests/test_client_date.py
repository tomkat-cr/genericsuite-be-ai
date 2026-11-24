"""
Tests for the optional "client_date" (user's LOCAL date) chatbot parameter.
No LLM, DB or network is used.
"""
from datetime import date, datetime, timezone

import pytest

from genericsuite.util.app_context import AppContext

from genericsuite_ai.lib import ai_gpt_fn_conversations as fn_conv
from genericsuite_ai.lib.ai_chatbot_commons import set_client_date
from genericsuite_ai.lib.ai_utilities import get_valid_client_date

TODAY_UTC = date(2026, 3, 16)
FIXED_NOW = datetime(2026, 3, 16, 2, 0, 5)


class FakeDatetime(datetime):
    """datetime with a fixed utcnow()"""

    @classmethod
    def utcnow(cls):
        return FIXED_NOW


@pytest.fixture(name="fixed_clock")
def fixture_fixed_clock(monkeypatch):
    monkeypatch.setattr(fn_conv, "datetime", FakeDatetime)


def make_context(client_date=None):
    ctx = AppContext()
    if client_date is not None:
        ctx.set_other_data("client_date", client_date)
    return ctx


# get_valid_client_date

@pytest.mark.parametrize("value", [
    "2026-03-15",  # 1 day before UTC date (UTC-5 user at 21:00)
    "2026-03-16",  # same day
    "2026-03-17",  # 1 day after UTC date (UTC+9 user)
])
def test_valid_client_date(value):
    assert get_valid_client_date(value, TODAY_UTC) == value


@pytest.mark.parametrize("value", [None, ""])
def test_missing_client_date(value):
    assert get_valid_client_date(value, TODAY_UTC) is None


@pytest.mark.parametrize("value", [
    "2026-3-16", "20260316", "2026-03-16T00:00:00", " 2026-03-16",
    "2026-03-16\n", "2026-02-30", "2026-13-01", "abcd-ef-gh",
    "2026/03/16", "٢٠٢٦-٠٣-١٦", 20260316, ["2026-03-16"], {"a": 1},
])
def test_malformed_client_date(value):
    assert get_valid_client_date(value, TODAY_UTC) is None


@pytest.mark.parametrize("value", [
    "2026-03-14", "2026-03-18", "1999-01-01", "2099-12-31",
])
def test_out_of_range_client_date(value):
    assert get_valid_client_date(value, TODAY_UTC) is None


def test_default_today_is_real_utc_date():
    today = datetime.now(timezone.utc).date().isoformat()
    assert get_valid_client_date(today) == today
    assert get_valid_client_date("1999-01-01") is None


# set_client_date (request -> app context)

def test_set_client_date_valid():
    ctx = make_context()
    today = datetime.now(timezone.utc).date().isoformat()
    set_client_date(ctx, {"client_date": today})
    assert ctx.get_other_data("client_date") == today


@pytest.mark.parametrize("params", [
    {}, {"client_date": "bad"}, {"client_date": "1999-01-01"},
    {"client_date": None},
])
def test_set_client_date_ignored(params):
    ctx = make_context()
    set_client_date(ctx, params)
    assert not ctx.get_other_data("client_date")


# get_current_date_time_func

def test_tool_without_client_date_is_unchanged(fixed_clock):
    fn_conv.cac.set(make_context())
    assert fn_conv.get_current_date_time_func({}) == \
        "today is 2026-03-16, and the current UTC time is 02:00:05"


def test_tool_with_no_context_is_unchanged(fixed_clock):
    fn_conv.cac.set(None)
    assert fn_conv.get_current_date_time_func({}) == \
        "today is 2026-03-16, and the current UTC time is 02:00:05"


def test_tool_with_client_date(fixed_clock):
    fn_conv.cac.set(make_context("2026-03-15"))
    result = fn_conv.get_current_date_time_func({})
    assert "2026-03-15" in result
    assert "local date" in result
    assert "2026-03-16" not in result
    assert "current UTC time is 02:00:05" in result


def test_tool_no_leak_between_contexts(fixed_clock):
    ctx_a = make_context("2026-03-15")
    ctx_b = make_context()
    fn_conv.cac.set(ctx_a)
    result_a = fn_conv.get_current_date_time_func({})
    fn_conv.cac.set(ctx_b)
    result_b = fn_conv.get_current_date_time_func({})
    assert "2026-03-15" in result_a
    assert result_b == \
        "today is 2026-03-16, and the current UTC time is 02:00:05"
    assert not ctx_b.get_other_data("client_date")
    fn_conv.cac.set(ctx_a)
    assert "2026-03-15" in fn_conv.get_current_date_time_func({})


def test_explicit_context_wins_over_shared_holder(fixed_clock):
    fn_conv.cac.set(make_context("2026-03-17"))
    ctx = make_context("2026-03-15")
    result = fn_conv.get_current_date_time_text(ctx)
    assert "2026-03-15" in result
    assert "2026-03-17" not in result
