from __future__ import annotations

from typing import Any, cast

import pytest

from anthropic.lib.tools._beta_runner import BaseToolRunner
from anthropic.types.beta.beta_message_param import BetaMessageParam


def _runner() -> BaseToolRunner[Any, Any]:
    return BaseToolRunner(
        params=cast(
            Any,
            {
                "model": "test-model",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "before"}],
            },
        ),
        options={},
        tools=[],
    )


@pytest.mark.parametrize("use_callable", [False, True], ids=["replacement", "callable"])
def test_set_messages_params_invalidates_cached_tool_response(use_callable: bool) -> None:
    runner = _runner()
    cached: BetaMessageParam = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_old",
                "content": "stale result",
            }
        ],
    }
    runner._cached_tool_call_response = cached

    if use_callable:
        runner.set_messages_params(
            lambda params: {
                **params,
                "messages": [{"role": "user", "content": "after"}],
            }
        )
    else:
        runner.set_messages_params(
            cast(
                Any,
                {
                    **runner._params,
                    "messages": [{"role": "user", "content": "after"}],
                },
            )
        )

    assert runner._cached_tool_call_response is None
