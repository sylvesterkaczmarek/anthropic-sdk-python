from __future__ import annotations

from typing import Any, cast

import pytest

from anthropic._exceptions import AnthropicError
from anthropic.lib.google_cloud import AnthropicGoogleCloud, AsyncAnthropicGoogleCloud


def _sync_client(value: object) -> AnthropicGoogleCloud:
    return AnthropicGoogleCloud(
        base_url="https://example.test/",
        workspace_id="wrkspc_x",
        token_provider=cast(Any, lambda: value),
    )


@pytest.mark.parametrize("value", ["", None], ids=["empty", "non-string"])
def test_sync_token_provider_rejects_invalid_result(value: object) -> None:
    client = _sync_client(value)

    with pytest.raises(AnthropicError, match="token_provider.*non-empty string"):
        client._get_token()


@pytest.mark.parametrize("value", ["", None], ids=["empty", "non-string"])
async def test_async_token_provider_rejects_invalid_async_result(value: object) -> None:
    async def provider() -> object:
        return value

    client = AsyncAnthropicGoogleCloud(
        base_url="https://example.test/",
        workspace_id="wrkspc_x",
        token_provider=cast(Any, provider),
    )

    with pytest.raises(AnthropicError, match="token_provider.*non-empty string"):
        await client._get_token()


async def test_async_client_rejects_invalid_sync_provider_result() -> None:
    client = AsyncAnthropicGoogleCloud(
        base_url="https://example.test/",
        workspace_id="wrkspc_x",
        token_provider=cast(Any, lambda: ""),
    )

    with pytest.raises(AnthropicError, match="token_provider.*non-empty string"):
        await client._get_token()
