"""End-to-end test script for GroqAsyncClient."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.models.groq_client import GroqAsyncClient
from app.models.schemas import Message, ModelRequest


async def main():
    client = GroqAsyncClient()
    print("Testing GroqAsyncClient non-streaming completion...")

    # Test 1: Non-streaming completion
    req = ModelRequest(
        messages=[Message(role="user", content="What are the three main components of the Transformer architecture? Be brief.")],
        max_tokens=60,
    )
    res = await client.complete(req)
    print(f"\n[Test 1 Non-Streaming]")
    print(f"Status: {res.status}")
    print(f"Model: {res.model}")
    print(f"Total Latency: {res.total_ms} ms")
    print(f"Tokens In: {res.tokens_in} | Tokens Out: {res.tokens_out}")
    print(f"Content: {res.content}")
    if res.rate_limit_info:
        print(f"RateLimitInfo: Limit={res.rate_limit_info.limit_requests}, Remaining={res.rate_limit_info.remaining_requests}, TokensLimit={res.rate_limit_info.limit_tokens}")

    # Test 2: Streaming completion with TTFT measurement
    print(f"\n[Test 2 Streaming with TTFT]")
    stream_req = ModelRequest(
        messages=[Message(role="user", content="Name 3 applications of speech-to-text models.")],
        max_tokens=50,
        stream=True,
    )
    tokens_streamed = []
    final_response = None
    async for chunk, response in client.stream_complete(stream_req):
        if chunk:
            tokens_streamed.append(chunk)
        if response:
            final_response = response

    print(f"Status: {final_response.status}")
    print(f"TTFT: {final_response.first_token_ms} ms")
    print(f"Total Time: {final_response.total_ms} ms")
    print(f"Streamed Text: {''.join(tokens_streamed)}")

    # Test 3: Timeout handling
    print(f"\n[Test 3 Timeout Handling (Simulated 0.001s timeout)]")
    timeout_req = ModelRequest(
        messages=[Message(role="user", content="Explain quantum physics.")],
        timeout_seconds=0.001,
    )
    timeout_res = await client.complete(timeout_req)
    print(f"Status: {timeout_res.status}")
    print(f"Error Info: {timeout_res.error_info}")
    assert timeout_res.status == "timeout"


if __name__ == "__main__":
    asyncio.run(main())
