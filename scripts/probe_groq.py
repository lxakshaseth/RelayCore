"""Probe script to investigate Groq API response headers, latency, and free tier limits."""

import asyncio
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from groq import AsyncGroq


async def probe_groq(model: str):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.")
        return

    print(f"\n==================================================")
    print(f"Testing model: {model}")
    print(f"==================================================")
    client = AsyncGroq(api_key=api_key, base_url="https://api.groq.com")

    # 1. Non-streaming probe with raw response headers
    print(f"--- [1] Probing non-streaming response for {model} ---")
    start = time.perf_counter()
    try:
        raw_res = await client.chat.completions.with_raw_response.create(
            messages=[{"role": "user", "content": "Respond with one word: 'READY'."}],
            model=model,
            max_tokens=10,
        )
        total_time = (time.perf_counter() - start) * 1000
        headers = raw_res.headers
        completion = await raw_res.parse()

        print(f"Status: OK | Total Time: {total_time:.2f} ms")
        print(f"Content: {completion.choices[0].message.content}")
        print(f"Usage: prompt_tokens={completion.usage.prompt_tokens}, completion_tokens={completion.usage.completion_tokens}")
        print("Rate Limit Headers:")
        for k, v in headers.items():
            if "ratelimit" in k.lower() or "retry-after" in k.lower():
                print(f"  {k}: {v}")

    except Exception as e:
        print(f"Non-streaming probe failed: {type(e).__name__}: {e}")

    # 2. Streaming probe to measure Time To First Token (TTFT)
    print(f"\n--- [2] Probing streaming response for {model} ---")
    start = time.perf_counter()
    ttft = None
    chunks_received = 0
    collected_text = []

    try:
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain self-attention in exactly two sentences."}],
            model=model,
            stream=True,
            max_tokens=80,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                chunks_received += 1
                collected_text.append(delta)

        total_stream_time = (time.perf_counter() - start) * 1000
        print(f"Streaming Complete | TTFT: {ttft:.2f} ms | Total Time: {total_stream_time:.2f} ms | Chunks: {chunks_received}")
        print(f"Generated text: {''.join(collected_text)}")

    except Exception as e:
        print(f"Streaming probe failed: {type(e).__name__}: {e}")


async def main():
    models_to_test = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]
    for m in models_to_test:
        await probe_groq(m)


if __name__ == "__main__":
    asyncio.run(main())
