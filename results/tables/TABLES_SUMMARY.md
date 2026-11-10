# The Relay Assignment: Compiled Results Tables

## Table 2a: Latency, one row per load scenario

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| steady_real | real | 5 | 3.3 | 91.5 | 639.0 / 989.8 / 1024.4 | 641.0 / 1672.2 / 1842.4 |
| steady | stand-in | 15 | 34.5 | 26.1 | 178.0 / 192.9 / 197.0 | 4794.0 / 12013.0 / 13497.0 |
| burst | stand-in | 25 | 1.1 | 1306.7 | 224.5 / 287.8 / 302.4 | 10142.0 / 11514.0 / 12313.4 |
| sustained_overload | stand-in | 40 | 27.1 | 88.6 | 229.0 / 298.2 / 306.0 | 5865.5 / 7141.1 / 7143.8 |
| limit_halved | stand-in | 20 | 31.2 | 38.4 | 226.0 / 243.5 / 244.7 | 10139.5 / 11219.3 / 13686.3 |
| daily_budget_exhausted | stand-in | 15 | 1.0 | 900.0 | not measured / not measured / not measured | 1.0 / 1.0 / 1.0 |
| retrieval_sweep_small | stand-in | 10 | 1.6 | 373.5 | 201.0 / 257.0 / 262.6 | 418.0 / 11610.8 / 12549.4 |
| retrieval_sweep_large | stand-in | 10 | 34.0 | 17.6 | not measured / not measured / not measured | 11176.0 / 14104.2 / 14105.6 |

## Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| steady_real | 100.0 | 0.0 | 0.0 | 0.0 | 1 | 906.0 | $0.0453 |
| steady | 53.3 | 46.7 | 0.0 | 0.47 | 3 | 514.8 | $0.0257 |
| burst | 44.0 | 56.0 | 0.0 | 0.12 | 3 | 327.4 | $0.0164 |
| sustained_overload | 35.0 | 65.0 | 0.0 | 0.1 | 2 | 203.7 | $0.0102 |
| limit_halved | 35.0 | 65.0 | 0.0 | 0.25 | 3 | 173.4 | $0.0087 |
| daily_budget_exhausted | 0.0 | 100.0 | 0.0 | 0.0 | 1 | 0.0 | $0.0000 |
| retrieval_sweep_small | 80.0 | 20.0 | 0.0 | 0.1 | 2 | 809.7 | $0.0405 |
| retrieval_sweep_large | 0.0 | 100.0 | 0.0 | 1.2 | 3 | 0.0 | $0.0000 |

## Table 3: Retrieval settings

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| medium_k3 | 120 | 39.2 | 0.0 | 22.5 | 267.1 | 217.5 / 719.1 | 3805.5 / 10218.1 |
| small_k6 | 10 | 90.0 | 0.0 | 0.0 | 809.7 | 201.0 / 257.0 | 418.0 / 11610.8 |
| large_k2 | 10 | 70.0 | 0.0 | 0.0 | 0.0 | not measured / not measured | 11176.0 / 14104.2 |
