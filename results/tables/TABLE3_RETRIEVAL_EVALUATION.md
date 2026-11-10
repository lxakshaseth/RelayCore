# Table 3: Systematic Retrieval Configuration Evaluation

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct & cited % | Wrong % | Correctly refused % | Incorrectly refused % | Citation correctness % | Tokens per answer | Cost per 1,000 | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Setting A: Small chunks (150w), low top-k (k=2), no re-ranking | 18 | 16.7% | 0.0% | 22.2% | 61.1% | 38.9% | 277.8 | $0.0139 | 664.0 / 1043.5 | 2.5 / 905.0 |
| Setting B: Medium chunks (350w), moderate top-k (k=4), no re-ranking | 18 | 33.3% | 0.0% | 22.2% | 44.4% | 55.6% | 811.1 | $0.0406 | 780.5 / 11244.5 | 2.0 / 11464.6 |
| Setting C: Larger chunks (800w), moderate top-k (k=3), with re-ranking | 18 | 33.3% | 11.1% | 22.2% | 33.3% | 55.6% | 539.4 | $0.0270 | 6039.0 / 10675.8 | 3.5 / 15005.9 |

## Per-Question Detailed Breakdown

### Setting A: Small chunks (150w), low top-k (k=2), no re-ranking

| ID | Category | Question | Grade | Citations Valid? | Total ms |
| --- | --- | --- | --- | --- | --- |
| eval_01 | direct_answer | What is the dimension of the embedding layers... | `correct` | Yes | 1727ms |
| eval_02 | direct_answer | At what frame rate does the Mimi neural audio... | `refused_wrongly` | No | 2ms |
| eval_03 | direct_answer | What baseline fixed-size chunking architectur... | `refused_wrongly` | No | 625ms |
| eval_04 | multiple_passages | Compare how positional information is encoded... | `refused_wrongly` | No | 3ms |
| eval_05 | multiple_passages | How do the key, value, and query projections ... | `refused_wrongly` | No | 3ms |
| eval_06 | multiple_passages | How does Moshi synchronize parallel audio str... | `correct` | Yes | 760ms |
| eval_07 | technical_details | What are the exact optimizer settings (beta1,... | `refused_wrongly` | No | 2ms |
| eval_08 | technical_details | What is the sampling rate, number of RVQ code... | `refused_wrongly` | No | 709ms |
| eval_09 | technical_details | What is the scaling factor applied to dot pro... | `correct` | Yes | 472ms |
| eval_10 | technical_details | What is the theoretical minimum latency break... | `refused_wrongly` | No | 1ms |
| eval_11 | unanswerable_out_of_corpus | What was the total dollar electricity cost in... | `refused_correctly` | Yes | 1ms |
| eval_12 | unanswerable_out_of_corpus | How does the Transformer model perform on the... | `refused_correctly` | Yes | 2ms |
| eval_13 | unanswerable_out_of_corpus | What is the recommended oven baking temperatu... | `refused_correctly` | Yes | 1ms |
| eval_14 | unanswerable_out_of_corpus | What is the recommended tire pressure in PSI ... | `refused_correctly` | Yes | 1ms |
| eval_15 | retrieval_failure_trap | How does the multi-head attention mechanism c... | `refused_wrongly` | No | 2ms |
| eval_16 | retrieval_failure_trap | What is the acoustic latency between the spea... | `refused_wrongly` | No | 1ms |
| eval_17 | retrieval_failure_trap | How many layers are in the feed-forward netwo... | `refused_wrongly` | No | 591ms |
| eval_18 | retrieval_failure_trap | How does the U-Net downsampling layer in Dyna... | `refused_wrongly` | No | 3ms |

### Setting B: Medium chunks (350w), moderate top-k (k=4), no re-ranking

| ID | Category | Question | Grade | Citations Valid? | Total ms |
| --- | --- | --- | --- | --- | --- |
| eval_01 | direct_answer | What is the dimension of the embedding layers... | `correct` | Yes | 1249ms |
| eval_02 | direct_answer | At what frame rate does the Mimi neural audio... | `correct` | Yes | 472ms |
| eval_03 | direct_answer | What baseline fixed-size chunking architectur... | `refused_wrongly` | No | 772ms |
| eval_04 | multiple_passages | Compare how positional information is encoded... | `refused_wrongly` | No | 2ms |
| eval_05 | multiple_passages | How do the key, value, and query projections ... | `refused_wrongly` | No | 2ms |
| eval_06 | multiple_passages | How does Moshi synchronize parallel audio str... | `correct` | Yes | 852ms |
| eval_07 | technical_details | What are the exact optimizer settings (beta1,... | `refused_wrongly` | No | 1ms |
| eval_08 | technical_details | What is the sampling rate, number of RVQ code... | `correct` | Yes | 794ms |
| eval_09 | technical_details | What is the scaling factor applied to dot pro... | `correct` | Yes | 11950ms |
| eval_10 | technical_details | What is the theoretical minimum latency break... | `correct` | Yes | 11379ms |
| eval_11 | unanswerable_out_of_corpus | What was the total dollar electricity cost in... | `refused_correctly` | Yes | 2ms |
| eval_12 | unanswerable_out_of_corpus | How does the Transformer model perform on the... | `refused_correctly` | Yes | 2ms |
| eval_13 | unanswerable_out_of_corpus | What is the recommended oven baking temperatu... | `refused_correctly` | Yes | 1ms |
| eval_14 | unanswerable_out_of_corpus | What is the recommended tire pressure in PSI ... | `refused_correctly` | Yes | 1ms |
| eval_15 | retrieval_failure_trap | How does the multi-head attention mechanism c... | `refused_wrongly` | No | 2ms |
| eval_16 | retrieval_failure_trap | What is the acoustic latency between the spea... | `refused_wrongly` | No | 1ms |
| eval_17 | retrieval_failure_trap | How many layers are in the feed-forward netwo... | `refused_wrongly` | No | 672ms |
| eval_18 | retrieval_failure_trap | How does the U-Net downsampling layer in Dyna... | `refused_wrongly` | No | 1ms |

### Setting C: Larger chunks (800w), moderate top-k (k=3), with re-ranking

| ID | Category | Question | Grade | Citations Valid? | Total ms |
| --- | --- | --- | --- | --- | --- |
| eval_01 | direct_answer | What is the dimension of the embedding layers... | `correct` | Yes | 7024ms |
| eval_02 | direct_answer | At what frame rate does the Mimi neural audio... | `correct` | Yes | 5600ms |
| eval_03 | direct_answer | What baseline fixed-size chunking architectur... | `correct` | Yes | 10920ms |
| eval_04 | multiple_passages | Compare how positional information is encoded... | `refused_wrongly` | No | 4ms |
| eval_05 | multiple_passages | How do the key, value, and query projections ... | `refused_wrongly` | No | 2ms |
| eval_06 | multiple_passages | How does Moshi synchronize parallel audio str... | `correct` | Yes | 9962ms |
| eval_07 | technical_details | What are the exact optimizer settings (beta1,... | `refused_wrongly` | No | 1ms |
| eval_08 | technical_details | What is the sampling rate, number of RVQ code... | `wrong` | No | 15005ms |
| eval_09 | technical_details | What is the scaling factor applied to dot pro... | `correct` | Yes | 4723ms |
| eval_10 | technical_details | What is the theoretical minimum latency break... | `wrong` | No | 15011ms |
| eval_11 | unanswerable_out_of_corpus | What was the total dollar electricity cost in... | `refused_correctly` | Yes | 2ms |
| eval_12 | unanswerable_out_of_corpus | How does the Transformer model perform on the... | `refused_correctly` | Yes | 1ms |
| eval_13 | unanswerable_out_of_corpus | What is the recommended oven baking temperatu... | `refused_correctly` | Yes | 1ms |
| eval_14 | unanswerable_out_of_corpus | What is the recommended tire pressure in PSI ... | `refused_correctly` | Yes | 1ms |
| eval_15 | retrieval_failure_trap | How does the multi-head attention mechanism c... | `refused_wrongly` | No | 2ms |
| eval_16 | retrieval_failure_trap | What is the acoustic latency between the spea... | `refused_wrongly` | No | 1ms |
| eval_17 | retrieval_failure_trap | How many layers are in the feed-forward netwo... | `correct` | Yes | 496ms |
| eval_18 | retrieval_failure_trap | How does the U-Net downsampling layer in Dyna... | `refused_wrongly` | No | 3ms |

