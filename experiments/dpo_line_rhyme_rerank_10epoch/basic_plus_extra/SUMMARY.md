# MSK Line-Rhyme LoRA-DPO Summary

- best chrF epoch: `0` / dev chrF `41.8622`
- best loss epoch: `5` / val loss `0.1771` / dev chrF `41.2196`
- generation: `6` candidates, first `3` candidates use line-by-line rhyme-biased decoding

## Epoch History

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.4007 | 0.2420 | 41.8622 |
| 1 | 0.2718 | 0.2075 | 41.3040 |
| 2 | 0.2361 | 0.1937 | 41.6740 |
| 3 | 0.2237 | 0.1929 | 41.4382 |
| 4 | 0.2066 | 0.1854 | 40.9378 |
| 5 | 0.1851 | 0.1771 | 41.2196 |
| 6 | 0.1881 | 0.1829 | 41.0901 |
| 7 | 0.1816 | 0.1908 | 41.1846 |
| 8 | 0.1790 | 0.1839 | 41.3934 |
| 9 | 0.1692 | 0.1897 | 41.2591 |

## Main Comparison

| method | chrF | avg-rank | BLEU | ROUGE-L | token-F1 | form/5 | rhyme | MATTR | distinct-2 | repetition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSK line-rhyme LoRA-DPO (best loss epoch 5) | 41.2196 | 2.30 | 23.37 | 0.3069 | 0.3903 | 4.527 | 0.8452 | 0.9206 | 0.9342 | 0.5830 |
| Previous MSK form/rhyme-aware LoRA-DPO | 42.3309 | 2.90 | 23.83 | 0.2885 | 0.3722 | 3.376 | 0.2500 | 0.9570 | 0.9467 | 0.5884 |
| MSK line-rhyme LoRA-DPO (best chrF epoch 0) | 41.8622 | 3.10 | 23.31 | 0.2903 | 0.3713 | 4.388 | 0.7619 | 0.9332 | 0.9457 | 0.5947 |
| Previous MSK SFT-initialized LoRA-DPO | 42.5315 | 3.30 | 24.35 | 0.3152 | 0.4014 | 3.111 | 0.1429 | 0.8966 | 0.9311 | 0.5860 |
| HUJ LoRA + DPO | 40.0744 | 3.40 | 22.15 | 0.2951 | 0.3471 | 4.408 | 0.8095 | 0.8601 | 0.9715 | 0.4648 |

Lower avg-rank is better across non-chrF metrics.
