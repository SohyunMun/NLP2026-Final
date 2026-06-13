# MSK Form-aware LoRA-DPO Summary

- SFT checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/trial10_best_loss_4gpu/basic_plus_extra__msk_sft_weighted_mbr/best_10-1e-05-sonnet.pt`
- train data: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic_plus_extra/sonnets_train_plus_extra_clean.txt`
- best dev chrF: `42.3309`
- best chrF epoch: `1`
- best DPO val-loss epoch: `2`
- reject counts: `{'mismatch': 650, 'repetition': 650, 'bad_rhyme': 650, 'bad_line_length': 650, 'repeated_endings': 650}`
- dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_form_rhyme_rerank_10epoch/basic_plus_extra/predictions/dev_best_chrf.txt`
- test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_form_rhyme_rerank_10epoch/basic_plus_extra/predictions/test_best_chrf.txt`

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.3647 | 0.2441 | 41.9654 |
| 1 | 0.2545 | 0.1907 | 42.3309 |
| 2 | 0.2142 | 0.1849 | 41.2569 |
| 3 | 0.1999 | 0.1852 | 41.0974 |
| 4 | 0.2057 | 0.1939 | 41.2789 |
| 5 | 0.1856 | 0.1860 | 41.9258 |
| 6 | 0.1910 | 0.1942 | 41.3042 |
| 7 | 0.1860 | 0.1883 | 41.5271 |
| 8 | 0.1798 | 0.1987 | 41.2374 |
| 9 | 0.1696 | 0.2081 | 41.7028 |
