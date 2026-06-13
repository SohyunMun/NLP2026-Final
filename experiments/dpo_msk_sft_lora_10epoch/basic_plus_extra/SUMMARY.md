# MSK SFT LoRA-DPO Summary

- SFT checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/trial10_best_loss_4gpu/basic_plus_extra__msk_sft_weighted_mbr/best_10-1e-05-sonnet.pt`
- train data: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic_plus_extra/sonnets_train_plus_extra_clean.txt`
- best dev chrF: `42.5315`
- best chrF epoch: `3`
- best DPO val-loss epoch: `3`
- dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic_plus_extra/predictions/dev_best_chrf.txt`
- test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic_plus_extra/predictions/test_best_chrf.txt`

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.6665 | 0.4481 | 41.7785 |
| 1 | 0.5379 | 0.3849 | 41.3283 |
| 2 | 0.4885 | 0.3638 | 42.1839 |
| 3 | 0.4455 | 0.3286 | 42.5315 |
| 4 | 0.4368 | 0.3487 | 41.5873 |
| 5 | 0.4303 | 0.3436 | 41.4015 |
| 6 | 0.4371 | 0.3342 | 41.3410 |
| 7 | 0.4063 | 0.3322 | 41.4647 |
| 8 | 0.4034 | 0.3386 | 41.2304 |
| 9 | 0.4104 | 0.3324 | 42.1655 |
