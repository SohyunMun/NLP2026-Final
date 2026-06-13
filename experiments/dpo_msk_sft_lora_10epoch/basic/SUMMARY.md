# MSK SFT LoRA-DPO Summary

- SFT checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/trial10_best_loss_4gpu/basic__msk_sft_weighted_mbr/best_10-1e-05-sonnet.pt`
- train data: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic/sonnets_train.txt`
- best dev chrF: `40.7847`
- best chrF epoch: `4`
- best DPO val-loss epoch: `9`
- dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic/predictions/dev_best_chrf.txt`
- test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic/predictions/test_best_chrf.txt`

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.5454 | 0.1120 | 40.4130 |
| 1 | 0.1121 | 0.0019 | 39.4445 |
| 2 | 0.0132 | 0.0004 | 40.7480 |
| 3 | 0.0077 | 0.0001 | 40.7611 |
| 4 | 0.0034 | 0.0000 | 40.7847 |
| 5 | 0.0006 | 0.0000 | 39.7451 |
| 6 | 0.0152 | 0.0000 | 40.0308 |
| 7 | 0.0022 | 0.0000 | 40.0094 |
| 8 | 0.0003 | 0.0000 | 40.1958 |
| 9 | 0.0001 | 0.0000 | 39.5147 |
