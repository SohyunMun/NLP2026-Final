# LoRA-SFT From Selected Checkpoint

- init checkpoint: `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/dapt_plus_extra/best_3-5e-06-sonnet.pt`
- train data: `/home/msko021220/nlp2026-final-MSK/sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt`
- best epoch: `5`
- best dev chrF: `41.7313`
- dev prediction: `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/selected_lora_plus_extra/predictions/dev_best_chrf.txt`
- test prediction: `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/selected_lora_plus_extra/predictions/test_best_chrf.txt`

| epoch | train loss | dev chrF |
|---:|---:|---:|
| 0 | 4.5032 | 41.0734 |
| 1 | 4.4487 | 41.3603 |
| 2 | 4.4082 | 41.1378 |
| 3 | 4.3750 | 41.4071 |
| 4 | 4.3541 | 41.3289 |
| 5 | 4.3323 | 41.7313 |
| 6 | 4.3140 | 40.9397 |
| 7 | 4.2976 | 41.3305 |
| 8 | 4.2831 | 41.7111 |
| 9 | 4.2670 | 41.3277 |
