# MSK Form-aware LoRA-DPO Summary

- SFT checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/sixway_ablation/dapt_sft_intermediate/best_10-1e-05-sonnet.pt`
- train data: `/home/msko021220/nlp2026-final-MSK/sonnet_data/strict_497/train_official_131_plus_extra_497_total_628.txt`
- line-rhyme generation: `False`
- best chrF dev chrF: `42.7768`
- best loss dev chrF: `42.2446`
- best chrF epoch: `8`
- best DPO val-loss epoch: `9`
- reject counts: `{'mismatch': 628, 'repetition': 628, 'bad_rhyme': 628, 'bad_line_length': 628, 'repeated_endings': 628, 'short_form': 628}`
- best-loss dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/predictions/dev_best_loss.txt`
- best-loss test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/predictions/test_best_loss.txt`
- best-chrF dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/predictions/dev_best_chrf.txt`
- best-chrF test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/predictions/test_best_chrf.txt`

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.5235 | 0.4246 | 41.1217 |
| 1 | 0.4382 | 0.3777 | 41.7812 |
| 2 | 0.4119 | 0.3555 | 42.3095 |
| 3 | 0.3840 | 0.3471 | 42.4900 |
| 4 | 0.3644 | 0.3407 | 42.4932 |
| 5 | 0.3685 | 0.3404 | 42.3892 |
| 6 | 0.3471 | 0.3219 | 42.3961 |
| 7 | 0.3385 | 0.3192 | 41.9988 |
| 8 | 0.3384 | 0.3183 | 42.7768 |
| 9 | 0.3321 | 0.3119 | 42.2446 |
