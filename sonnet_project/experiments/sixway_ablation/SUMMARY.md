# Main Sonnet Experiment Summary

## Data

- basic train: `/home/msko021220/nlp2026-final-MSK/sonnet_project/data/basic/train_131.txt`
- plus-extra train: `/home/msko021220/nlp2026-final-MSK/sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt`
- dev prompts/gold: `/home/msko021220/nlp2026-final-MSK/sonnet_project/data/strict_497/dev_prompts_12.txt`, `/home/msko021220/nlp2026-final-MSK/sonnet_project/data/strict_497/dev_gold_12.txt`
- test prompts: `/home/msko021220/nlp2026-final-MSK/sonnet_project/data/strict_497/test_prompts_12.txt`
- test gold is unavailable; test chrF is intentionally blank.

## Run Definitions

| run | definition | checkpoint |
|---|---|---|
| `base_basic` | plain GPT-2 full fine-tuning on official 131 train | `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/base_basic/best_10-2e-05-sonnet.pt` |
| `base_plus_extra` | plain GPT-2 full fine-tuning on official 131 + strict extra 497 | `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/base_plus_extra/best_10-2e-05-sonnet.pt` |
| `sft_plus_extra` | prompt-focused SFT on official 131 + strict extra 497 | `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/sft_plus_extra/best_10-1e-05-sonnet.pt` |
| `dapt_plus_extra` | DAPT-only checkpoint on official 131 + strict extra 497, evaluated directly | `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/dapt_plus_extra/best_3-5e-06-sonnet.pt` |
| `selected_lora_plus_extra` | LoRA-SFT from better non-DPO checkpoint: dapt_plus_extra | `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/selected_lora_plus_extra/best_chrf_lora_sft.pt` |
| `dapt_sft_lora_dpo_best_chrf` | DAPT -> SFT -> form/rhyme LoRA-DPO, best dev chrF checkpoint | `/home/msko021220/nlp2026-final-MSK/sonnet_project/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/best_chrf_lora_dpo_form_rhyme.pt` |
| `poemetric_reranking` | reference-free POEMetric reranking on top of the best DPO checkpoint | `sonnet_project/experiments/dpo_reranking/predictions/` |

## Dev Evaluation

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_basic | 41.8252 | 0.0000 | 0.5365 | 0.9513 | 0.5567 | 0.2267 | 0.5998 |
| base_plus_extra | 41.0941 | 0.0000 | 0.5392 | 0.9542 | 0.5726 | 0.2632 | 0.6116 |
| sft_plus_extra | 40.4982 | 0.0000 | 0.5358 | 0.9481 | 0.5664 | 0.3231 | 0.6161 |
| dapt_plus_extra | 41.1442 | 0.0000 | 0.5510 | 0.9332 | 0.5495 | 0.2450 | 0.6002 |
| selected_lora_plus_extra | 41.7313 | 0.0000 | 0.5455 | 0.9422 | 0.5706 | 0.2785 | 0.6122 |
| dapt_sft_lora_dpo_best_chrf | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |
| poemetric_reranking | 42.0672 | 0.1667 | 0.5804 | 0.9610 | 0.5731 | 0.3307 | 0.6359 |

## Test Evaluation

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_basic |  | 0.0000 | 0.5524 | 0.9550 | 0.5736 | 0.1880 | 0.6048 |
| base_plus_extra |  | 0.0833 | 0.5512 | 0.9562 | 0.5988 | 0.1422 | 0.6054 |
| sft_plus_extra |  | 0.0000 | 0.5381 | 0.9537 | 0.5737 | 0.3073 | 0.6181 |
| dapt_plus_extra |  | 0.0000 | 0.5431 | 0.9429 | 0.5592 | 0.1927 | 0.5953 |
| selected_lora_plus_extra |  | 0.0000 | 0.5383 | 0.9468 | 0.5717 | 0.3073 | 0.6158 |
| dapt_sft_lora_dpo_best_chrf |  | 0.0000 | 0.5571 | 0.9178 | 0.5455 | 0.3349 | 0.6105 |
| poemetric_reranking |  | 0.1667 | 0.5871 | 0.9631 | 0.5922 | 0.3672 | 0.6496 |
