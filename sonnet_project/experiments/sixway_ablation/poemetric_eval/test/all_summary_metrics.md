# Sonnet Evaluation: chrF / Sonnet-or-Not / POEMetric

- rank_by: `POEMetric_proxy`
- score_part: `full`
- prediction_mode: `full`

| rank | model | chrF | Sonnet-or-Not pass | form accuracy | lexical diversity | overall quality | theme | POEMetric |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `poemetric_reranking` |  | 0.1667 | 0.5871 | 0.9631 | 0.5922 | 0.3672 | 0.6496 |
| 2 | `sft_plus_extra` |  | 0.0000 | 0.5381 | 0.9537 | 0.5737 | 0.3073 | 0.6181 |
| 3 | `selected_lora_plus_extra` |  | 0.0000 | 0.5383 | 0.9468 | 0.5717 | 0.3073 | 0.6158 |
| 4 | `dapt_sft_lora_dpo_best_chrf` |  | 0.0000 | 0.5571 | 0.9178 | 0.5455 | 0.3349 | 0.6105 |
| 5 | `base_plus_extra` |  | 0.0833 | 0.5512 | 0.9562 | 0.5988 | 0.1422 | 0.6054 |
| 6 | `base_basic` |  | 0.0000 | 0.5524 | 0.9550 | 0.5736 | 0.1880 | 0.6048 |
| 7 | `dapt_plus_extra` |  | 0.0000 | 0.5431 | 0.9429 | 0.5592 | 0.1927 | 0.5953 |

## Metric Definitions

- `chrF`: sacreBLEU default chrF. Gold reference가 있을 때만 계산됨.
- `Sonnet-or-Not pass`: exact 14 lines, line length, rhyme, final couplet, form threshold를 모두 만족한 비율.
- `form accuracy`: 0.35 exact14 + 0.20 line length + 0.30 Shakespearean rhyme pairs + 0.15 final couplet rhyme.
- `lexical diversity`: 0.50 MATTR + 0.50 distinct-2.
- `overall quality`: 0.35 form + 0.25 lexical diversity + 0.25 non-repetition + 0.15 imagery/literary-device proxy.
- `POEMetric`: 0.30 form + 0.25 lexical diversity + 0.30 overall quality + 0.15 prompt/theme overlap.

## Thresholds

- Sonnet-or-Not form threshold: `0.7`
- line length threshold: `0.5`
- rhyme-pair threshold: `0.25`
- final-couplet threshold: `0.25`
