# Sonnet Evaluation: chrF / Sonnet-or-Not / POEMetric

- rank_by: `POEMetric_proxy`
- score_part: `full`
- prediction_mode: `full`

| rank | model | chrF | Sonnet-or-Not pass | form accuracy | lexical diversity | overall quality | theme | POEMetric |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `sft_plus_extra` | 40.4982 | 0.0000 | 0.5358 | 0.9481 | 0.5664 | 0.3231 | 0.6162 |
| 2 | `selected_lora_plus_extra` | 41.7313 | 0.0000 | 0.5455 | 0.9422 | 0.5706 | 0.2785 | 0.6122 |
| 3 | `base_plus_extra` | 41.0941 | 0.0000 | 0.5392 | 0.9542 | 0.5726 | 0.2632 | 0.6116 |
| 4 | `dapt_plus_extra` | 41.1442 | 0.0000 | 0.5510 | 0.9332 | 0.5495 | 0.2450 | 0.6002 |
| 5 | `base_basic` | 41.8252 | 0.0000 | 0.5365 | 0.9513 | 0.5567 | 0.2267 | 0.5998 |
| 6 | `dapt_sft_lora_dpo_best_chrf` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |

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
