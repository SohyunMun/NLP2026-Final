# Sonnet Evaluation: chrF / Sonnet-or-Not / POEMetric

- rank_by: `POEMetric_proxy`
- score_part: `full`
- prediction_mode: `full`

| rank | model | chrF | Sonnet-or-Not pass | form accuracy | lexical diversity | overall quality | theme | POEMetric |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `dapt_sft_intermediate` | 40.9748 | 0.0000 | 0.5354 | 0.9500 | 0.5637 | 0.2068 | 0.5982 |
| 2 | `dapt_sft_lora_dpo_best_chrf` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |

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
