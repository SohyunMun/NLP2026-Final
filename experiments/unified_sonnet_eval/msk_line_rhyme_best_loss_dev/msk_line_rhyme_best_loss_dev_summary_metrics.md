# Sonnet evaluation summary: msk_line_rhyme_best_loss_dev

- prediction file: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_line_rhyme_rerank_10epoch/basic_plus_extra/predictions/dev_best_loss.txt`
- evaluated sonnets: 12
- scoring part for reference/diversity: `full`

| category | metric | value | direction |
|---|---:|---:|---|
| Reference similarity | `chrF` | 38.193021 | higher |
| Reference similarity | `BLEU` | 23.370958 | higher |
| Reference similarity | `ROUGE-L` | 0.306950 | higher |
| Reference similarity | `token-F1` | 0.390274 | higher |
| Sonnet form | `exact_14_lines_rate` | 1.000000 | higher |
| Sonnet form | `line_count_score` | 1.000000 | higher |
| Sonnet form | `line_length_score` | 0.788690 | higher |
| Sonnet form | `shakespearean_rhyme_pair_score` | 0.925000 | higher |
| Sonnet form | `final_couplet_rhyme` | 1.000000 | higher |
| Diversity | `MATTR` | 0.920642 | higher |
| Diversity | `distinct_1` | 0.416983 | higher |
| Diversity | `distinct_2` | 0.934200 | higher |
| Repetition | `repetition_rate` | 0.583017 | lower |
| Prompt faithfulness | `prompt_preservation` | 1.000000 | higher |
| Theme | `prompt_continuation_theme_overlap` | 0.199132 | higher |
| POEMetric proxy | `imagery_literary_device_score` | 0.179291 | higher |
| Leakage check | `leakage_any_line_overlap_rate` | 0.000000 | lower |
| Leakage check | `leakage_ngram_overlap_rate` | 0.000000 | lower |
