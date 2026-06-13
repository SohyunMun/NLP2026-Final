# Sonnet evaluation summary: msk_line_rhyme_best_loss_test

- prediction file: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_line_rhyme_rerank_10epoch/basic_plus_extra/predictions/test_best_loss.txt`
- evaluated sonnets: 12
- scoring part for reference/diversity: `full`

| category | metric | value | direction |
|---|---:|---:|---|
| Reference similarity | `chrF` |  | higher |
| Reference similarity | `BLEU` |  | higher |
| Reference similarity | `ROUGE-L` |  | higher |
| Reference similarity | `token-F1` |  | higher |
| Sonnet form | `exact_14_lines_rate` | 1.000000 | higher |
| Sonnet form | `line_count_score` | 1.000000 | higher |
| Sonnet form | `line_length_score` | 0.752976 | higher |
| Sonnet form | `shakespearean_rhyme_pair_score` | 0.892857 | higher |
| Sonnet form | `final_couplet_rhyme` | 0.891667 | higher |
| Diversity | `MATTR` | 0.934089 | higher |
| Diversity | `distinct_1` | 0.465743 | higher |
| Diversity | `distinct_2` | 0.963481 | higher |
| Repetition | `repetition_rate` | 0.534257 | lower |
| Prompt faithfulness | `prompt_preservation` | 1.000000 | higher |
| Theme | `prompt_continuation_theme_overlap` | 0.175000 | higher |
| POEMetric proxy | `imagery_literary_device_score` | 0.261755 | higher |
| Leakage check | `leakage_any_line_overlap_rate` | 0.000000 | lower |
| Leakage check | `leakage_ngram_overlap_rate` | 0.000000 | lower |
