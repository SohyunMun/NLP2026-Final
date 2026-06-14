# Train-chrF Predictor Reranking

이 실험은 train gold만 사용해 후보의 chrF를 예측하는 ridge-regression reranker를 학습하고, dev/test에서는 gold 없이 feature만으로 후보를 선택한 결과이다.

## Fairness

- Reranker label: train candidate vs train gold `chrF`.
- Dev/test candidate selection: gold reference 미사용.
- Dev gold는 최종 평가에서만 사용.
- Test gold는 없으므로 사용하지 않음.

## Setup

- train data: `sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt`
- max train examples: `all`
- train candidates: `4` per prompt
- decoding strategies: `top_p,top_k`
- ridge alpha: `10.0`
- train candidate rows: `2512`
- train candidate RMSE: `1.4731`
- train candidate MAE: `1.1515`

## Internal Train Validation

- validation groups: `62`
- candidate RMSE: `1.4937`
- candidate MAE: `1.1860`
- mean selected candidate chrF: `41.6582`

## Selected Outputs

| split | selected rows | prediction | selected metrics |
|---|---:|---|---|
| dev | 12 | `sonnet_project/experiments/train_chrf_reranking_strict628/predictions/dev_train_chrf_reranked.txt` | `sonnet_project/experiments/train_chrf_reranking_strict628/candidate_metrics/dev_train_chrf_selected.csv` |
| test | 12 | `sonnet_project/experiments/train_chrf_reranking_strict628/predictions/test_train_chrf_reranked.txt` | `sonnet_project/experiments/train_chrf_reranking_strict628/candidate_metrics/test_train_chrf_selected.csv` |

## Evaluation

Dev set에는 gold reference가 있으므로 chrF를 포함해 평가했다. Test set에는 gold reference가 없으므로 reference-free proxy만 계산했다.

| split | model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | theme | POEMetric |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev | `dpo_single` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |
| dev | `poemetric_reranked` | 42.0672 | 0.1667 | 0.5804 | 0.9610 | 0.5731 | 0.3307 | 0.6359 |
| dev | `train_chrf_reranked` | 42.1890 | 0.0000 | 0.5580 | 0.9627 | 0.5619 | 0.1304 | 0.5962 |
| test | `dpo_single` |  | 0.0000 | 0.5571 | 0.9178 | 0.5455 | 0.3349 | 0.6105 |
| test | `poemetric_reranked` |  | 0.1667 | 0.5871 | 0.9631 | 0.5922 | 0.3672 | 0.6496 |
| test | `train_chrf_reranked` |  | 0.0833 | 0.5690 | 0.9661 | 0.5932 | 0.2432 | 0.6267 |

## Largest Learned Weights

| rank | feature | standardized ridge weight |
|---:|---|---:|
| 1 | `line_length_score` | 0.3092 |
| 2 | `mbr_score` | 0.2521 |
| 3 | `final_couplet_rhyme` | -0.1676 |
| 4 | `MATTR` | 0.1516 |
| 5 | `sonnet_form_accuracy` | 0.1470 |
| 6 | `poemetric_overall_quality_proxy` | 0.1088 |
| 7 | `shakespearean_rhyme_pair_score` | 0.1000 |
| 8 | `lexical_diversity` | 0.0936 |
| 9 | `non_repetition` | 0.0653 |
| 10 | `repetition_rate` | -0.0653 |
| 11 | `prompt_continuation_theme_overlap` | -0.0470 |
| 12 | `POEMetric_proxy` | 0.0428 |
