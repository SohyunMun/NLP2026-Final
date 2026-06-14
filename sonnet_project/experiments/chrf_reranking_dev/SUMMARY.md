# Dev-only chrF Reranking Analysis

이 실험은 `chrF` gold reference가 있는 dev set에서만 수행한 분석이다.
test gold는 없고, test 결과를 보고 설정을 바꾸는 것을 피하기 위해 test에는 적용하지 않았다.

## Fairness Policy

- `oracle_chrf`: 각 dev prompt의 gold를 보고 최고 chrF 후보를 선택함. 최종 성능이 아니라 후보군 상한선 확인용.
- `leave_one_out_chrf_tuned`: held-out prompt를 하나 뺀 나머지 dev prompt에서 chrF가 가장 높은 reference-free recipe를 고른 뒤 held-out prompt에 적용함.
- 따라서 `leave_one_out_chrf_tuned`도 작은 dev set에 대한 분석용이며, test 성능 주장에는 사용하지 않음.

## Outputs

- oracle prediction: `sonnet_project/experiments/chrf_reranking_dev/predictions/dev_oracle_chrf.txt`
- leave-one-out prediction: `sonnet_project/experiments/chrf_reranking_dev/predictions/dev_loo_chrf_tuned.txt`
- candidate metrics with chrF: `sonnet_project/experiments/chrf_reranking_dev/candidate_metrics_with_chrf.csv`
- fold decisions: `sonnet_project/experiments/chrf_reranking_dev/loo_folds.csv`
- recipe summary: `sonnet_project/experiments/chrf_reranking_dev/recipe_summary.csv`

## Best Fixed Recipes on Full Dev

| rank | recipe | dev corpus chrF | mean selected-candidate chrF |
|---:|---|---:|---:|
| 1 | `lexical_only` | 42.2889 | 42.2739 |
| 2 | `candidate_4` | 42.2354 | 42.2330 |
| 3 | `content_proxy` | 42.2146 | 42.1856 |
| 4 | `candidate_3` | 42.2042 | 42.1874 |
| 5 | `balanced_proxy` | 42.1955 | 42.1716 |
| 6 | `chrf_proxy_light` | 42.1284 | 42.1013 |
| 7 | `non_repetition_only` | 42.1205 | 42.1033 |
| 8 | `poemetric_only` | 42.1131 | 42.0897 |
| 9 | `theme_only` | 42.0842 | 42.0578 |
| 10 | `poemetric_rerank_score` | 42.0672 | 42.0455 |

## Selection Summary

- oracle mean candidate chrF: `43.1413`
- leave-one-out mean candidate chrF: `41.8380`

## Evaluation Comparison

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dpo_single` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |
| `poemetric_reranked` | 42.0672 | 0.1667 | 0.5804 | 0.9610 | 0.5731 | 0.3307 | 0.6359 |
| `chrf_oracle_dev` | 43.1516 | 0.0000 | 0.5464 | 0.9585 | 0.5621 | 0.2068 | 0.6032 |
| `chrf_loo_tuned_dev` | 41.8480 | 0.0833 | 0.5542 | 0.9620 | 0.5730 | 0.3108 | 0.6252 |

## Interpretation

- `chrf_oracle_dev`는 각 dev prompt의 gold reference를 직접 보고 후보를 고른 결과이므로, 실제 inference 성능이 아니라 후보군 안에 더 높은 chrF 후보가 존재하는지 보는 upper-bound 분석이다.
- `chrf_loo_tuned_dev`는 held-out prompt의 gold를 보지 않도록 leave-one-out 방식으로 구성했지만, dev set이 12개뿐이라 안정적인 모델 선택 기준으로 보기 어렵다.
- 공정한 최종 비교에서는 `dpo_single`과 gold-free `poemetric_reranked`를 중심으로 보는 것이 맞다.
- chrF 기준으로 후보를 직접 고르면 dev chrF는 `42.7768`에서 `43.1516`으로 오르지만, 이는 test에 적용할 수 없는 oracle 설정이다.
- held-out prompt gold를 보지 않는 LOO chrF-tuned 방식은 dev chrF가 `41.8480`으로 낮아져, 현재 후보군/feature만으로는 chrF를 안정적으로 예측하는 reranker를 만들기 어렵다는 것을 보여준다.
