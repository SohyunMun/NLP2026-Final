# Sonnet Generation Evaluation Metrics Guide

이 문서는 최종 sonnet generation 실험에서 사용한 공통 평가 지표와 실행 방법을 정리한 가이드이다.

## 최종 평가 스크립트

```bash
/home/msko021220/nlp2026-final-MSK/sonnet_project/scripts/evaluate_sonnet_poemetric.py
```

이 스크립트는 생성 결과 텍스트 파일만 읽어서 평가하며, GPU나 모델 로딩을 사용하지 않는다.

## 입력 파일 형식

생성 결과는 아래처럼 번호가 붙은 sonnet block 형식이어야 한다.

```text
0
first line
second line
...

1
first line
second line
...
```

prediction 번호와 prompt/gold 번호가 달라도 block 순서대로 정렬해 평가한다.

## Dev 평가 예시

dev set에는 gold reference가 있으므로 `chrF`, `Sonnet-or-Not proxy`, `POEMetric proxy`를 모두 계산한다.

```bash
cd /home/msko021220/nlp2026-final-MSK

/home/msko021220/.conda/envs/busi2/bin/python sonnet_project/scripts/evaluate_sonnet_poemetric.py \
  --prompts sonnet_project/data/strict_497/dev_prompts_12.txt \
  --gold sonnet_project/data/strict_497/dev_gold_12.txt \
  --out_dir sonnet_project/experiments/custom_eval/dev \
  --run dapt_sft_lora_dpo_best_chrf=sonnet_project/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/predictions/dev_best_chrf.txt
```

## Test 평가 예시

test set에는 gold reference가 없으므로 `--gold`를 넣지 않는다. 이 경우 `chrF`는 비워지고, 생성물 자체로 계산할 수 있는 `Sonnet-or-Not proxy`와 `POEMetric proxy`만 계산한다.

```bash
cd /home/msko021220/nlp2026-final-MSK

/home/msko021220/.conda/envs/busi2/bin/python sonnet_project/scripts/evaluate_sonnet_poemetric.py \
  --prompts sonnet_project/data/strict_497/test_prompts_12.txt \
  --out_dir sonnet_project/experiments/custom_eval/test \
  --run dapt_sft_lora_dpo_best_chrf=sonnet_project/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/predictions/test_best_chrf.txt
```

## 전체 six-way 평가 결과

이미 계산된 최종 결과는 아래 파일에 정리되어 있다.

| 파일 | 내용 |
|---|---|
| `sonnet_project/experiments/sixway_ablation/SUMMARY.md` | dev/test 핵심 결과표와 해석 |
| `sonnet_project/experiments/sixway_ablation/poemetric_eval/dev/all_summary_metrics.csv` | dev 전체 요약 CSV |
| `sonnet_project/experiments/sixway_ablation/poemetric_eval/test/all_summary_metrics.csv` | test 전체 요약 CSV |
| `sonnet_project/reports/SONNET_GENERATION_PROJECT_REPORT_KO.md` | 데이터, 평가, 실험 세부 정보, 결과 해석 |

## 사용한 지표

| 범주 | 지표 | 해석 |
|---|---|---|
| Reference similarity | `chrF` | gold reference와 character n-gram 단위로 얼마나 비슷한지 측정. dev에서만 계산 |
| Sonnet-or-Not proxy | `sonnet_or_not` | 14행, line length, rhyme pair, final couplet, form threshold를 모두 통과하면 1 |
| POEMetric proxy | `poemetric_proxy` | form, lexical diversity, overall quality, theme overlap을 가중합한 rule-based proxy |
| POEMetric component | `form_accuracy` | 14행 여부, 행 길이, Shakespearean rhyme pair, final couplet rhyme 반영 |
| POEMetric component | `lexical_diversity` | MATTR과 distinct-2 기반 어휘 다양성 |
| POEMetric component | `overall_quality` | form, lexical diversity, non-repetition, imagery/literary-device score 결합 |
| POEMetric component | `theme_overlap` | prompt와 continuation 사이의 핵심어/주제 overlap |

## 계산식

```text
POEMetric proxy
= 0.30 * form_accuracy
+ 0.25 * lexical_diversity
+ 0.30 * overall_quality
+ 0.15 * theme_overlap
```

```text
form_accuracy
= 0.35 * exact_14_lines
+ 0.20 * line_length_score
+ 0.30 * shakespearean_rhyme_pair_score
+ 0.15 * final_couplet_rhyme
```

```text
lexical_diversity
= 0.50 * MATTR
+ 0.50 * distinct_2
```

```text
overall_quality
= 0.35 * form_accuracy
+ 0.25 * lexical_diversity
+ 0.25 * non_repetition
+ 0.15 * imagery_literary_device_score
```

주의: 본 프로젝트의 `Sonnet-or-Not`과 `POEMetric`은 논문 공식 evaluator가 아니라, 같은 데이터와 같은 생성 결과를 일관되게 비교하기 위해 구현한 재현 가능한 proxy metric이다.

## 보조 평가 스크립트

`evaluate_sonnet_metrics.py`는 BLEU, ROUGE-L, token-F1, leakage check까지 포함하는 확장 평가용 스크립트이다. 최종 보고서의 핵심 결과는 `evaluate_sonnet_poemetric.py` 기준으로 정리했다.
