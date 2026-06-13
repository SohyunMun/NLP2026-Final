# Sonnet Generation Evaluation Guide

이 문서는 sonnet generation 결과를 같은 기준으로 평가하기 위한 공통 스크립트 사용법이다.

평가 스크립트:

```bash
/home/msko021220/nlp2026-final-MSK/evaluate_sonnet_metrics.py
```

GPU나 모델 로딩을 사용하지 않고, 생성 결과 텍스트 파일만 읽어서 평가한다.

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

prediction 번호가 gold/prompt 번호와 다르면 기본값 `--align auto`가 index 순서로 맞춘다. 현재 우리 실험처럼 prediction은 `0, 1, 2...`, dev gold는 `132, 133...`인 경우에도 그대로 사용 가능하다.

## Dev 평가 예시

gold reference가 있는 dev set에서는 모든 지표가 계산된다.

```bash
cd /home/msko021220/nlp2026-final-MSK

/home/msko021220/.conda/envs/busi2/bin/python evaluate_sonnet_metrics.py \
  --name msk_line_rhyme_best_loss_dev \
  --pred experiments/dpo_line_rhyme_rerank_10epoch/basic_plus_extra/predictions/dev_best_loss.txt \
  --gold sonnet_data/strict_497/dev_gold_12.txt \
  --prompts sonnet_data/strict_497/dev_prompts_12.txt \
  --train_file sonnet_data/strict_497/train_official_131_plus_extra_497_total_628.txt \
  --dev_file sonnet_data/strict_497/dev_prompts_12.txt \
  --test_file sonnet_data/strict_497/test_prompts_12.txt \
  --out_dir experiments/unified_sonnet_eval/msk_line_rhyme_best_loss_dev
```

## Test 평가 예시

test set은 gold가 없으면 `--gold`를 빼고 실행한다. 이 경우 chrF, BLEU, ROUGE-L, token-F1은 비어 있고, 형식/다양성/반복/prompt/leakage 지표는 계산된다.

```bash
cd /home/msko021220/nlp2026-final-MSK

/home/msko021220/.conda/envs/busi2/bin/python evaluate_sonnet_metrics.py \
  --name msk_line_rhyme_best_loss_test \
  --pred experiments/dpo_line_rhyme_rerank_10epoch/basic_plus_extra/predictions/test_best_loss.txt \
  --prompts sonnet_data/strict_497/test_prompts_12.txt \
  --train_file sonnet_data/strict_497/train_official_131_plus_extra_497_total_628.txt \
  --dev_file sonnet_data/strict_497/dev_prompts_12.txt \
  --test_file sonnet_data/strict_497/test_prompts_12.txt \
  --out_dir experiments/unified_sonnet_eval/msk_line_rhyme_best_loss_test
```

## 출력 파일

실행하면 `--out_dir` 아래에 네 파일이 생성된다.

| 파일 | 내용 |
|---|---|
| `{name}_per_sonnet_metrics.csv` | sonnet별 세부 지표 |
| `{name}_summary_metrics.csv` | 모델별 요약 지표 1행 |
| `{name}_summary_metrics.json` | 요약 지표 JSON |
| `{name}_summary_metrics.md` | 사람이 읽기 쉬운 요약표 |

## 사용하는 지표

| 범주 | 지표 | 해석 |
|---|---|---|
| Reference similarity | chrF | character n-gram F-score. gold와 표면적으로 얼마나 비슷한지 측정. 높을수록 좋음 |
| Reference similarity | BLEU | n-gram precision 기반 유사도. 높을수록 좋음 |
| Reference similarity | ROUGE-L | longest common subsequence 기반 유사도. 높을수록 좋음 |
| Reference similarity | token-F1 | token overlap의 precision/recall 조화평균. 높을수록 좋음 |
| Sonnet form | exact 14 lines | 정확히 14행이면 1, 아니면 0. 평균은 14행 성공률 |
| Sonnet form | line count score | 14행에 가까울수록 높은 점수 |
| Sonnet form | line length score | 각 행이 대략 8-12 token이면 높은 점수 |
| Sonnet form | Shakespearean rhyme pair score | ABAB CDCD EFEF GG pair가 맞는 정도 |
| Sonnet form | final couplet rhyme | 마지막 두 행 GG rhyme이 맞는 정도 |
| Diversity | MATTR | moving-average type-token ratio. 어휘 다양성. 높을수록 좋음 |
| Diversity | distinct-1 / distinct-2 | corpus-level unique unigram/bigram 비율. 높을수록 좋음 |
| Repetition | repetition rate | 반복 token 비율. 낮을수록 좋음 |
| Prompt faithfulness | prompt preservation | 생성 결과의 첫 prompt lines가 입력 prompt와 같은지 확인 |
| Theme | prompt-continuation theme overlap | prompt와 continuation의 핵심어/주제군 overlap |
| POEMetric proxy | imagery / literary device lexicon score | imagery 단어, literary marker, alliteration 기반 proxy |
| Leakage check | train/dev/test line or n-gram overlap | 생성 continuation이 train/dev/test source와 겹치는 정도. 낮을수록 좋음 |

## 옵션 메모

- `--score_part full`: reference similarity와 diversity를 prompt 포함 전체 14행 기준으로 계산한다. 기존 실험 결과와 맞추려면 이 기본값을 사용한다.
- `--score_part continuation`: prompt를 제외한 생성 continuation만 기준으로 계산한다. 순수 생성 성능을 더 엄격히 보고 싶을 때 사용한다.
- `--prediction_mode full`: prediction 파일에 prompt lines까지 포함되어 있을 때 사용한다. 현재 우리 생성 결과의 기본 형식이다.
- `--prediction_mode continuation`: prediction 파일이 continuation만 포함할 때 사용한다. 이 경우 prompt 파일을 붙여 full sonnet 형식 지표를 계산한다.
- `--leakage_part continuation`: leakage는 기본적으로 prompt를 제외한 생성 continuation만 검사한다. prompt는 원래 입력으로 주어지기 때문에 leakage 계산에 넣지 않는 것이 안전하다.
- `--leakage_source NAME=PATH`: train/dev/test 외 추가 corpus를 leakage source로 넣을 때 반복해서 사용할 수 있다.

## 주의점

dev gold를 leakage source로 넣으면 정답과의 overlap이 leakage처럼 표시될 수 있다. 일반적인 leakage 점검에서는 train corpus, dev/test prompt, 추가 학습 데이터 후보 등을 넣고, 평가 대상 gold reference는 `--gold`로만 넣는 편이 좋다.
