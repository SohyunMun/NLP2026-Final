# Sonnet Data Layout

이 폴더는 sonnet generation 실험에서 사용할 데이터를 한곳에 정리한 canonical data directory이다.

## Recommended Data Groups

### 1. Basic

공식 Shakespeare sonnet train만 사용하는 기본 설정이다.

| split | file | count |
|---|---|---:|
| train | `basic/train_131.txt` | 131 |
| dev prompt | `basic/dev_prompts_12.txt` | 12 |
| dev gold | `basic/dev_gold_12.txt` | 12 |
| test prompt | `basic/test_prompts_12.txt` | 12 |

### 2. Strict 497 Extra

공식 train 131개에 leakage와 공식 중복을 제거한 extra sonnet 497개를 더한 설정이다. 현재 이후 실험에서는 이 그룹을 기본 확장 데이터로 사용한다.

| split | file | count |
|---|---|---:|
| official train | `strict_497/official_train_131.txt` | 131 |
| strict extra train | `strict_497/extra_train_strict_497.txt` | 497 |
| train combined | `strict_497/train_official_131_plus_extra_497_total_628.txt` | 628 |
| dev prompt | `strict_497/dev_prompts_12.txt` | 12 |
| dev gold | `strict_497/dev_gold_12.txt` | 12 |
| test prompt | `strict_497/test_prompts_12.txt` | 12 |

## Why Strict 497

SHM branch의 최신 extra data는 553개였지만, prompt-level leakage와 공식 train 중복 가능성을 더 엄격하게 제거했다.

제거 기준:

- dev/test prompt 또는 dev gold와 line-level overlap이 있는 extra sonnet 제거
- 공식 train과 strict-normalized exact duplicate인 extra sonnet 제거
- 최종 사용 extra sonnet 수: 497

세부 정제 기록은 `docs/strict_497_manifest.md` 참고.

## Evaluation Script Example

```bash
cd /home/msko021220/nlp2026-final-MSK

/home/msko021220/.conda/envs/busi2/bin/python sonnet_project/scripts/evaluate_sonnet_metrics.py \
  --name MODEL_NAME_dev \
  --pred PATH/TO/generated_dev.txt \
  --gold sonnet_project/data/strict_497/dev_gold_12.txt \
  --prompts sonnet_project/data/strict_497/dev_prompts_12.txt \
  --train_file sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt \
  --dev_file sonnet_project/data/strict_497/dev_prompts_12.txt \
  --test_file sonnet_project/data/strict_497/test_prompts_12.txt \
  --out_dir sonnet_project/experiments/custom_eval/MODEL_NAME_dev
```

## Notes

- repository root의 `data/` 폴더는 원래 과제의 전체 task 데이터를 보존한 위치이다.
- `sonnet_project/data/`는 sonnet generation 전용으로 다시 정리한 폴더이다.
- 새 실험에서는 가능하면 `sonnet_project/data/strict_497/` 경로를 사용한다.
