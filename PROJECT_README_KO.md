# Sonnet Generation Project

이 프로젝트는 `SohyunMun/NLP2026-Final`의 기본 `main` 브랜치 과제 코드를 기반으로, sonnet generation 성능을 여섯 가지 학습 설정으로 비교한 결과를 정리한 것이다.

## 핵심 산출물

| 항목 | 경로 |
|---|---|
| 최종 보고서 | `reports/SONNET_GENERATION_PROJECT_REPORT_KO.md` |
| 최종 결과 요약 | `experiments/sixway_ablation/SUMMARY.md` |
| dev 평가 CSV | `experiments/sixway_ablation/poemetric_eval/dev/all_summary_metrics.csv` |
| test 평가 CSV | `experiments/sixway_ablation/poemetric_eval/test/all_summary_metrics.csv` |
| 평가 스크립트 | `evaluate_sonnet_poemetric.py` |
| six-way 실험 실행 스크립트 | `run_sixway_sonnet_ablation.py` |
| canonical sonnet 데이터 | `sonnet_data/` |

## 비교한 실험

1. 기본 모델 + 기본 데이터
2. 기본 모델 + 추가 데이터
3. SFT + 추가 데이터
4. DAPT + 추가 데이터
5. SFT 또는 DAPT 중 dev chrF가 더 좋은 모델 + LoRA + 추가 데이터
6. DAPT -> SFT checkpoint를 DPO policy/ref 초기값으로 사용 + LoRA-DPO + 추가 데이터

## 평가 지표

- `chrF`: gold reference가 있는 dev set에서만 계산.
- `Sonnet-or-Not, Bot? proxy`: 14행 구조, line length, Shakespearean rhyme pair, final couplet rhyme, form threshold를 모두 만족하는지 평가.
- `POEMetric proxy`: form accuracy, lexical diversity, overall quality, theme overlap을 결합한 재현 가능한 rule-based proxy.

## 최종 결론

- dev chrF 최고 모델: `dapt_sft_lora_dpo_best_chrf` (`42.7768`)
- dev/test POEMetric 최고 모델: `sft_plus_extra`
- DPO는 chrF와 form accuracy를 올렸지만 lexical diversity와 overall quality를 낮춰 POEMetric 전체에서는 SFT보다 불리했다.
- Sonnet-or-Not pass는 대부분 0으로, 엄격한 rhyme/form constraint는 추가 개선이 필요하다.

## 재현

이미 생성된 checkpoint와 prediction이 있으면 아래 명령은 학습을 다시 돌리지 않고 누락된 평가만 재생성한다.

```bash
cd /home/msko021220/nlp2026-final-MSK
/home/msko021220/.conda/envs/busi2/bin/python run_sixway_sonnet_ablation.py --gpus 0,1,2,3
```

평가만 별도로 돌리려면 다음 형식을 사용한다.

```bash
cd /home/msko021220/nlp2026-final-MSK
/home/msko021220/.conda/envs/busi2/bin/python evaluate_sonnet_poemetric.py \
  --prompts sonnet_data/strict_497/dev_prompts_12.txt \
  --gold sonnet_data/strict_497/dev_gold_12.txt \
  --out_dir experiments/custom_eval/dev \
  --run MODEL_NAME=PATH/TO/predictions.txt
```

