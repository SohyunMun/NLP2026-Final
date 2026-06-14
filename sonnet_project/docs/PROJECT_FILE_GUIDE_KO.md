# Project File Guide

이 문서는 최종 제출/공유 시 어떤 파일을 보면 되는지 정리한 안내서이다.

## Entry Points

| 파일 | 역할 |
|---|---|
| `../task_scripts/README_KO.md` | 과제 흐름별 실행 스크립트 안내 |
| `README_KO.md` | 프로젝트 빠른 안내 |
| `reports/SONNET_GENERATION_PROJECT_REPORT_KO.md` | 데이터, 평가, 실험, 결과 해석을 포함한 최종 보고서 |
| `experiments/sixway_ablation/SUMMARY.md` | 7개 메인 설정의 dev/test 정량 결과 |
| `experiments/dpo_reranking/SUMMARY.md` | POEMetric 기반 reranking 세부 결과 |
| `docs/EVALUATION_METRICS_GUIDE.md` | 최종 metric script 사용 가이드 |
| `data/README.md` | 데이터 split과 strict 497 정제 설명 |

## Reproducibility Scripts

| 파일 | 역할 |
|---|---|
| `scripts/run_sixway_sonnet_ablation.py` | 6개 학습 실험의 학습/생성/평가 통합 runner |
| `scripts/run_dpo_reranking.py` | DPO checkpoint 기반 POEMetric reranking runner |
| `scripts/sonnet_generation_enhanced.py` | 추가 실험용 sonnet generation 확장 구현 |
| `scripts/evaluate_sonnet_poemetric.py` | chrF, Sonnet-or-Not proxy, POEMetric proxy 평가 |
| `scripts/evaluate_sonnet_metrics.py` | BLEU, ROUGE-L, token-F1 등 더 넓은 metric 평가 |
| `scripts/generate_sonnet_checkpoint.py` | 저장된 checkpoint로 dev/test prediction 생성 |
| `scripts/train_lora_sft_from_checkpoint.py` | 기존 checkpoint에서 LoRA-SFT 수행 |
| `scripts/run_msk_sft_lora_dpo.py` | LoRA-DPO 기본 구현 |
| `scripts/run_msk_sft_lora_dpo_form_rhyme.py` | form/rhyme-aware LoRA-DPO 구현 |

## Final Result Artifacts

| 경로 | 내용 |
|---|---|
| `experiments/sixway_ablation/base_basic/` | 기본 GPT-2 + 기본 데이터 |
| `experiments/sixway_ablation/base_plus_extra/` | 기본 GPT-2 + 추가 데이터 |
| `experiments/sixway_ablation/sft_plus_extra/` | SFT + 추가 데이터 |
| `experiments/sixway_ablation/dapt_plus_extra/` | DAPT + 추가 데이터 |
| `experiments/sixway_ablation/selected_lora_plus_extra/` | DAPT 기반 LoRA-SFT |
| `experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/` | DAPT -> SFT -> LoRA-DPO |
| `experiments/sixway_ablation/poemetric_eval/dev/` | dev 평가 결과 |
| `experiments/sixway_ablation/poemetric_eval/test/` | test 평가 결과 |
| `experiments/dpo_reranking/` | 7번째 메인 설정인 POEMetric reranking 결과 |

## Notes

- `.pt` checkpoint 파일은 `.gitignore`에 의해 Git 추적 대상에서 제외된다.
- `test` split에는 gold reference가 없으므로 test chrF는 계산하지 않는다.
- 예전 trial, smoke-test, superseded DPO pipeline 결과는 최종 프로젝트에서 제거했다.

## Folder Tree

```text
sonnet_project/
├── README_KO.md
├── data/
│   ├── basic/
│   ├── strict_497/
│   └── docs/
├── docs/
│   ├── EVALUATION_METRICS_GUIDE.md
│   └── PROJECT_FILE_GUIDE_KO.md
├── experiments/
│   ├── sixway_ablation/
│   └── dpo_reranking/
├── reports/
│   └── SONNET_GENERATION_PROJECT_REPORT_KO.md
└── scripts/
    ├── run_sixway_sonnet_ablation.py
    ├── run_dpo_reranking.py
    ├── sonnet_generation_enhanced.py
    ├── evaluate_sonnet_poemetric.py
    ├── evaluate_sonnet_metrics.py
    ├── generate_sonnet_checkpoint.py
    ├── train_lora_sft_from_checkpoint.py
    ├── run_msk_sft_lora_dpo.py
    └── run_msk_sft_lora_dpo_form_rhyme.py
```
