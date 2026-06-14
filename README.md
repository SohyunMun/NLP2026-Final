# 자연어처리 2026-1 기말 프로젝트: GPT-2 기반 Sonnet Generation

본 프로젝트는 자연어처리 2026-1 지정주제 기말 프로젝트인 GPT-2 구현 및 응용 과제의 제출용 저장소이다. 과제의 기본 요구사항인 GPT-2 구성 요소 구현, optimizer 구현, sentiment classification, paraphrase detection, sonnet generation을 포함하며, 최종적으로 sonnet generation 성능 향상을 위해 데이터 확장, 학습 방식 비교, LoRA/DPO 기반 개선 실험을 수행했다.

## 1. 프로젝트 목표

본 프로젝트의 목표는 다음 세 가지이다.

1. GPT-2의 핵심 구성 요소를 직접 구현하고 제공된 테스트로 검증한다.
2. GPT-2 기반 sonnet generation baseline을 완성한다.
3. sonnet generation 성능 향상을 위해 데이터 정제, 추가 학습, SFT/DAPT/LoRA/DPO 실험을 수행하고 정량적으로 비교한다.

## 2. 제출 코드 구성

| 구분 | 주요 파일 및 폴더 | 설명 |
|---|---|---|
| GPT-2 구현 | `modules/attention.py`, `modules/gpt2_layer.py`, `models/gpt2.py` | attention, transformer block, GPT-2 forward 구현 |
| Optimizer 구현 | `optimizer.py` | AdamW optimizer 구현 |
| Sentiment classification | `classifier.py` | GPT-2 기반 문장 분류 실험 |
| Paraphrase detection | `paraphrase_detection.py` | GPT-2 기반 paraphrase detection 실험 |
| Sonnet baseline | `sonnet_generation.py` | 과제 starter code 기반 sonnet generation 구현 |
| Sonnet 개선 실험 | `sonnet_project/` | 데이터 정제, 확장 학습, 평가 지표, 실험 결과 정리 |
| 실행 스크립트 | `task_scripts/` | 과제 흐름별 실행 wrapper |

루트의 `sonnet_generation.py`는 공식 과제의 baseline entry로 유지했다. 추가 실험에 필요한 확장 구현은 `sonnet_project/` 내부에 분리하여, 기본 과제 코드와 개선 실험 코드가 섞이지 않도록 구성했다.

## 3. 환경 설정

과제에서 제공한 `env.yml`을 사용한다. 제출 코드 재현 시에는 아래 명령으로 환경을 생성한다.

```bash
conda env create -f env.yml
conda activate nlp_final
```

주의 사항:

- `env.yml`의 패키지 버전은 변경하지 않는다.
- 공식 과제 파일의 command option과 기본 실행 방식은 유지한다.
- 개선 실험은 별도 폴더인 `sonnet_project/`와 `task_scripts/`를 통해 실행한다.
- GPU 환경에서는 서버의 CUDA/driver/PyTorch CUDA build를 변경하지 않고, 코드와 실험 설정만 조정한다.

## 4. 실행 방법

저장소 루트에서 다음 명령을 실행한다.

```bash
cd <repository-root>
```

### 4.1 GPT-2 구현 검증

`optimizer.py`와 GPT-2 구현이 올바르게 동작하는지 확인한다.

```bash
bash task_scripts/01_run_gpt2_code_completion.sh
```

위 스크립트는 기본적으로 다음 파일을 실행한다.

- `optimizer_test.py`
- `sanity_check.py`

classifier까지 함께 실행하려면 다음 옵션을 사용한다.

```bash
bash task_scripts/01_run_gpt2_code_completion.sh --full-classifier
```

### 4.2 Sonnet generation baseline 실행

기본 sonnet generation 코드는 루트의 `sonnet_generation.py`에서 실행된다.

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh --use_gpu
```

이 실행은 과제 starter code의 기본 command option을 유지한 baseline이다.

### 4.3 Sonnet generation 개선 실험 실행

데이터 확장과 여러 학습 방법을 비교하는 개선 실험은 다음 명령으로 실행한다.

```bash
bash task_scripts/03_run_sonnet_generation_improvements.sh
```

여러 GPU를 사용할 수 있는 환경에서는 다음처럼 GPU 번호를 지정할 수 있다.

```bash
GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh
```

기존 산출물이 있어도 전체 실험을 다시 실행하려면 다음처럼 실행한다.

```bash
FORCE=1 GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh
```

실제로 실행하기 전에 어떤 명령이 수행되는지만 확인하려면 `--dry-run`을 사용한다.

```bash
bash task_scripts/03_run_sonnet_generation_improvements.sh --dry-run
```

## 5. Sonnet 데이터 구성

sonnet generation 실험에는 공식 train/dev/test split과 정제된 추가 sonnet 데이터를 사용했다.

| 데이터 | 경로 | 개수 | 설명 |
|---|---|---:|---|
| 기본 train | `sonnet_project/data/basic/train_131.txt` | 131 | 공식 Shakespeare sonnet train set |
| dev prompt | `sonnet_project/data/strict_497/dev_prompts_12.txt` | 12 | dev 입력 prompt |
| dev gold | `sonnet_project/data/strict_497/dev_gold_12.txt` | 12 | dev reference sonnet |
| test prompt | `sonnet_project/data/strict_497/test_prompts_12.txt` | 12 | test 입력 prompt |
| 추가 train | `sonnet_project/data/strict_497/extra_train_strict_497.txt` | 497 | 중복과 leakage 위험을 제거한 추가 sonnet |
| 확장 train | `sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt` | 628 | 공식 train 131개 + 추가 train 497개 |

추가 데이터는 dev/test prompt 및 dev gold와의 line-level overlap, 공식 train과의 exact duplicate 가능성을 제거한 뒤 사용했다. 데이터 정제 기준은 `sonnet_project/data/docs/strict_497_manifest.md`에 정리했다.

## 6. 비교한 학습 설정

sonnet generation 개선을 위해 아래 여섯 가지 설정을 비교했다.

| run | 설명 |
|---|---|
| `base_basic` | 기본 GPT-2를 공식 train 131개로 fine-tuning |
| `base_plus_extra` | 기본 GPT-2를 공식 train 131개 + 추가 497개로 fine-tuning |
| `sft_plus_extra` | prompt-conditioned sonnet generation에 맞춘 SFT |
| `dapt_plus_extra` | sonnet corpus에 대한 domain-adaptive pretraining |
| `selected_lora_plus_extra` | dev chrF 기준으로 선택한 checkpoint에서 LoRA-SFT |
| `dapt_sft_lora_dpo_best_chrf` | DAPT -> SFT -> LoRA-DPO 순서로 학습한 best chrF 모델 |

실험 실행 코드는 `sonnet_project/scripts/run_sixway_sonnet_ablation.py`에 정리되어 있다.

## 7. 평가 지표

평가는 reference-based metric과 reference-free proxy metric을 함께 사용했다.

| 범주 | 지표 | 설명 |
|---|---|---|
| Reference similarity | `chrF` | 생성 sonnet과 gold reference의 character n-gram 유사도 |
| Sonnet-or-Not proxy | exact 14 lines, line length, rhyme pair, final couplet rhyme | sonnet 형식과 rhyme 조건 충족 여부 |
| POEMetric proxy | form accuracy, lexical diversity, overall quality, theme overlap | 형식, 어휘 다양성, 품질 proxy, prompt 주제 반영 정도를 결합 |
| Leakage check | line/ngram overlap | train/dev/test 간 중복 및 leakage 위험 확인 |

dev set에는 gold reference가 있으므로 `chrF`를 계산했다. test set에는 gold reference가 제공되지 않으므로 test `chrF`는 계산하지 않고, reference-free proxy metric만 사용했다.

평가 스크립트:

```bash
python sonnet_project/scripts/evaluate_sonnet_poemetric.py \
  --prompts sonnet_project/data/strict_497/dev_prompts_12.txt \
  --gold sonnet_project/data/strict_497/dev_gold_12.txt \
  --out_dir sonnet_project/experiments/custom_eval/dev \
  --run MODEL_NAME=PATH/TO/predictions.txt
```

평가 지표의 상세 정의는 `sonnet_project/docs/EVALUATION_METRICS_GUIDE.md`에 정리했다.

## 8. 주요 실험 결과

dev set 평가 결과는 다음과 같다.

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_basic` | 41.8252 | 0.0000 | 0.5365 | 0.9513 | 0.5567 | 0.2267 | 0.5998 |
| `base_plus_extra` | 41.0941 | 0.0000 | 0.5392 | 0.9542 | 0.5726 | 0.2632 | 0.6116 |
| `sft_plus_extra` | 40.4982 | 0.0000 | 0.5358 | 0.9481 | 0.5664 | 0.3231 | 0.6161 |
| `dapt_plus_extra` | 41.1442 | 0.0000 | 0.5510 | 0.9332 | 0.5495 | 0.2450 | 0.6002 |
| `selected_lora_plus_extra` | 41.7313 | 0.0000 | 0.5455 | 0.9422 | 0.5706 | 0.2785 | 0.6122 |
| `dapt_sft_lora_dpo_best_chrf` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |

주요 해석:

- `chrF` 기준 최고 모델은 `dapt_sft_lora_dpo_best_chrf`이다.
- `POEMetric` 기준으로는 `sft_plus_extra`가 가장 안정적인 결과를 보였다.
- DPO는 reference similarity와 form score를 개선했지만, lexical diversity와 overall quality proxy는 낮아지는 경향을 보였다.
- 엄격한 Sonnet-or-Not pass rate는 대부분 0에 가까웠기 때문에, rhyme과 line structure를 직접 제어하는 추가 개선이 필요하다.

전체 dev/test 결과와 세부 해석은 `sonnet_project/experiments/sixway_ablation/SUMMARY.md`와 `sonnet_project/reports/SONNET_GENERATION_PROJECT_REPORT_KO.md`에 정리했다.

## 9. 산출물 위치

| 산출물 | 경로 |
|---|---|
| 최종 결과 요약 | `sonnet_project/experiments/sixway_ablation/SUMMARY.md` |
| 최종 보고서 | `sonnet_project/reports/SONNET_GENERATION_PROJECT_REPORT_KO.md` |
| 데이터 설명 | `sonnet_project/data/README.md` |
| 평가 지표 설명 | `sonnet_project/docs/EVALUATION_METRICS_GUIDE.md` |
| 과제 흐름별 실행법 | `task_scripts/README_KO.md` |
