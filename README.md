# 자연어처리 2026-1 지정주제 기말 프로젝트: GPT-2 구축

이 저장소는 자연어처리 2026-1 기말 프로젝트의 GPT-2 구현 과제와 sonnet generation 개선 실험을 함께 정리한 `MSK` 브랜치이다.

공식 starter code의 기본 구조와 command option은 루트 파일에 유지했고, 추가 데이터 정제, 확장 학습, DPO/LoRA 실험, 평가 스크립트, 최종 결과는 `sonnet_project/` 아래에 분리했다.

## 프로젝트 구성

| 구분 | 위치 | 설명 |
|---|---|---|
| GPT-2 코드 완성 | `modules/`, `models/`, `optimizer.py`, `classifier.py` | attention, GPT-2 layer, GPT-2 model, optimizer, classifier 구현 |
| Sonnet generation 기본 구현 | `sonnet_generation.py` | 과제 starter code의 공식 baseline entry |
| Sonnet generation 개선 실험 | `sonnet_project/` | 데이터 정제, 추가 학습, SFT/DAPT/LoRA/DPO, 평가 및 결과 정리 |
| 실행 wrapper | `task_scripts/` | 과제 흐름별 실행 스크립트 |
| 원본 과제 데이터 | `data/` | 공식 train/dev/test 및 기타 task 데이터 |

## 빠른 실행

아래 세 스크립트는 과제 진행 흐름을 그대로 나눈 것이다.

```bash
cd /home/msko021220/nlp2026-final-MSK
```

### 1. GPT-2 코드 완성 확인

```bash
bash task_scripts/01_run_gpt2_code_completion.sh
```

기본 실행은 `optimizer_test.py`와 `sanity_check.py`를 실행한다. classifier까지 포함하려면 다음처럼 실행한다.

```bash
bash task_scripts/01_run_gpt2_code_completion.sh --full-classifier
```

### 2. Sonnet generation 기본 코드 실행

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh --use_gpu
```

이 스크립트는 루트의 `sonnet_generation.py`를 실행한다. 이 파일은 공식 과제용 baseline이므로 command option을 임의로 바꾸지 않고 유지한다.

### 3. Sonnet generation 개선 실험 실행

```bash
GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh
```

이 스크립트는 `sonnet_project/scripts/run_sixway_sonnet_ablation.py`를 실행한다. 이미 생성된 checkpoint와 prediction이 있으면 누락된 단계만 재사용/평가하고, 전체를 다시 돌리고 싶으면 다음처럼 실행한다.

```bash
FORCE=1 GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh
```

긴 학습을 시작하기 전에 실제 실행될 명령만 확인하려면 `--dry-run`을 붙인다.

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh --dry-run --use_gpu
GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh --dry-run
```

자세한 실행법은 `task_scripts/README_KO.md` 참고.

## 환경 설정

과제 안내에 따라 `env.yml`의 버전은 변경하지 않는다.

```bash
conda env create -f env.yml
conda activate nlp_final
```

특정 Python 실행 파일을 명시해야 하는 경우 wrapper script에 `PYTHON_BIN`을 지정할 수 있다.

```bash
PYTHON_BIN=/path/to/python bash task_scripts/01_run_gpt2_code_completion.sh
```

주의 사항:

- PART-I에서는 제공된 환경의 패키지만 사용한다.
- 공식 과제 command option이나 파라미터는 루트 baseline 코드에서 변경하지 않는다.
- 개선 실험에 필요한 확장 코드는 `sonnet_project/` 내부에 분리한다.
- GPU/CUDA/driver/PyTorch CUDA build는 변경하지 않는다.

## PART-I: GPT-2 코드 완성

완성 대상 파일:

| 파일 | 역할 |
|---|---|
| `modules/attention.py` | multi-head self-attention 구현 |
| `modules/gpt2_layer.py` | GPT-2 transformer block 구현 |
| `models/gpt2.py` | GPT-2 model forward path 구현 |
| `optimizer.py` | AdamW optimizer 구현 |
| `classifier.py` | GPT 기반 sentiment classifier 구현 |

검증:

```bash
python optimizer_test.py
python sanity_check.py
python classifier.py
```

wrapper script를 사용할 경우:

```bash
bash task_scripts/01_run_gpt2_code_completion.sh
```

## PART-II: Sonnet Generation 기본 구현

공식 baseline 파일:

```text
sonnet_generation.py
```

주요 내용:

- `SonnetGPT.forward()` 구현
- autoregressive language modeling 기반 sonnet 생성
- 첫 세 줄 prompt를 조건으로 나머지 sonnet 생성
- 공식 starter code의 command option 유지

실행:

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh --use_gpu
```

## Sonnet Generation 개선 실험

개선 실험은 루트 baseline을 직접 복잡하게 만들지 않고 `sonnet_project/`에 분리했다.

핵심 파일:

| 항목 | 경로 |
|---|---|
| 프로젝트 안내 | `sonnet_project/README_KO.md` |
| 최종 보고서 | `sonnet_project/reports/SONNET_GENERATION_PROJECT_REPORT_KO.md` |
| 최종 결과표 | `sonnet_project/experiments/sixway_ablation/SUMMARY.md` |
| 데이터 설명 | `sonnet_project/data/README.md` |
| 평가 지표 설명 | `sonnet_project/docs/EVALUATION_METRICS_GUIDE.md` |
| 파일 구조 안내 | `sonnet_project/docs/PROJECT_FILE_GUIDE_KO.md` |
| 통합 평가 스크립트 | `sonnet_project/scripts/evaluate_sonnet_poemetric.py` |
| six-way 실험 runner | `sonnet_project/scripts/run_sixway_sonnet_ablation.py` |
| 개선용 sonnet 구현 | `sonnet_project/scripts/sonnet_generation_enhanced.py` |

비교한 실험 설정:

| run | 설명 |
|---|---|
| `base_basic` | 기본 GPT-2 + 공식 train 131개 |
| `base_plus_extra` | 기본 GPT-2 + 공식 train 131개 + 추가 497개 |
| `sft_plus_extra` | SFT 방식 + 추가 데이터 |
| `dapt_plus_extra` | DAPT 방식 + 추가 데이터 |
| `selected_lora_plus_extra` | dev chrF 기준 더 나은 checkpoint에서 LoRA-SFT |
| `dapt_sft_lora_dpo_best_chrf` | DAPT -> SFT -> LoRA-DPO, best dev chrF checkpoint |

## 데이터

sonnet generation 전용 데이터는 `sonnet_project/data/`에 정리했다.

| 데이터 그룹 | 파일 | 개수 |
|---|---|---:|
| 기본 train | `sonnet_project/data/basic/train_131.txt` | 131 |
| dev prompt | `sonnet_project/data/strict_497/dev_prompts_12.txt` | 12 |
| dev gold | `sonnet_project/data/strict_497/dev_gold_12.txt` | 12 |
| test prompt | `sonnet_project/data/strict_497/test_prompts_12.txt` | 12 |
| strict extra train | `sonnet_project/data/strict_497/extra_train_strict_497.txt` | 497 |
| 확장 train | `sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt` | 628 |

추가 데이터는 SHM branch의 extra data를 기반으로 하되, dev/test prompt 및 dev gold와 겹칠 가능성이 있는 항목과 공식 train 중복 가능성을 제거했다. 최종적으로 leakage 위험을 줄인 strict extra 497개만 사용했다.

세부 정제 기준은 `sonnet_project/data/docs/strict_497_manifest.md` 참고.

## 평가 방법

사용한 평가 지표:

| 범주 | 지표 | 설명 |
|---|---|---|
| Reference similarity | `chrF` | 생성문과 gold sonnet의 character n-gram 유사도 |
| Sonnet-or-Not, Bot? proxy | 14 lines, line length, rhyme pair, final couplet rhyme | sonnet 형식과 rhyme 조건을 rule-based로 평가 |
| POEMetric proxy | form accuracy, lexical diversity, overall quality, theme overlap | 시 형식, 어휘 다양성, 품질 proxy, prompt/theme 반영 정도 평가 |
| Leakage check | line/ngram overlap | train/dev/test 간 중복 또는 leakage 위험 확인 |

dev set에는 gold reference가 있으므로 `chrF`를 계산했다. test set에는 gold reference가 없으므로 test `chrF`는 비워두고, reference-free proxy 지표만 계산했다.

평가만 별도로 실행하려면:

```bash
python sonnet_project/scripts/evaluate_sonnet_poemetric.py \
  --prompts sonnet_project/data/strict_497/dev_prompts_12.txt \
  --gold sonnet_project/data/strict_497/dev_gold_12.txt \
  --out_dir sonnet_project/experiments/custom_eval/dev \
  --run MODEL_NAME=PATH/TO/predictions.txt
```

## 주요 결과

dev 평가 결과:

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_basic` | 41.8252 | 0.0000 | 0.5365 | 0.9513 | 0.5567 | 0.2267 | 0.5998 |
| `base_plus_extra` | 41.0941 | 0.0000 | 0.5392 | 0.9542 | 0.5726 | 0.2632 | 0.6116 |
| `sft_plus_extra` | 40.4982 | 0.0000 | 0.5358 | 0.9481 | 0.5664 | 0.3231 | 0.6161 |
| `dapt_plus_extra` | 41.1442 | 0.0000 | 0.5510 | 0.9332 | 0.5495 | 0.2450 | 0.6002 |
| `selected_lora_plus_extra` | 41.7313 | 0.0000 | 0.5455 | 0.9422 | 0.5706 | 0.2785 | 0.6122 |
| `dapt_sft_lora_dpo_best_chrf` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |

결론:

- dev `chrF`는 `dapt_sft_lora_dpo_best_chrf`가 가장 높다.
- dev/test `POEMetric`은 `sft_plus_extra`가 가장 안정적이다.
- DPO는 reference similarity와 form score를 올렸지만 lexical diversity와 overall quality proxy는 낮아졌다.
- 엄격한 Sonnet-or-Not pass는 대부분 0이므로 rhyme/form constraint를 더 직접적으로 넣는 개선이 필요하다.

전체 결과는 `sonnet_project/experiments/sixway_ablation/SUMMARY.md` 참고.

## 권장 사용 순서

1. `bash task_scripts/01_run_gpt2_code_completion.sh`로 GPT-2 기본 구현 검증
2. `bash task_scripts/02_run_sonnet_generation_baseline.sh --use_gpu`로 공식 baseline 실행
3. `GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh`로 개선 실험 실행
4. `sonnet_project/experiments/sixway_ablation/SUMMARY.md`에서 결과 확인
5. `sonnet_project/reports/SONNET_GENERATION_PROJECT_REPORT_KO.md`를 기반으로 최종 보고서 작성

## 제출 관점 정리

제출 시 핵심적으로 보여줄 내용:

- GPT-2 구현 파일이 정상 테스트를 통과했는지
- `sonnet_generation.py`가 공식 baseline entry로 유지되는지
- 추가 실험이 공식 baseline과 분리되어 있는지
- 사용 데이터와 leakage 제거 과정이 명확한지
- `chrF`, Sonnet-or-Not proxy, POEMetric proxy 결과가 표로 비교되는지
- 어떤 모델이 어떤 지표에서 유리했는지 해석이 포함되는지

관련 문서:

- `task_scripts/README_KO.md`
- `sonnet_project/README_KO.md`
- `sonnet_project/data/README.md`
- `sonnet_project/docs/EVALUATION_METRICS_GUIDE.md`
- `sonnet_project/reports/SONNET_GENERATION_PROJECT_REPORT_KO.md`
