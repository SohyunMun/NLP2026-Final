# Shakespearean Sonnet Generation Model Evaluation Guide

이 문서는 6가지 소네트 생성 모델 설계 변형(Variation)을 테스트하고 평가 비교하기 위한 종합 안내서입니다. 이 실험은 GPU 환경이 구축된 서버에서 실행하도록 최적화되어 있으며, 쉘 스크립트를 사용해 모든 과정을 자동화할 수 있습니다.

본 가이드는 실험 수행자 및 **AI AGENT**가 원활히 프로세스를 파이프라인으로 수행하고 결과를 요약 보고하도록 작성되었습니다.

---

## 1. 실험 설계 매트릭스 (Experiment Matrix)

각 설계 변형 모델은 아래와 같이 서로 다른 데이터 구성 및 PEFT/DPO 조합을 가집니다.

| 번호 | 실험 모델명 | 학습 데이터셋 | PEFT 기법 | DPO 정렬 여부 | 특징 및 비고 |
|:---:|---|---|:---:|:---:|---|
| **1** | `sonnet_baseline` | 기본 소네트 데이터 | N/A (Full Fine-tuning) | X | 기본적인 파인튜닝 성능 측정 기준점 |
| **2** | `sonnet_dpo` | 기본 소네트 데이터 | N/A (Full DPO) | O | 데이터 무작위 훼손(Corrupted)을 통한 오프라인 Preference Alignment 학습 |
| **3** | `sonnet_lora_dpo` | 기본 소네트 데이터 | LoRA ($r=8, \alpha=16$) | O | LoRA 어댑터 가중치만을 대상으로 DPO 정렬 수행 |
| **4** | `sonnet_peft_dpo` | 기본 소네트 데이터 | LoRA + Prefix Tuning | O | 가상 프리픽스 토큰과 LoRA를 결합하여 DPO 정렬 수행 |
| **5** | `sonnet_DAPT_LORA_PEPT_DPO` | 기본 소네트 + 셰익스피어 데이터 | LoRA + Prefix Tuning | O | **3-Stage 점진적 학습 Pipeline**<br>Stage 1: DAPT (셰익스피어 극본)<br>Stage 2: SFT with PEFT (기본 소네트)<br>Stage 3: DPO (기본 소네트) |
| **6** | `sonnet_baseline` | 기본 소네트 + 셰익스피어 데이터 | N/A (Full Fine-tuning) | X | 추가 데이터셋으로 사전 적응시킨 Baseline 성능 비교용 |

---

## 2. 실행 방법 (How to Run)

### 사전 준비사항
1. Python 3.8+ 환경 및 PyTorch(CUDA 지원 환경)를 마련합니다.
2. `requirements.txt` 또는 환경 파일(`env.yml`)을 사용하여 의존성을 설치합니다.
3. 데이터 디렉토리(`data/`) 아래에 `sonnets.txt`와 기타 개발셋이 제대로 존재하고 있는지 확인합니다. 

*참고: 셰익스피어 데이터 `shakespeare_plays.txt`는 스크립트 실행 시 존재하지 않는 경우 자동으로 Karpathy 원본 레포지토리에서 다운로드(`curl` 사용)되어 데이터셋으로 병합됩니다.*

### 실행 자동화 스크립트
모든 시나리오는 제공된 `run_experiments.sh` 쉘 스크립트를 통해 손쉽게 실행할 수 있습니다.

```bash
# 1. 쉘 스크립트에 실행 권한 부여
chmod +x run_experiments.sh

# 2. 도움말 및 사용법 확인
./run_experiments.sh --help
```

### 다양한 실행 옵션 예시

*주의: GPU 환경에서 전체 실험을 돌리기 위해서는 명령어 끝에 반드시 `--gpu` 플래그를 추가해야 가속학습이 적용됩니다.*

```bash
# A. 검증 모드 (Quick Mode) - 모든 파이프라인의 오류 여부를 1 에포크 학습으로 빠르게 확인
./run_experiments.sh --all --quick --gpu

# B. 전체 6가지 설계 변형 학습 및 연속 평가 순차 실행 (기본 10 Epochs 설정)
./run_experiments.sh --all --gpu

# C. 특정 단일 실험만 단독 실행 (예: 5번 Ultimate 3-Stage Pipeline만 가속 실행)
./run_experiments.sh --run 5 --gpu

# D. 이미 완료된 실험 로그들로부터 평가 지표 테이블만 파싱하여 요약 출력
./run_experiments.sh --summary
```

---

## 3. 출력물 구조 (Output Artifacts)

스크립트 실행 시 다음과 같은 디렉토리 구조 및 파일들이 자동으로 생성되어 정리됩니다.

```
.
├── checkpoints/              # 각 실험별 최종 최적 가중치 백업본 (.pt)
│   ├── best_baseline_default.pt
│   ├── best_dpo_only.pt
│   ├── best_lora_dpo.pt
│   ├── best_peft_dpo.pt
│   ├── best_ultimate_dpo.pt  # 5번 Ultimate DPO
│   └── best_baseline_with_shakespeare.pt
│
├── logs/                     # 각 실험 실행 시의 훈련/검증 손실 및 최종 평가 로그 (.log)
│   ├── 1_baseline_default.log
│   └── ...
│
├── predictions/              # 모델이 추론(Inference) 단계에서 생성한 소네트 모음 (.txt)
│   ├── 1_baseline_default.txt
│   ├── 5_dapt_lora_peft_dpo.txt
│   └── ...
```

---

## 4. 결과 요약 분석 및 평가 지표 해석

모든 실험이 완료되거나 `--summary` 옵션을 주었을 때, AI AGENT 및 수행자는 아래와 같은 포맷의 비교 분석표를 결과로 획득할 수 있습니다.

### 핵심 평가지표 정의
* **평가 대상 데이터**: 서브셋이 아닌 12개의 전체 홀드아웃 소네트 데이터셋(`data/TRUE_sonnets_held_out_dev.txt`)을 타겟으로 정밀 평가를 진행합니다.
* **chrF Score**: 타겟(Gold) 소네트와 어휘/문맥적으로 얼마나 유사한지 글자 n-gram 수준에서 평가한 척도 (0~100)
* **Pass Rate (Sonnet or Not, Bot?)**: 14줄 정형 조건, 음절 편차 허용범위, 최소 압운(Rhyme) 조건 통과 비율 (0%~100%)
* **POEMetric Score**: 시의 형식적 정확도(음절수, 정형률, 압운) 30%, 어휘 다양성 20%, 생성 품질(chrF) 30%, 주제 일관성(Theme Alignment) 20%를 통합한 최종 평가 수치 (최대 1.0)

*경고: 학습 진행 도중 CUDA Out of Memory(OOM) 오류가 발생할 경우, 각 스크립트 실행의 `--batch_size`를 `2` 또는 `1`로 줄여서 실행하도록 파라미터를 수정하십시오.*
