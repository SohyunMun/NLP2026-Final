# Task Scripts

이 폴더는 과제 흐름을 세 가지 실행 단위로 나눈 wrapper script 모음이다.

## 1. GPT-2 코드 완성하기

```bash
bash task_scripts/01_run_gpt2_code_completion.sh
```

기본 실행은 아래 공식 구현 테스트를 수행한다.

- `optimizer_test.py`
- `sanity_check.py`

classifier까지 전체 학습하려면 다음처럼 실행한다.

```bash
bash task_scripts/01_run_gpt2_code_completion.sh --full-classifier
```

## 2. Sonnet Generation 코드 완성하기

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh
```

이 스크립트는 루트의 `sonnet_generation.py`를 실행한다. 이 파일은 과제 starter code의 command option을 유지한 공식 baseline entry이다.

GPU를 사용할 때는 기존 공식 옵션을 그대로 넘긴다.

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh --use_gpu
```

## 3. Sonnet Generation 코드/데이터 개선하기

```bash
bash task_scripts/03_run_sonnet_generation_improvements.sh
```

이 스크립트는 `sonnet_project/scripts/run_sixway_sonnet_ablation.py`를 실행한 뒤, 7번째 메인 설정인 `POEMetric reranking`을 위해 `sonnet_project/scripts/run_dpo_reranking.py`도 실행한다. 개선 실험은 루트 `sonnet_generation.py`가 아니라 `sonnet_project/scripts/sonnet_generation_enhanced.py`를 사용한다.

GPU 번호를 지정하려면 환경 변수로 넘긴다.

```bash
GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh
```

기존 산출물이 있어도 다시 실행하려면 다음처럼 실행한다.

```bash
FORCE=1 GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh
```

## Python 실행 파일 지정

특정 conda 환경의 Python을 쓰려면 `PYTHON_BIN`을 지정한다.

```bash
PYTHON_BIN=/home/msko021220/.conda/envs/busi2/bin/python \
  bash task_scripts/01_run_gpt2_code_completion.sh
```

## 실행 전 명령만 확인

긴 학습을 바로 시작하지 않고 어떤 명령이 실행될지만 보려면 `--dry-run`을 사용한다.

```bash
bash task_scripts/02_run_sonnet_generation_baseline.sh --dry-run --use_gpu
GPUS=0,1,2,3 bash task_scripts/03_run_sonnet_generation_improvements.sh --dry-run
```
