#!/bin/bash

# run_experiments.sh
# 6가지 소네트 생성 모델 설계 변형 실행 및 평가 자동화 스크립트

set -e

# 필요한 디렉토리 자동 생성
mkdir -p checkpoints logs predictions data

# 도움말 출력 함수
show_help() {
    echo "Usage: ./run_experiments.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all             Run all 6 experiments sequentially."
    echo "  --run <num>       Run a specific experiment (1 to 6)."
    echo "  --quick           Enable Quick Mode (epochs reduced to 1 for validation)."
    echo "  --gpu             Use GPU for training (adds --use_gpu flag)."
    echo "  --summary         Parse existing logs and print the comparison summary table."
    echo "  --help            Show this help message."
    echo ""
    echo "Experiments:"
    echo "  1: sonnet_baseline - Default Dataset"
    echo "  2: sonnet_dpo - Default Dataset"
    echo "  3: sonnet_lora_dpo - Default Dataset"
    echo "  4: sonnet_peft_dpo - Default Dataset"
    echo "  5: sonnet_DAPT_LORA_PEPT_DPO - Default + Shakespeare Dataset (DAPT)"
    echo "  6: sonnet_baseline - Default + Shakespeare Dataset"
}

# 기본 변수 설정
RUN_ALL=false
RUN_NUM=0
QUICK=false
USE_GPU_FLAG=""
SHOW_SUMMARY_ONLY=false

# 아규먼트 파싱
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all) RUN_ALL=true ;;
        --run) RUN_NUM="$2"; shift ;;
        --quick) QUICK=true ;;
        --gpu) USE_GPU_FLAG="--use_gpu" ;;
        --summary) SHOW_SUMMARY_ONLY=true ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# 요약 테이블 파싱용 파이썬 코드 정의
print_summary() {
    python3 -c '
import re, glob, os
logs = sorted(glob.glob("logs/*.log"))
if not logs:
    print("No log files found in logs/ directory.")
    exit()

headers = ["Experiment", "chrF Score", "Pass Rate", "POEMetric", "Form Acc", "Lexical Div", "Quality"]
rows = []
for log_path in logs:
    name = os.path.basename(log_path).replace(".log", "")
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    chrf = re.search(r"chrF Score:\s*([0-9.]+)", content)
    pass_rate = re.search(r"Pass Rate\):\s*([0-9.%]+)", content)
    poem = re.search(r"POEMetric Score:\s*([0-9.]+)", content)
    form = re.search(r"Form Accuracy:\s*([0-9.]+)", content)
    lex = re.search(r"Lexical Diversity:\s*([0-9.]+)", content)
    qual = re.search(r"Overall Quality:\s*([0-9.]+)", content)
    
    chrf_val = chrf.group(1) if chrf else "N/A"
    pass_val = pass_rate.group(1) if pass_rate else "N/A"
    poem_val = poem.group(1) if poem else "N/A"
    form_val = form.group(1) if form else "N/A"
    lex_val = lex.group(1) if lex else "N/A"
    qual_val = qual.group(1) if qual else "N/A"
    
    rows.append([name, chrf_val, pass_val, poem_val, form_val, lex_val, qual_val])

summary_md = []
summary_md.append("# Experiment Evaluation Summary Report\n")
summary_md.append("| " + " | ".join(headers) + " |")
summary_md.append("|" + "|".join(["---"] * len(headers)) + "|")
for r in rows:
    summary_md.append("| " + " | ".join(r) + " |")

md_content = "\n".join(summary_md)

print("\n==========================================================================================")
print("                            EXPERIMENT COMPARISON SUMMARY                                 ")
print("==========================================================================================")
for line in summary_md[1:]:
    print(line)
print("==========================================================================================\n")

os.makedirs("predictions", exist_ok=True)
with open("predictions/evaluation_summary.md", "w", encoding="utf-8") as f:
    f.write(md_content)
print(">>> Summary report saved to predictions/evaluation_summary.md")
'
}

if [ "$SHOW_SUMMARY_ONLY" = true ]; then
    print_summary
    exit 0
fi

# 실행할 모드가 지정되지 않은 경우
if [ "$RUN_ALL" = false ] && [ "$RUN_NUM" -eq 0 ]; then
    echo "Error: Please specify either --all or --run <num>."
    show_help
    exit 1
fi

# 기본 하이퍼파라미터 (Epochs) 설정
EPOCHS_BASE=10
EPOCHS_DPO=10
EPOCHS_LORA=10
EPOCHS_PEFT=10
EPOCHS_DAPT_STAGE=5
EPOCHS_SFT_STAGE=5
EPOCHS_DPO_STAGE=5

if [ "$QUICK" = true ]; then
    echo ">>> [Quick Mode] Training epochs restricted to 1."
    EPOCHS_BASE=1
    EPOCHS_DPO=1
    EPOCHS_LORA=1
    EPOCHS_PEFT=1
    EPOCHS_DAPT_STAGE=1
    EPOCHS_SFT_STAGE=1
    EPOCHS_DPO_STAGE=1
fi

# 데이터 및 리소스 사전 검증
prepare_data() {
    echo ">>> Preparing Datasets..."
    # 1. 셰익스피어 데이터 확인 및 필요시 다운로드
    if [ ! -f "data/shakespeare_plays.txt" ]; then
        echo ">>> shakespeare_plays.txt not found. Downloading..."
        curl -o data/shakespeare_plays.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    fi

    # 2. 소네트 데이터셋 존재 검증
    if [ ! -f "data/sonnets.txt" ]; then
        echo "Error: data/sonnets.txt is missing! Baseline data is required to run training."
        exit 1
    fi

    # 3. 6번 실험을 위한 결합 데이터셋 (Sonnets + Shakespeare) 생성
    echo ">>> Merging sonnets and Shakespeare dataset for Experiment 6..."
    cat data/sonnets.txt data/shakespeare_plays.txt > data/sonnets_and_shakespeare.txt
}

prepare_data

# [실험 1] sonnet_baseline (기본 데이터)
run_exp_1() {
    echo ""
    echo "===================================================================="
    echo " [Experiment 1] Running sonnet_baseline (Default Dataset)"
    echo "===================================================================="
    python3 sonnet_baseline.py --epochs $EPOCHS_BASE --batch_size 8 --sonnet_out predictions/1_baseline_default.txt $USE_GPU_FLAG 2>&1 | tee logs/1_baseline_default.log
    
    CP_FILE="best_${EPOCHS_BASE}-1e-05-sonnet-baseline.pt"
    if [ -f "$CP_FILE" ]; then
        mv "$CP_FILE" checkpoints/best_baseline_default.pt
        echo ">>> Checkpoint backed up to checkpoints/best_baseline_default.pt"
    else
        echo "Warning: Checkpoint $CP_FILE not found."
    fi
}

# [실험 2] sonnet_dpo (기본 데이터)
run_exp_2() {
    echo ""
    echo "===================================================================="
    echo " [Experiment 2] Running sonnet_dpo (Default Dataset)"
    echo "===================================================================="
    python3 sonnet_dpo.py --epochs $EPOCHS_DPO --batch_size 4 --sonnet_out predictions/2_dpo_only.txt $USE_GPU_FLAG 2>&1 | tee logs/2_dpo_only.log
    
    CP_FILE="best_${EPOCHS_DPO}-5e-06-sonnet-dpo.pt"
    if [ -f "$CP_FILE" ]; then
        mv "$CP_FILE" checkpoints/best_dpo_only.pt
        echo ">>> Checkpoint backed up to checkpoints/best_dpo_only.pt"
    else
        echo "Warning: Checkpoint $CP_FILE not found."
    fi
}

# [실험 3] sonnet_lora_dpo (기본 데이터)
run_exp_3() {
    echo ""
    echo "===================================================================="
    echo " [Experiment 3] Running sonnet_lora_dpo (Default Dataset)"
    echo "===================================================================="
    python3 sonnet_lora_dpo.py --epochs $EPOCHS_LORA --batch_size 4 --sonnet_out predictions/3_lora_dpo.txt $USE_GPU_FLAG 2>&1 | tee logs/3_lora_dpo.log
    
    CP_FILE="best_${EPOCHS_LORA}-0.0001-sonnet-lora-dpo.pt"
    if [ -f "$CP_FILE" ]; then
        mv "$CP_FILE" checkpoints/best_lora_dpo.pt
        echo ">>> Checkpoint backed up to checkpoints/best_lora_dpo.pt"
    else
        echo "Warning: Checkpoint $CP_FILE not found."
    fi
}

# [실험 4] sonnet_peft_dpo (기본 데이터)
run_exp_4() {
    echo ""
    echo "===================================================================="
    echo " [Experiment 4] Running sonnet_peft_dpo (Default Dataset)"
    echo "===================================================================="
    python3 sonnet_peft_dpo.py --epochs $EPOCHS_PEFT --batch_size 4 --sonnet_out predictions/4_peft_dpo.txt $USE_GPU_FLAG 2>&1 | tee logs/4_peft_dpo.log
    
    CP_FILE="best_${EPOCHS_PEFT}-5e-05-sonnet-peft-dpo.pt"
    if [ -f "$CP_FILE" ]; then
        mv "$CP_FILE" checkpoints/best_peft_dpo.pt
        echo ">>> Checkpoint backed up to checkpoints/best_peft_dpo.pt"
    else
        echo "Warning: Checkpoint $CP_FILE not found."
    fi
}

# [실험 5] sonnet_DAPT_LORA_PEPT_DPO (기본 + 셰익스피어 추가 데이터)
run_exp_5() {
    echo ""
    echo "===================================================================="
    echo " [Experiment 5] Running sonnet_DAPT_LORA_PEPT_DPO (3-Stage Ultimate)"
    echo "===================================================================="
    python3 sonnet_DAPT_LORA_PEPT_DPO.py --stage all \
        --epochs_dapt $EPOCHS_DAPT_STAGE \
        --epochs_sft $EPOCHS_SFT_STAGE \
        --epochs_dpo $EPOCHS_DPO_STAGE \
        --batch_size 4 \
        --sonnet_out predictions/5_dapt_lora_peft_dpo.txt \
        $USE_GPU_FLAG 2>&1 | tee logs/5_dapt_lora_peft_dpo.log
        
    # 체크포인트 백업
    if [ -f "best_ultimate_dapt.pt" ]; then mv "best_ultimate_dapt.pt" checkpoints/best_ultimate_dapt.pt; fi
    if [ -f "best_ultimate_sft.pt" ]; then mv "best_ultimate_sft.pt" checkpoints/best_ultimate_sft.pt; fi
    if [ -f "best_ultimate_dpo.pt" ]; then 
        mv "best_ultimate_dpo.pt" checkpoints/best_ultimate_dpo.pt
        echo ">>> Ultimate DPO Checkpoint backed up to checkpoints/best_ultimate_dpo.pt"
    fi
}

# [실험 6] sonnet_baseline (기본 + 셰익스피어 추가 데이터)
run_exp_6() {
    echo ""
    echo "===================================================================="
    echo " [Experiment 6] Running sonnet_baseline (Default + Shakespeare)"
    echo "===================================================================="
    python3 sonnet_baseline.py --epochs $EPOCHS_BASE --batch_size 8 \
        --sonnet_path data/sonnets_and_shakespeare.txt \
        --sonnet_out predictions/6_baseline_with_shakespeare.txt \
        $USE_GPU_FLAG 2>&1 | tee logs/6_baseline_with_shakespeare.log
        
    CP_FILE="best_${EPOCHS_BASE}-1e-05-sonnet-baseline.pt"
    if [ -f "$CP_FILE" ]; then
        mv "$CP_FILE" checkpoints/best_baseline_with_shakespeare.pt
        echo ">>> Checkpoint backed up to checkpoints/best_baseline_with_shakespeare.pt"
    else
        echo "Warning: Checkpoint $CP_FILE not found."
    fi
}

# 전체 시나리오 실행 제어
if [ "$RUN_ALL" = true ]; then
    run_exp_1
    run_exp_2
    run_exp_3
    run_exp_4
    run_exp_5
    run_exp_6
    print_summary
else
    case $RUN_NUM in
        1) run_exp_1 ;;
        2) run_exp_2 ;;
        3) run_exp_3 ;;
        4) run_exp_4 ;;
        5) run_exp_5 ;;
        6) run_exp_6 ;;
        *) echo "Error: Invalid experiment number: $RUN_NUM. Please choose a value between 1 and 6." ; exit 1 ;;
    esac
    print_summary
fi
