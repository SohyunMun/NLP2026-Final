#!/usr/bin/env python3
from run_utils import parse_common_args, run_python

args = parse_common_args("DAPT on poetryeval_poemetric for 160 epochs, then LoRA fine-tuning on assignment sonnets for 75 epochs.")
run_python(args.project_root, "sonnet_generation.py", [
    "--variation", "dapt_lora",
    "--model_size", "gpt2",
    "--pretrain_corpus_path", "github_ready/data/poetryeval_poemetric_sonnets_567.txt",
    "--pretrain_epochs", "160",
    "--sonnet_path", "data/sonnets.txt",
    "--epochs", "75",
    "--run_name", "dapt_lora_poetryeval_poemetric_reproduce",
    "--milestone_epochs", "75",
    "--sonnet_out", "predictions/dapt_lora_poetryeval_poemetric_reproduce_predictions.txt",
], args.gpu, args.no_gpu, args.dry_run)
