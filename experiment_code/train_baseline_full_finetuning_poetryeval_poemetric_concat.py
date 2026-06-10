#!/usr/bin/env python3
from run_utils import parse_common_args, run_python

args = parse_common_args("Baseline full fine-tuning with assignment + poetryeval_poemetric concat, epoch 100.")
run_python(args.project_root, "sonnet_generation.py", [
    "--variation", "baseline",
    "--model_size", "gpt2",
    "--sonnet_path", "github_ready/data/sonnets_train_plus_poetryeval_poemetric_567.txt",
    "--epochs", "100",
    "--run_name", "baseline_poetryeval_poemetric_concat_reproduce",
    "--milestone_epochs", "100",
    "--sonnet_out", "predictions/baseline_poetryeval_poemetric_concat_reproduce_predictions.txt",
], args.gpu, args.no_gpu, args.dry_run)
