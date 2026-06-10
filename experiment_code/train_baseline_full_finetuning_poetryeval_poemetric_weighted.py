#!/usr/bin/env python3
from run_utils import parse_common_args, run_python

args = parse_common_args("Baseline full fine-tuning with poetryeval_poemetric weighted sampling, epoch 75.")
run_python(args.project_root, "sonnet_generation.py", [
    "--variation", "baseline",
    "--model_size", "gpt2",
    "--sonnet_path", "github_ready/data/sonnets_train_poetryeval_poemetric_assignment4x.txt",
    "--epochs", "75",
    "--run_name", "baseline_poetryeval_poemetric_weighted_reproduce",
    "--milestone_epochs", "75",
    "--sonnet_out", "predictions/baseline_poetryeval_poemetric_weighted_reproduce_predictions.txt",
], args.gpu, args.no_gpu, args.dry_run)
