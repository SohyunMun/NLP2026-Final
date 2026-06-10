#!/usr/bin/env python3
from run_utils import parse_common_args, run_python

args = parse_common_args("train/reuse DAPT(160 poetryeval_poemetric) + LoRA model, then rerank candidates.")

checkpoint = "github_ready/top10_models/dapt_lora_poetryeval_poemetric_reranking_base_model.pt" if args.skip_training else "models/dapt_lora_poetryeval_poemetric_reproduce_epoch75.pt"

if not args.skip_training:
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

run_python(args.project_root, "generate_reranked_sonnets.py", [
    "--checkpoint", checkpoint,
    "--out", "predictions/dapt_lora_poetryeval_poemetric_rerank_reproduce_predictions.txt",
    "--num_candidates", "12",
], args.gpu, args.no_gpu, args.dry_run)
