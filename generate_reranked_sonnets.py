#!/usr/bin/env python3
import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from datasets import SonnetsDataset
from evaluate_sonnets import sonnet_form_diagnostics, mattr, words
from sonnet_generation import SonnetGPT, seed_everything
from evaluation import test_sonnet

BAD_PATTERNS = re.compile(r"\b(enter|exit|exeunt|scene|act|king|queen|duke|messenger|lord|lady)\b|[\[\]_{}]", re.I)


def proxy_score(poem):
    form = sonnet_form_diagnostics(poem)
    lex = mattr(poem)
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    line_count_penalty = abs(len(lines) - 14) / 14.0
    bad_hits = len(BAD_PATTERNS.findall(poem))
    token_count = max(len(words(poem)), 1)
    bad_penalty = min(0.5, bad_hits / 20.0)
    length_penalty = 0.0
    if token_count < 80:
        length_penalty += (80 - token_count) / 200.0
    if token_count > 220:
        length_penalty += (token_count - 220) / 300.0
    return (
        0.55 * form["form_partial_raw"]
        + 0.30 * lex
        + 0.15 * form["line_count_score"]
        - 0.25 * line_count_penalty
        - bad_penalty
        - length_penalty
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dev_sonnet_path', default='data/sonnets_held_out_dev.txt')
    parser.add_argument('--gold', default='data/TRUE_sonnets_held_out_dev.txt')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--num_candidates', type=int, default=12)
    parser.add_argument('--seed', type=int, default=11711)
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.92)
    parser.add_argument('--repetition_penalty', type=float, default=1.35)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = saved['args']
    model = SonnetGPT(model_args)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    held_ds = SonnetsDataset(args.dev_sonnet_path)
    sonnets = []
    temperatures = [args.temperature, 0.8, 0.9, 1.0, 1.05, 1.1]
    top_ps = [args.top_p, 0.85, 0.9, 0.95]
    penalties = [args.repetition_penalty, 1.2, 1.3, 1.45]

    with torch.no_grad():
        for sid, prompt in held_ds:
            best_text = None
            best_score = -math.inf
            for cand_idx in range(args.num_candidates):
                cand_seed = args.seed + sid * 1009 + cand_idx
                random.seed(cand_seed)
                np.random.seed(cand_seed % (2**32 - 1))
                torch.manual_seed(cand_seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(cand_seed)
                temp = temperatures[cand_idx % len(temperatures)]
                top_p = top_ps[(cand_idx // len(temperatures)) % len(top_ps)]
                penalty = penalties[(cand_idx // (len(temperatures) * len(top_ps))) % len(penalties)]
                enc = model.tokenizer(prompt, return_tensors='pt').to(device)
                _, gen = model.generate(
                    enc['input_ids'],
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=penalty,
                )
                full = f'{prompt}{gen}'
                score = proxy_score(full)
                if score > best_score:
                    best_score = score
                    best_text = full
            sonnets.append((sid, best_text))
            print(f'id={sid} proxy_score={best_score:.4f}')
            print(best_text)
            print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('--Generated Sonnets-- \n\n')
        for sid, txt in sonnets:
            f.write(f'\n{sid}\n{txt}\n')

    try:
        score = test_sonnet(test_path=str(out_path), gold_path=args.gold)
        print(f'chrF = {score:.4f}')
    except Exception as exc:
        print(f'chrF failed: {exc}')


if __name__ == '__main__':
    main()
