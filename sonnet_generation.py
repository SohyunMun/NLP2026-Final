'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.

Variations:
  baseline : Full fine-tune (스타터 코드 최소 구현)
  lora     : LoRA fine-tune
  dapt     : DAPT + Full fine-tune
  dapt_lora: DAPT + LoRA
  prefix   : DAPT + LoRA + Prefix Tuning

실행:
  python sonnet_generation.py --use_gpu --variation baseline
  python sonnet_generation.py --use_gpu --variation lora
  python sonnet_generation.py --use_gpu --variation dapt
  python sonnet_generation.py --use_gpu --variation dapt_lora
  python sonnet_generation.py --use_gpu --variation prefix

'''

import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from einops import rearrange

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False

# 셰익스피어 소네트 운율 체계 ABAB CDCD EFEF GG (0-indexed)
RHYME_SCHEME = {2: 0, 3: 1, 6: 4, 7: 5, 10: 8, 11: 9, 13: 12}


# ── 재현성 ────────────────────────────────────────────────────────────────────

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ── LoRA [2] ──────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.original = linear
        self.scaling = alpha / r
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))


def _apply_lora(gpt_model, r=8, alpha=16):
    for layer in gpt_model.gpt_layers:
        sa = layer.self_attention
        sa.query = LoRALinear(sa.query, r=r, alpha=alpha)
        sa.key = LoRALinear(sa.key, r=r, alpha=alpha)
        sa.value = LoRALinear(sa.value, r=r, alpha=alpha)
        layer.attention_dense = LoRALinear(layer.attention_dense, r=r, alpha=alpha)


# ── Prefix Tuning [3] ─────────────────────────────────────────────────────────

class PrefixTuning(nn.Module):
    """각 Transformer 레이어 입력에 학습 가능한 soft prefix 토큰을 추가한다 [3].

    prefix_vectors: (num_layers, prefix_len, d_model) — 직접 학습되는 텐서.
    forward pass에서 각 레이어의 hidden state 앞에 concat된다.
    """
    def __init__(self, num_layers: int, d_model: int, prefix_len: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        self.prefix_len = prefix_len
        self.num_layers = num_layers
        # (num_layers, prefix_len, d_model) 직접 학습 가능한 파라미터
        self.prefix_vectors = nn.Parameter(
            torch.randn(num_layers, prefix_len, d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self):
        return self.dropout(self.prefix_vectors)


# ── 모델 ──────────────────────────────────────────────────────────────────────

class SonnetGPT(nn.Module):
    """Sonnet 생성을 위해 설계된 GPT-2 모델."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.variation = getattr(args, 'variation', 'baseline')
        self.prefix_tuning = None

        if self.variation in ('lora', 'dapt_lora', 'prefix'):
            _apply_lora(self.gpt,
                        r=getattr(args, 'lora_r', 8),
                        alpha=getattr(args, 'lora_alpha', 16))

        if self.variation == 'prefix':
            self.prefix_tuning = PrefixTuning(
                num_layers=args.l,
                d_model=args.d,
                prefix_len=getattr(args, 'prefix_len', 20),
            )

        # 기본: 전체 파라미터 학습 가능 (train()에서 freeze 설정)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        if self.prefix_tuning is not None:
            # embed → prefix concat → encode 순서로 수동 forward
            batch_size = input_ids.shape[0]
            prefix_len = self.prefix_tuning.prefix_len

            token_emb = self.gpt.embed(input_ids)             # (B, T, d)
            prefix_vecs = self.prefix_tuning.forward()        # (num_layers, prefix_len, d)
            # 첫 레이어 prefix를 embedding에 concat
            prefix_exp = prefix_vecs[0].unsqueeze(0).expand(batch_size, -1, -1)  # (B, prefix_len, d)
            hidden = torch.cat([prefix_exp, token_emb], dim=1)                   # (B, prefix_len+T, d)

            prefix_mask = torch.ones(batch_size, prefix_len,
                                     dtype=attention_mask.dtype,
                                     device=attention_mask.device)
            extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            hidden = self.gpt.encode(hidden, extended_mask)
            hidden = self.gpt.final_layer_norm(hidden)
            # prefix 부분 제거 후 token logit 반환
            hidden = hidden[:, prefix_len:, :]
        else:
            out = self.gpt(input_ids, attention_mask)
            hidden = out['last_hidden_state']
        return self.gpt.hidden_state_to_token(hidden)

    def get_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.9, top_p=0.9,
                 max_length=300, repetition_penalty=1.3):
        token_ids = encoding.to(self.get_device())
        attn_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

        newline_id = self.tokenizer.encode('\n')[0]
        prompt_newlines = self.tokenizer.decode(token_ids[0].tolist()).count('\n')
        remaining_lines = max(13 - prompt_newlines, 1)
        generated_newlines = 0
        prompt_len = token_ids.shape[1]

        for _ in range(max_length):
            logits = self.forward(token_ids, attn_mask)
            logits_last = logits[:, -1, :].clone() / temperature

            # repetition penalty
            if repetition_penalty != 1.0:
                for tid in set(token_ids[0].tolist()):
                    if logits_last[0, tid] < 0:
                        logits_last[0, tid] *= repetition_penalty
                    else:
                        logits_last[0, tid] /= repetition_penalty

            probs = F.softmax(logits_last, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = True
            filtered = sorted_probs * mask
            filtered /= filtered.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(filtered, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_idx)

            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1)

            if sampled_token.item() == newline_id:
                generated_newlines += 1
                if generated_newlines >= remaining_lines:
                    break

        generated_tokens = token_ids[0][prompt_len:]
        return token_ids, self.tokenizer.decode(generated_tokens.cpu().tolist())


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, args, filepath,
                    epoch, best_loss, patience_counter):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'args': args,
        'epoch': epoch,
        'best_loss': best_loss,
        'patience_counter': patience_counter,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }, filepath)


def save_best(model, optimizer, args, filepath):
    torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'args': args}, filepath)



class TextChunkDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, chunk_size=256, tokenizer=None):
        tok = tokenizer or GPT2Tokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        ids = tok.encode(text)
        self.chunks = [
            torch.tensor(ids[i:i + chunk_size], dtype=torch.long)
            for i in range(0, len(ids) - chunk_size, chunk_size)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    def collate_fn(self, batch):
        max_len = max(b.shape[0] for b in batch)
        padded, masks = [], []
        for b in batch:
            pad_len = max_len - b.shape[0]
            padded.append(torch.cat([b, torch.zeros(pad_len, dtype=torch.long)]))
            masks.append(torch.cat([torch.ones(b.shape[0]), torch.zeros(pad_len)]).long())
        return {'token_ids': torch.stack(padded), 'attention_mask': torch.stack(masks)}

# ── DAPT [1] ─────────────────────────────────────────────────────────────────

def pretrain_shakespeare(args, model, device):
    """Stage 1: DAPT on Shakespeare or a supplied domain corpus."""
    from dataset import ShakespeareDataset
    ckpt_path = f'{args.variation}_stage1_{args.filepath}'

    if getattr(args, 'pretrain_corpus_path', None):
        ds = TextChunkDataset(args.pretrain_corpus_path, chunk_size=args.chunk_size)
    else:
        ds = ShakespeareDataset(chunk_size=args.chunk_size)
    dl = DataLoader(ds, shuffle=True, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    optimizer = AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=0.01)
    total_steps = len(dl) * args.pretrain_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        if start_epoch >= args.pretrain_epochs:
            print("Stage 1 이미 완료.\n")
            return model

    print(f"\n=== Stage 1: DAPT ({args.model_size}) — {len(ds):,}청크, {args.pretrain_epochs}epoch ===")
    for epoch in range(start_epoch, args.pretrain_epochs):
        model.train()
        total_loss, n = 0, 0
        for batch in tqdm(dl, desc=f'pretrain-{epoch}', disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
            labels = b_ids[:, 1:].contiguous().flatten()
            mask = b_mask[:, 1:].contiguous().flatten().bool()
            loss = F.cross_entropy(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n += 1
        print(f"  Epoch {epoch}: pretrain loss = {total_loss / n:.3f}")
        if (epoch + 1) % 5 == 0 or epoch == args.pretrain_epochs - 1:
            save_checkpoint(model, optimizer, scheduler, args, ckpt_path,
                            epoch, total_loss / n, 0)
    print("Stage 1 완료.\n")
    return model



def milestone_epoch_set(args):
    raw = getattr(args, 'milestone_epochs', '') or ''
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(',') if x.strip()}


def milestone_model_path(args, epoch_num):
    run_name = getattr(args, 'run_name', None) or args.variation
    return os.path.join(args.milestone_model_dir, f'{run_name}_epoch{epoch_num}.pt')


def milestone_prediction_path(args, epoch_num):
    run_name = getattr(args, 'run_name', None) or args.variation
    return os.path.join(args.milestone_pred_dir, f'{run_name}_epoch{epoch_num}_generated_sonnets.txt')

# ── Fine-tuning ───────────────────────────────────────────────────────────────

def set_trainable_params(model, variation):
    """variation에 따라 학습할 파라미터를 설정한다."""
    if variation == 'baseline':
        # 전체 파라미터 학습
        for param in model.parameters():
            param.requires_grad = True

    elif variation == 'lora':
        # LoRA 어댑터만 학습
        for name, param in model.named_parameters():
            param.requires_grad = ('lora_A' in name or 'lora_B' in name)

    elif variation == 'dapt':
        # DAPT 후 전체 fine-tune
        for param in model.parameters():
            param.requires_grad = True

    elif variation == 'dapt_lora':
        # DAPT 후 LoRA만 학습
        for name, param in model.named_parameters():
            param.requires_grad = ('lora_A' in name or 'lora_B' in name)

    elif variation == 'prefix':
        # GPT-2 동결, LoRA 어댑터 + Prefix 파라미터만 학습
        for name, param in model.named_parameters():
            param.requires_grad = (
                'lora_A' in name or 'lora_B' in name or 'prefix_tuning' in name
            )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{variation}] 학습 파라미터: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def train(args, model, device):
    """소네트 fine-tuning (early stopping 포함)."""
    ckpt_path = f'{args.variation}_stage2_{args.filepath}'
    best_fp = f'{args.variation}_best_{args.filepath}'

    ds = SonnetsDataset(args.sonnet_path)
    dl = DataLoader(ds, shuffle=True, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    dev_ds = SonnetsDataset(args.dev_sonnet_path)

    set_trainable_params(model, args.variation)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    total_steps = len(dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best_loss, patience_counter, start_epoch = float('inf'), 0, 0

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        if ckpt['scheduler']:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['best_loss']
        patience_counter = ckpt['patience_counter']
        print(f"체크포인트 로드: epoch={start_epoch}, best_loss={best_loss:.3f}")

    print(f"\n=== Stage 2: fine-tuning [{args.variation}] — {len(ds)}소네트, max {args.epochs}epoch ===")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, n = 0, 0
        for batch in tqdm(dl, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
            labels = b_ids[:, 1:].contiguous().flatten()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n += 1

        train_loss /= n
        print(f"Epoch {epoch}: train loss = {train_loss:.3f}")

        # dev 샘플 출력
        model.eval()
        for batch in dev_ds:
            enc = model.tokenizer(batch[1], return_tensors='pt').to(device)
            _, gen = model.generate(enc['input_ids'],
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    repetition_penalty=args.repetition_penalty)
            print(f"{batch[1]}{gen}\n")
            break

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            save_best(model, optimizer, args, best_fp)
            print(f"Best model saved (loss={best_loss:.3f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")

        save_checkpoint(model, optimizer, scheduler, args, ckpt_path,
                        epoch, best_loss, patience_counter)

        epoch_num = epoch + 1
        if epoch_num in milestone_epoch_set(args):
            os.makedirs(args.milestone_model_dir, exist_ok=True)
            milestone_fp = milestone_model_path(args, epoch_num)
            save_best(model, optimizer, args, milestone_fp)
            print(f"Milestone model saved: {milestone_fp}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    return best_fp


# ── 소네트 생성 ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_submission_sonnets(args, best_filepath):
    from evaluation import test_sonnet
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(best_filepath, weights_only=False)

    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    held_ds = SonnetsDataset(args.dev_sonnet_path)
    gold_path = 'data/TRUE_sonnets_held_out_dev.txt'
    out_path = args.sonnet_out
    sonnets = []

    for batch in held_ds:
        enc = model.tokenizer(batch[1], return_tensors='pt').to(device)
        _, gen = model.generate(enc['input_ids'],
                                temperature=args.temperature,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty)
        full = f'{batch[1]}{gen}'
        sonnets.append((batch[0], full))
        print(f"{full}\n")

    with open(out_path, 'w') as f:
        f.write("--Generated Sonnets-- \n\n")
        for sid, txt in sonnets:
            f.write(f"\n{sid}\n{txt}\n")

    try:
        score = test_sonnet(test_path=out_path, gold_path=gold_path)
        print(f"\n[{args.variation}] chrF = {score:.4f}")
    except Exception as e:
        print(f"chrF 계산 실패: {e}")

    return out_path


# ── 인자 파서 ─────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--variation", type=str, default="baseline",
                   choices=["baseline", "lora", "dapt", "dapt_lora", "prefix"],
                   help="실험 variation 선택")

    p.add_argument("--sonnet_path",          default="data/sonnets.txt")
    p.add_argument("--held_out_sonnet_path", default="data/sonnets_held_out.txt")
    p.add_argument("--dev_sonnet_path",      default="data/sonnets_held_out_dev.txt")
    p.add_argument("--sonnet_out",           default=None)
    p.add_argument("--run_name",             default=None,
                   help="Unique run name used for checkpoint/prediction naming")
    p.add_argument("--milestone_epochs",     default="",
                   help="Comma-separated epochs to save and generate, e.g. 25,50,75,100")
    p.add_argument("--milestone_model_dir",  default="models")
    p.add_argument("--milestone_pred_dir",   default="predictions")

    p.add_argument("--seed",    type=int,    default=11711)
    p.add_argument("--use_gpu", action='store_true')
    p.add_argument("--resume",  action='store_true')
    p.add_argument("--init_checkpoint", default=None,
                   help="Optional .pt checkpoint used to initialize model weights before fine-tuning")

    # DAPT
    p.add_argument("--pretrain_epochs", type=int,   default=5)
    p.add_argument("--pretrain_lr",     type=float, default=2e-5)
    p.add_argument("--chunk_size",      type=int,   default=256)
    p.add_argument("--pretrain_corpus_path", default=None)

    # Fine-tuning
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm",type=float, default=1.0)

    # Generation
    p.add_argument("--temperature",         type=float, default=0.9)
    p.add_argument("--top_p",               type=float, default=0.9)
    p.add_argument("--repetition_penalty",  type=float, default=1.3)

    # Model
    p.add_argument("--model_size", choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                   default='gpt2')

    # LoRA
    p.add_argument("--lora_r",     type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    # Prefix Tuning
    p.add_argument("--prefix_len", type=int, default=20)

    return p.parse_args()


def add_arguments(args):
    sizes = {
        'gpt2':        (768,  12, 12),
        'gpt2-medium': (1024, 24, 16),
        'gpt2-large':  (1280, 36, 20),
        'gpt2-xl':     (1600, 48, 25),
    }
    args.d, args.l, args.num_heads = sizes[args.model_size]
    return args


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    args = add_arguments(args)

    if args.sonnet_out is None:
        args.sonnet_out = f'predictions/{args.variation}_generated_sonnets.txt'

    run_label = args.run_name or args.variation
    args.filepath = f'{run_label}_{args.variation}_{args.model_size}_{args.epochs}-{args.lr}.pt'

    seed_everything(args.seed)
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    print(f"=== Variation: {args.variation} | Model: {args.model_size} | Epochs: {args.epochs} ===")

    model = SonnetGPT(args)
    model = model.to(device)

    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt['model'])
        print(f"Initialized model from {args.init_checkpoint}")

    # DAPT가 필요한 variation은 Stage 1 먼저 수행
    if args.variation in ('dapt', 'dapt_lora', 'prefix'):
        model = pretrain_shakespeare(args, model, device)

    best_filepath = train(args, model, device)

    milestones = sorted(milestone_epoch_set(args))
    generated_milestones = False
    for epoch_num in milestones:
        milestone_fp = milestone_model_path(args, epoch_num)
        if os.path.exists(milestone_fp):
            args.sonnet_out = milestone_prediction_path(args, epoch_num)
            os.makedirs(os.path.dirname(args.sonnet_out), exist_ok=True)
            generate_submission_sonnets(args, milestone_fp)
            generated_milestones = True

    if not generated_milestones:
        generate_submission_sonnets(args, best_filepath)
