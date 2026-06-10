'''
셰익스피어 전작 DAPT용 데이터셋.
'''

import os
import re
import urllib.request

import torch
from torch.utils.data import Dataset

GUTENBERG_URL = "https://www.gutenberg.org/files/100/100-0.txt"
LOCAL_PATH = "data/shakespeare_complete.txt"


def download_shakespeare():
    if not os.path.exists(LOCAL_PATH):
        print("셰익스피어 전작 다운로드 중...")
        urllib.request.urlretrieve(GUTENBERG_URL, LOCAL_PATH)
        print(f"저장 완료: {LOCAL_PATH}")
    return LOCAL_PATH


class ShakespeareDataset(Dataset):
    def __init__(self, chunk_size=256, tokenizer=None):
        from transformers import GPT2Tokenizer
        path = download_shakespeare()
        with open(path, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        # Gutenberg 헤더/푸터 제거
        start = text.find("THE SONNETS")
        end = text.rfind("End of the Project Gutenberg")
        if start != -1:
            text = text[start:end if end != -1 else len(text)]

        tok = tokenizer or GPT2Tokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
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
