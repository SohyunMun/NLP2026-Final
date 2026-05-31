"""
dataset.py — 추가 데이터셋 다운로드 및 처리 파이프라인.

Shakespeare 전작 텍스트를 Project Gutenberg에서 다운로드하고
GPT-2 학습용 청크 데이터셋으로 변환한다.

Reference:
  Gururangan et al. (2020) Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.
  ACL 2020. https://arxiv.org/abs/2004.10964
"""

import os
import re
import torch
import requests
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

SHAKESPEARE_URL = "https://www.gutenberg.org/files/100/100-0.txt"
SHAKESPEARE_PATH = "data/shakespeare_complete.txt"

SPENSER_URL = "https://www.gutenberg.org/files/4088/4088-0.txt"
SPENSER_PATH = "data/spenser_amoretti.txt"
COMBINED_PATH = "data/sonnets_combined.txt"


def download_shakespeare(save_path=SHAKESPEARE_PATH):
    """Shakespeare 전작 텍스트를 Project Gutenberg에서 다운로드."""
    if os.path.exists(save_path):
        print(f"[dataset.py] 캐시 파일 사용: {save_path}")
        return save_path

    print("[dataset.py] Shakespeare 전작 다운로드 중...")
    response = requests.get(SHAKESPEARE_URL, timeout=60)
    response.encoding = 'utf-8'
    text = response.text

    # Gutenberg 푸터 제거
    end_marker = "End of the Project Gutenberg"
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    # 빈 줄 정리
    lines = [line for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"[dataset.py] 저장 완료: {save_path} ({size_mb:.1f}MB)")
    return save_path


def download_spenser(save_path=SPENSER_PATH):
    """
    Spenser의 Amoretti (89 sonnets)를 Project Gutenberg에서 다운로드하고
    sonnets.txt와 동일한 형식(숫자 + 14행)으로 파싱하여 저장한다.
    """
    if os.path.exists(save_path):
        return save_path

    print("[dataset.py] Spenser's Amoretti 다운로드 중...")
    response = requests.get(SPENSER_URL, timeout=60)
    response.encoding = 'utf-8'
    text = response.text

    # 'SONNET I.' 형태의 구분자로 분할
    sections = re.split(r'\n\s*SONNET\s+[IVXLCDM]+\.?\s*\n', text)

    sonnets = []
    for section in sections[1:]:
        lines = [l.strip() for l in section.strip().split('\n') if l.strip()]
        # 설명 텍스트나 짧은 섹션 제외, 14행 추출
        text_lines = [l for l in lines if not l.startswith('[') and len(l) > 5]
        if len(text_lines) >= 14:
            sonnets.append('\n'.join(text_lines[:14]))

    # sonnets.txt 호환 형식으로 저장 (번호 200부터 시작)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Amoretti\nby Edmund Spenser\n\n")
        for i, sonnet in enumerate(sonnets, start=200):
            f.write(f"\n{i}\n\n{sonnet}\n")

    print(f"[dataset.py] Spenser Amoretti {len(sonnets)}개 저장: {save_path}")
    return save_path


def get_combined_sonnets_path(shakespeare_path="data/sonnets.txt",
                               combined_path=COMBINED_PATH):
    """Shakespeare 소네트 + Spenser Amoretti를 합쳐 단일 학습 파일로 반환한다."""
    if os.path.exists(combined_path):
        return combined_path

    spenser_path = download_spenser()

    with open(shakespeare_path, 'r', encoding='utf-8') as f:
        shakespeare_text = f.read().rstrip()

    with open(spenser_path, 'r', encoding='utf-8') as f:
        spenser_lines = f.read().split('\n')
    # 헤더 3줄 제거 후 소네트 본문만 추가
    spenser_body = '\n'.join(spenser_lines[3:])

    combined = shakespeare_text + '\n\n' + spenser_body
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"[dataset.py] 결합 데이터셋 저장: {combined_path}")
    return combined_path


class ShakespeareDataset(Dataset):
    """
    Shakespeare 전작 텍스트 데이터셋.

    텍스트를 chunk_size 토큰 단위로 분할하여 GPT-2 언어 모델 학습에 사용한다.
    도메인 적응적 사전학습(Domain-Adaptive Pre-Training, DAPT)을 위한 데이터셋이다.

    Reference:
      Gururangan et al. (2020) Don't Stop Pretraining. https://arxiv.org/abs/2004.10964
    """

    def __init__(self, file_path=None, chunk_size=256):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chunk_size = chunk_size

        if file_path is None:
            file_path = download_shakespeare()

        self.chunks = self._load_and_chunk(file_path)

    def _load_and_chunk(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        token_ids = self.tokenizer.encode(text)
        print(f"[ShakespeareDataset] {len(token_ids):,} 토큰 로드 완료")

        # next-token prediction을 위해 chunk_size+1 길이로 분할
        chunks = [
            token_ids[i:i + self.chunk_size + 1]
            for i in range(0, len(token_ids) - self.chunk_size, self.chunk_size)
        ]
        print(f"[ShakespeareDataset] {len(chunks):,}개 청크 생성 (chunk_size={self.chunk_size})")
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]

    def collate_fn(self, all_data):
        max_len = max(len(c) for c in all_data)
        token_ids, masks = [], []
        for chunk in all_data:
            pad = max_len - len(chunk)
            token_ids.append(chunk + [self.tokenizer.pad_token_id] * pad)
            masks.append([1] * len(chunk) + [0] * pad)
        return {
            'token_ids': torch.LongTensor(token_ids),
            'attention_mask': torch.LongTensor(masks)
        }
