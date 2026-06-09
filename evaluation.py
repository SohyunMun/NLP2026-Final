# !/usr/bin/env python3

"""
Quora paraphrase detection을 위한 평가.

model_eval_paraphrase: 레이블 정보가 있는 dev 및 train dataloader에 적합함.
model_test_paraphrase: 레이블 정보가 없는 test dataloader에 적합.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids, labels = batch['token_ids'], batch['attention_mask'], batch['sent_ids'], batch[
      'labels'].flatten()

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()  # Character n-gram F-score

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)


import pronouncing
import re

# 셰익스피어 고어 처리를 위한 수동 발음 매핑 사전
GORE_PRONUNCIATION = {
    'thee': 'DH IY1',
    'thine': 'DH AY1 N',
    'dost': 'D AH1 S T',
    'hath': 'HH AE1 TH',
    'art': 'AA1 R T',
    'shalt': 'SH AE1 L T',
    'thou': 'DH AW1',
    'thy': 'DH AY1',
    'the': 'DH AH0',
    'ye': 'Y IY1'
}

def get_word_phones(word):
    word = word.lower().strip(".,;:!?-\"()'")
    if not word:
        return None
    # 1. 고어 사전 확인
    if word in GORE_PRONUNCIATION:
        return GORE_PRONUNCIATION[word]
    # 2. CMU 사전 확인
    phones_list = pronouncing.phones_for_word(word)
    if phones_list:
        return phones_list[0]
    return None

def count_syllables_and_stress(line):
    # 구두점 제거 후 단어 토큰화
    words = re.findall(r"\b\w+(?:'\w+)?\b", line)
    syllables = 0
    stress_pattern = []
    
    for word in words:
        phones = get_word_phones(word)
        if phones:
            stresses = [int(char) for char in phones if char.isdigit()]
            syllables += len(stresses)
            stress_pattern.extend(stresses)
        else:
            # 사전에 없는 경우 모음 갯수로 추정
            vowels = len(re.findall(r"[aeiouyAEIOUY]+", word))
            vowels = max(1, vowels)
            syllables += vowels
            stress_pattern.extend([0] * vowels)
            
    return syllables, stress_pattern

def check_rhyme(word1, word2):
    word1 = word1.lower().strip(".,;:!?-\"()'")
    word2 = word2.lower().strip(".,;:!?-\"()'")
    if not word1 or not word2:
        return False
    if word1 == word2:
        return True
        
    phones1 = get_word_phones(word1)
    phones2 = get_word_phones(word2)
    
    if phones1 and phones2:
        def get_rime(phones):
            parts = phones.split()
            for i in range(len(parts)-1, -1, -1):
                if any(c.isdigit() for c in parts[i]):
                    return " ".join(parts[i:])
            return parts[-1]
        return get_rime(phones1) == get_rime(phones2)
        
    rhymes_w1 = pronouncing.rhymes(word1)
    if rhymes_w1 and word2 in rhymes_w1:
        return True
    return False

def evaluate_poetic_metrics(sonnet_text):
    """
    개별 소네트 텍스트(14행)를 분석하여 음절 정확도, 강세 패턴(오보격) 정확도, 운율 정확도를 리턴합니다.
    """
    lines = [line.strip() for line in sonnet_text.strip().split('\n') if line.strip()]
    if len(lines) < 14:
        lines = lines + [""] * (14 - len(lines))
    lines = lines[:14]
    
    # 1. 음절 정확도
    syllable_errors = []
    perfect_syllable_lines = 0
    for line in lines:
        if not line:
            syllable_errors.append(10)
            continue
        sc, _ = count_syllables_and_stress(line)
        syllable_errors.append(abs(sc - 10))
        if sc == 10:
            perfect_syllable_lines += 1
    mean_syllable_error = sum(syllable_errors) / 14
    syllable_acc = perfect_syllable_lines / 14
    
    # 2. 강세 정확도 (Meter)
    ideal_pattern = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    stress_accuracies = []
    for line in lines:
        if not line:
            stress_accuracies.append(0.0)
            continue
        _, sp = count_syllables_and_stress(line)
        sp = [1 if x >= 1 else 0 for x in sp]
        min_len = min(len(sp), len(ideal_pattern))
        if min_len == 0:
            stress_accuracies.append(0.0)
            continue
        match_count = sum(1 for a, b in zip(sp[:min_len], ideal_pattern[:min_len]) if a == b)
        len_penalty = abs(len(sp) - len(ideal_pattern))
        acc = max(0.0, (match_count - len_penalty) / len(ideal_pattern))
        stress_accuracies.append(acc)
    meter_acc = sum(stress_accuracies) / 14
    
    # 3. 운율 정확도
    rhyme_pairs = [(0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)]
    matched_rhymes = 0
    for p1, p2 in rhyme_pairs:
        def get_last_word(line):
            words = re.findall(r"\b\w+(?:'\w+)?\b", line)
            return words[-1] if words else None
        
        w1 = get_last_word(lines[p1])
        w2 = get_last_word(lines[p2])
        if w1 and w2 and check_rhyme(w1, w2):
            matched_rhymes += 1
            
    rhyme_acc = matched_rhymes / len(rhyme_pairs)
    
    return {
        "mean_syllable_error": mean_syllable_error,
        "syllable_accuracy": syllable_acc,
        "meter_accuracy": meter_acc,
        "rhyme_accuracy": rhyme_acc
    }