# Sonnet Generation: Six-Way Training and Evaluation Report

## 1. 프로젝트 목표

본 프로젝트의 목표는 Shakespeare-style sonnet generation에서 학습 방식과 데이터 구성이 생성 품질에 미치는 영향을 같은 데이터와 같은 평가 지표로 비교하는 것이다. 비교 대상은 기본 GPT-2 fine-tuning, 추가 데이터 사용, SFT, DAPT, LoRA-SFT, LoRA-DPO이다.

최종 비교는 아래 세 평가 축을 사용한다.

| 평가 축 | 목적 |
|---|---|
| chrF | gold reference와 문자 n-gram 단위로 얼마나 비슷한지 측정 |
| Sonnet-or-Not, Bot? proxy | 생성물이 엄격한 sonnet 형식 조건을 통과하는지 측정 |
| POEMetric proxy | form, lexical diversity, overall quality, theme overlap을 결합해 시적 품질을 측정 |

## 2. 데이터

### 2.1 사용 데이터셋

Sonnet generation 전용 데이터는 `sonnet_project/data/` 아래에 canonical layout으로 정리했다.

| group | split | file | count |
|---|---|---|---:|
| basic | train | `sonnet_project/data/basic/train_131.txt` | 131 |
| basic | dev prompts | `sonnet_project/data/basic/dev_prompts_12.txt` | 12 |
| basic | dev gold | `sonnet_project/data/basic/dev_gold_12.txt` | 12 |
| basic | test prompts | `sonnet_project/data/basic/test_prompts_12.txt` | 12 |
| strict_497 | official train | `sonnet_project/data/strict_497/official_train_131.txt` | 131 |
| strict_497 | strict extra train | `sonnet_project/data/strict_497/extra_train_strict_497.txt` | 497 |
| strict_497 | combined train | `sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt` | 628 |
| strict_497 | dev prompts | `sonnet_project/data/strict_497/dev_prompts_12.txt` | 12 |
| strict_497 | dev gold | `sonnet_project/data/strict_497/dev_gold_12.txt` | 12 |
| strict_497 | test prompts | `sonnet_project/data/strict_497/test_prompts_12.txt` | 12 |

`test_prompts_12.txt`에는 prompt만 있고 gold reference가 없다. 따라서 test에서는 chrF를 계산하지 않고, Sonnet-or-Not과 POEMetric 계열 proxy만 계산했다.

### 2.2 추가 데이터 정제

초기 external sonnet 후보는 519개였다. 최종 실험에는 leakage와 중복 가능성을 제거한 strict extra 497개만 사용했다.

| 정제 단계 | 제거 수 | 설명 |
|---|---:|---|
| held-out line overlap 제거 | 4 | dev/test prompt 또는 dev gold와 line-level overlap이 있는 block 제거 |
| official-train duplicate 제거 | 18 | punctuation/case normalization 후 official train과 중복되는 block 제거 |
| 최종 extra | 497 | strict extra train으로 사용 |

정제 기록은 `sonnet_project/data/docs/strict_497_manifest.md`에 저장되어 있다.

### 2.3 관련 참고 문헌과 데이터 출처

- William Shakespeare, *Shakespeare's Sonnets*, Project Gutenberg eBook No. 1041. Project Gutenberg는 해당 eBook을 미국 내 public domain으로 제공한다: https://www.gutenberg.org/ebooks/1041
- Walsh, Preus, Antoniak. 2024. "Sonnet or Not, Bot? Poetry Evaluation for Large Models and Datasets." Findings of EMNLP 2024. 이 연구는 fixed poetic form 인식과 form-aware poetry evaluation의 참고 문헌으로 사용했다: https://aclanthology.org/2024.findings-emnlp.914/
- Li, Wang, Wilkinson. 2026. "POEMetric: The Last Stanza of Humanity." ICLR 2026. 본 프로젝트의 POEMetric proxy는 form, lexical diversity, theme, quality를 함께 보는 관점에서 이 연구를 참고했다: https://openreview.net/forum?id=9VkJ058cTa

주의: 본 프로젝트의 `Sonnet-or-Not`과 `POEMetric`은 논문 공식 evaluator가 아니라, 로컬에서 재현 가능하도록 구현한 rule-based proxy이다.

## 3. 평가 방법

평가 스크립트는 `sonnet_project/scripts/evaluate_sonnet_poemetric.py`이다. 이 스크립트는 CPU-only로 동작하며, 생성 결과 텍스트, prompt, gold reference를 읽어 지표를 계산한다.

### 3.1 chrF

`chrF`는 `sacreBLEU`의 기본 character n-gram F-score이다. gold reference가 있는 dev set에서만 계산한다. test set은 gold reference가 없으므로 chrF를 비워 둔다.

### 3.2 Sonnet-or-Not, Bot? proxy

Sonnet-or-Not pass는 아래 조건을 모두 만족할 때 1, 아니면 0이다. 최종 값은 12개 sample의 pass rate이다.

| 조건 | threshold |
|---|---:|
| exact 14 lines | 1.00 |
| line length score | >= 0.50 |
| Shakespearean rhyme pair score | >= 0.25 |
| final couplet rhyme | >= 0.25 |
| form accuracy | >= 0.70 |

### 3.3 POEMetric proxy

POEMetric proxy는 아래와 같이 계산한다.

```text
POEMetric
= 0.30 * form accuracy
+ 0.25 * lexical diversity
+ 0.30 * overall quality proxy
+ 0.15 * theme overlap
```

하위 지표는 다음과 같다.

```text
form accuracy
= 0.35 * exact 14 lines
+ 0.20 * line length score
+ 0.30 * Shakespearean rhyme pair score
+ 0.15 * final couplet rhyme

lexical diversity
= 0.50 * MATTR
+ 0.50 * distinct-2

overall quality proxy
= 0.35 * form accuracy
+ 0.25 * lexical diversity
+ 0.25 * non-repetition
+ 0.15 * imagery/literary-device lexicon score
```

즉 form과 theme은 POEMetric의 하위 요소이다. 따라서 결과 해석에서는 chrF와 POEMetric을 구분하고, POEMetric 상승 원인을 form, lexical diversity, overall quality, theme으로 분해해 본다.

## 4. 실험 세부 정보

### 4.1 공통 설정

| 항목 | 값 |
|---|---|
| base architecture | course GPT-2 implementation, `model_size=gpt2` |
| optimizer | AdamW |
| batch size | 8 |
| primary dev split | `sonnet_project/data/strict_497/dev_prompts_12.txt` |
| dev gold | `sonnet_project/data/strict_497/dev_gold_12.txt` |
| test split | `sonnet_project/data/strict_497/test_prompts_12.txt` |
| hardware | GPU 0,1,2,3 parallel scheduling |
| final runner | `sonnet_project/scripts/run_sixway_sonnet_ablation.py` |

### 4.2 Six-way experiment matrix

| run name | 설정 | train data | epoch | learning rate | generation/rerank |
|---|---|---|---:|---:|---|
| `base_basic` | 기본 GPT-2 full fine-tuning | official train 131 | 10 | 2e-5 | candidates 1, no MBR |
| `base_plus_extra` | 기본 GPT-2 full fine-tuning + extra | official 131 + strict extra 497 | 10 | 2e-5 | candidates 1, no MBR |
| `sft_plus_extra` | prompt-focused SFT | official 131 + strict extra 497 | 10 | 1e-5 | candidates 2, model score 2.0, MBR 4.0 |
| `dapt_plus_extra` | DAPT-style domain adaptation | official 131 + strict extra 497 | 3 | 5e-6 | candidates 2, model score 1.0, MBR 2.0 |
| `selected_lora_plus_extra` | better non-DPO checkpoint + LoRA-SFT | official 131 + strict extra 497 | 10 | 1.5e-4 | LoRA r=8, alpha=16, candidates 4 |
| `dapt_sft_lora_dpo_best_chrf` | DAPT -> SFT checkpoint + LoRA-DPO | official 131 + strict extra 497 | 10 | 1.5e-4 | LoRA r=8, alpha=16, beta=0.05, candidates 6 |

`selected_lora_plus_extra`는 `sft_plus_extra`와 `dapt_plus_extra` 중 dev chrF가 더 높았던 `dapt_plus_extra` checkpoint에서 시작했다.

`dapt_sft_lora_dpo_best_chrf`는 `dapt_sft_intermediate` checkpoint를 DPO의 policy와 reference 초기값으로 모두 사용했다. Reference model은 freeze하고, policy model에만 LoRA adapter를 붙여 DPO를 수행했다.

### 4.3 DPO preference construction

DPO 학습에서는 winner를 실제 sonnet으로 두고, loser를 아래 방식으로 만든다.

| reject type | 설명 |
|---|---|
| mismatch | prompt는 유지하되 다른 sonnet의 continuation을 붙인 reject |
| repetition | winner continuation을 반복해 만든 reject |
| bad_rhyme | rhyme structure가 약한 reject |
| bad_line_length | line length가 불안정한 reject |
| repeated_endings | line ending 반복이 많은 reject |
| short_form | 14행 조건을 만족하지 못하는 reject |

최종 DPO run의 reject count는 각 유형 628개이다.

### 4.4 실행 시간

로그에 absolute timestamp를 별도 기록하지 않았기 때문에 runtime은 관찰 기준의 대략값이다.

| stage | 대략 시간 |
|---|---:|
| first wave: base/basic, base/extra, SFT, DAPT parallel | 약 5-10분 |
| selected LoRA-SFT | 약 20-25분 |
| DAPT -> SFT intermediate | 약 5-10분 |
| final LoRA-DPO | 약 25-30분 |
| 전체 six-way run | 약 1-1.5시간 |

재실행 시 GPU 점유 상태와 generation candidate 수에 따라 시간이 달라진다.

## 5. 정량적 결과

### 5.1 Dev results

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_basic` | 41.8252 | 0.0000 | 0.5365 | 0.9513 | 0.5567 | 0.2267 | 0.5998 |
| `base_plus_extra` | 41.0941 | 0.0000 | 0.5392 | 0.9542 | 0.5726 | 0.2632 | 0.6116 |
| `sft_plus_extra` | 40.4982 | 0.0000 | 0.5358 | 0.9481 | 0.5664 | 0.3231 | 0.6161 |
| `dapt_plus_extra` | 41.1442 | 0.0000 | 0.5510 | 0.9332 | 0.5495 | 0.2450 | 0.6002 |
| `selected_lora_plus_extra` | 41.7313 | 0.0000 | 0.5455 | 0.9422 | 0.5706 | 0.2785 | 0.6122 |
| `dapt_sft_lora_dpo_best_chrf` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |

### 5.2 Test results

Test set은 gold reference가 없어서 chrF를 계산하지 않았다.

| model | Sonnet-or-Not | form | lexical diversity | overall quality | theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|
| `base_basic` | 0.0000 | 0.5524 | 0.9550 | 0.5736 | 0.1880 | 0.6048 |
| `base_plus_extra` | 0.0833 | 0.5512 | 0.9562 | 0.5988 | 0.1422 | 0.6054 |
| `sft_plus_extra` | 0.0000 | 0.5381 | 0.9537 | 0.5737 | 0.3073 | 0.6181 |
| `dapt_plus_extra` | 0.0000 | 0.5431 | 0.9429 | 0.5592 | 0.1927 | 0.5953 |
| `selected_lora_plus_extra` | 0.0000 | 0.5383 | 0.9468 | 0.5717 | 0.3073 | 0.6158 |
| `dapt_sft_lora_dpo_best_chrf` | 0.0000 | 0.5571 | 0.9178 | 0.5455 | 0.3349 | 0.6105 |

## 6. 요소별 ablation 해석

### 6.1 추가 데이터 사용 여부

비교: `base_basic` -> `base_plus_extra`

| metric | 변화 |
|---|---:|
| dev chrF | -0.7311 |
| dev form | +0.0027 |
| dev theme | +0.0365 |
| dev POEMetric | +0.0118 |

추가 데이터는 chrF에는 부정적이었지만 POEMetric에는 긍정적이었다. 이는 extra data가 dev gold와 표면적으로 완전히 같은 분포는 아니지만, 주제 연결과 전체 품질 proxy에는 도움이 되었기 때문으로 해석한다.

### 6.2 SFT 효과

비교: `base_plus_extra` -> `sft_plus_extra`

| metric | 변화 |
|---|---:|
| dev chrF | -0.5959 |
| dev form | -0.0034 |
| dev theme | +0.0599 |
| dev POEMetric | +0.0045 |

SFT는 chrF를 올리지는 못했지만 theme과 POEMetric을 개선했다. 특히 test POEMetric은 `0.6181`로 전체 최고이다. SFT가 "앞 3줄 prompt를 받아 sonnet을 이어 쓰는 과제 형식"을 직접 학습했기 때문으로 볼 수 있다.

### 6.3 DAPT 효과

비교: `base_plus_extra` -> `dapt_plus_extra`

| metric | 변화 |
|---|---:|
| dev chrF | +0.0501 |
| dev form | +0.0118 |
| dev theme | -0.0182 |
| dev POEMetric | -0.0114 |

DAPT는 form과 chrF에는 약간 긍정적이었지만 lexical diversity와 POEMetric에는 부정적이었다. 단독 DAPT는 domain style 적응에는 도움이 되지만 prompt-following과 시적 다양성까지 보장하지는 못했다.

### 6.4 LoRA-SFT 효과

비교: `dapt_plus_extra` -> `selected_lora_plus_extra`

| metric | 변화 |
|---|---:|
| dev chrF | +0.5871 |
| dev form | -0.0055 |
| dev theme | +0.0335 |
| dev POEMetric | +0.0120 |

LoRA-SFT는 DAPT 단독보다 안정적으로 좋았다. 다만 test POEMetric은 `sft_plus_extra`보다 약간 낮았다.

### 6.5 DPO 효과

비교: `dapt_sft_intermediate` -> `dapt_sft_lora_dpo_best_chrf`

| metric | 변화 |
|---|---:|
| dev chrF | +1.8020 |
| dev form | +0.0259 |
| dev theme | +0.0335 |
| dev POEMetric | -0.0011 |

DPO는 chrF와 form을 크게 개선했다. 그러나 lexical diversity와 overall quality가 낮아져 POEMetric 전체 점수는 거의 개선되지 않았다. 즉 DPO는 reference similarity와 형식 보정에는 강하지만, 시적 다양성과 전체 품질에는 trade-off가 있었다.

## 7. 학습 곡선

### 7.1 LoRA-SFT learning curve

`selected_lora_plus_extra`는 epoch 5에서 best dev chrF를 달성했다.

| epoch | train loss | dev chrF |
|---:|---:|---:|
| 0 | 4.5032 | 41.0734 |
| 1 | 4.4487 | 41.3603 |
| 2 | 4.4082 | 41.1378 |
| 3 | 4.3750 | 41.4071 |
| 4 | 4.3541 | 41.3289 |
| 5 | 4.3323 | 41.7313 |
| 6 | 4.3140 | 40.9397 |
| 7 | 4.2976 | 41.3305 |
| 8 | 4.2831 | 41.7111 |
| 9 | 4.2670 | 41.3277 |

Train loss는 계속 감소했지만 dev chrF는 진동했다. 이는 작은 dev set 12개와 generation stochasticity 때문에 train loss와 generation metric이 완전히 일치하지 않기 때문이다.

### 7.2 LoRA-DPO learning curve

`dapt_sft_lora_dpo_best_chrf`는 epoch 8에서 best dev chrF를 달성했다.

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.5235 | 0.4246 | 41.1217 |
| 1 | 0.4382 | 0.3777 | 41.7812 |
| 2 | 0.4119 | 0.3555 | 42.3095 |
| 3 | 0.3840 | 0.3471 | 42.4900 |
| 4 | 0.3644 | 0.3407 | 42.4932 |
| 5 | 0.3685 | 0.3404 | 42.3892 |
| 6 | 0.3471 | 0.3219 | 42.3961 |
| 7 | 0.3385 | 0.3192 | 41.9988 |
| 8 | 0.3384 | 0.3183 | 42.7768 |
| 9 | 0.3321 | 0.3119 | 42.2446 |

DPO val loss는 거의 지속적으로 감소했지만 dev chrF는 epoch 8이 최고였다. 따라서 최종 모델은 best validation loss가 아니라 best dev chrF checkpoint를 사용했다.

## 8. 종합 결론과 다음 단계

### 8.1 종합 결론

| 목적 | 가장 적합한 모델 |
|---|---|
| dev chrF 최대화 | `dapt_sft_lora_dpo_best_chrf` |
| dev/test POEMetric 최대화 | `sft_plus_extra` |
| test theme overlap 최대화 | `dapt_sft_lora_dpo_best_chrf` |
| test Sonnet-or-Not pass | `base_plus_extra`, 단 12개 중 1개 통과 |

이번 결과는 예상과 부분적으로 일치한다. DPO는 preference pair를 통해 form과 reference similarity를 강하게 밀어주므로 chrF가 올라간 것은 예상 가능했다. 반대로 SFT가 POEMetric에서 강한 것도 자연스럽다. SFT는 prompt-to-continuation 과제 형식을 직접 학습하므로 theme과 전반적 안정성이 개선되기 쉽다.

예상보다 나빴던 부분은 Sonnet-or-Not pass이다. DPO가 form accuracy를 올렸음에도 엄격한 pass threshold는 거의 통과하지 못했다. 이는 rhyme detector 기준에서 final couplet과 Shakespearean rhyme pair를 안정적으로 만족하지 못했기 때문이다.

### 8.2 다음 단계

1. Decoding 단계에서 14-line hard constraint를 적용한다.
2. Final couplet rhyme 후보를 별도 생성 또는 reranking한다.
3. DPO pair에 lexical diversity와 non-repetition reward를 추가해 chrF 상승과 POEMetric 하락의 trade-off를 줄인다.
4. Dev set 12개는 작으므로 더 큰 validation split 또는 cross-validation을 사용한다.
5. Test gold가 없는 상태에서는 test chrF를 계산할 수 없으므로, reference-free metric과 human evaluation을 병행한다.

## 9. 주요 파일 구조

```text
nlp2026-final-MSK/
├── README.md
├── data/                         # 원래 과제 데이터
├── models/                       # 기본 GPT-2 구현
├── modules/                      # attention/layer 구현
├── sonnet_generation.py          # 과제 공식 CLI를 유지한 기본 sonnet generation entry
└── sonnet_project/
    ├── README_KO.md
    ├── data/
    │   ├── README.md
    │   ├── basic/
    │   ├── strict_497/
    │   └── docs/
    ├── docs/
    │   ├── EVALUATION_METRICS_GUIDE.md
    │   └── PROJECT_FILE_GUIDE_KO.md
    ├── experiments/
    │   └── sixway_ablation/
    │       ├── SUMMARY.md
    │       ├── runs.json
    │       ├── poemetric_eval/
    │       ├── base_basic/
    │       ├── base_plus_extra/
    │       ├── sft_plus_extra/
    │       ├── dapt_plus_extra/
    │       ├── selected_lora_plus_extra/
    │       └── dapt_sft_lora_dpo_best_chrf/
    ├── reports/
    │   └── SONNET_GENERATION_PROJECT_REPORT_KO.md
    └── scripts/
        ├── sonnet_generation_enhanced.py
        ├── evaluate_sonnet_poemetric.py
        ├── evaluate_sonnet_metrics.py
        ├── run_sixway_sonnet_ablation.py
        ├── generate_sonnet_checkpoint.py
        ├── train_lora_sft_from_checkpoint.py
        ├── run_msk_sft_lora_dpo.py
        └── run_msk_sft_lora_dpo_form_rhyme.py
```
