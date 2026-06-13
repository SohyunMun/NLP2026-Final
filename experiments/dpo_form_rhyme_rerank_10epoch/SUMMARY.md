# Form/rhyme-aware DPO rerun summary

## Setup

- Base model: `trial10_best_loss_4gpu/basic_plus_extra__msk_sft_weighted_mbr/best_10-1e-05-sonnet.pt`
- Data group: `basic_plus_extra`
- Policy/ref initialization: both from the MSK SFT checkpoint.
- Reference model: frozen.
- Policy model: LoRA adapters only.
- DPO rejected variants per training sonnet:
  - mismatch continuation
  - repetition-heavy continuation
  - bad-rhyme continuation
  - bad line-length continuation
  - repeated-ending continuation
- Reranking: model score + MBR + line count + line length + rhyme + lexical diversity - repetition penalty.

## Main training result

| metric | value |
|---|---:|
| best dev chrF | 42.3309 |
| best chrF epoch | 1 |
| best DPO validation-loss epoch | 2 |
| total DPO pairs | 3250 |
| train / validation pairs | 2925 / 325 |

## Epoch history

| epoch | train loss | val loss | dev chrF |
|---:|---:|---:|---:|
| 0 | 0.3647 | 0.2441 | 41.9654 |
| 1 | 0.2545 | 0.1907 | 42.3309 |
| 2 | 0.2142 | 0.1849 | 41.2569 |
| 3 | 0.1999 | 0.1852 | 41.0974 |
| 4 | 0.2057 | 0.1939 | 41.2789 |
| 5 | 0.1856 | 0.1860 | 41.9258 |
| 6 | 0.1910 | 0.1942 | 41.3042 |
| 7 | 0.1860 | 0.1883 | 41.5271 |
| 8 | 0.1798 | 0.1987 | 41.2374 |
| 9 | 0.1696 | 0.2081 | 41.7028 |

## Comparison

| model | chrF | BLEU | form/5 | rhyme | MATTR | repetition | non-chrF avg rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| New MSK form/rhyme-aware LoRA-DPO | 42.3309 | 23.83 | 3.376 | 0.2500 | 0.9570 | 0.5884 | 5.70 |
| Previous MSK SFT-initialized LoRA-DPO | 42.5315 | 24.35 | 3.111 | 0.1429 | 0.8966 | 0.5860 | 6.20 |
| Previous MSK SFT + weighted loss + MBR/rerank | 41.4811 | 23.81 | 3.141 | 0.1429 | 0.9508 | 0.6129 | 6.80 |
| HUJ LoRA + DPO | 40.0744 | 22.15 | 4.408 | 0.8095 | 0.8601 | 0.4648 | 6.70 |

## Interpretation

The new model improves the intended non-chrF dimensions over the previous MSK DPO: form score, rhyme score, lexical diversity, and overall non-chrF average rank all improve. The cost is a small drop in chrF from 42.5315 to 42.3309. Longer form-aware DPO training did not help chrF; the best chrF checkpoint appeared early at epoch 1, while later epochs tended to over-optimize the preference objective and reduce gold similarity.

## Output paths

- Training summary: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_form_rhyme_rerank_10epoch/basic_plus_extra/SUMMARY.md`
- Best checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_form_rhyme_rerank_10epoch/basic_plus_extra/best_chrf_lora_dpo_form_rhyme.pt`
- Dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_form_rhyme_rerank_10epoch/basic_plus_extra/predictions/dev_best_chrf.txt`
- Test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_form_rhyme_rerank_10epoch/basic_plus_extra/predictions/test_best_chrf.txt`
- Non-chrF evaluation: `/home/msko021220/nlp2026-final-MSK/experiments/non_chrf_eval_form_rhyme/NON_CHRF_SUMMARY.md`
