# DPO + POEMetric Reranking Experiment

이 실험은 새 모델을 학습하지 않고, 기존 best DPO policy에서 prompt당 여러 후보를 생성한 뒤 POEMetric 중심 reference-free metric으로 최종 후보를 다시 선택한 후처리 실험이다.

Gold reference는 reranking 과정에서 사용하지 않고, 최종 dev 평가에서만 `chrF` 계산에 사용했다. 따라서 같은 방식은 gold가 없는 test set에도 적용 가능하다.

## Model

- SFT base checkpoint: `sonnet_project/experiments/sixway_ablation/dapt_sft_intermediate/best_10-1e-05-sonnet.pt`
- DPO checkpoint: `sonnet_project/experiments/sixway_ablation/dapt_sft_lora_dpo_best_chrf/best_chrf_lora_dpo_form_rhyme.pt`

## Reranking Setup

- candidates per prompt: `6`
- decoding strategies: `top_p,top_k`
- temperature: `0.90`
- top_p: `0.92`
- top_k: `60`
- repetition penalty: `1.08`
- no repeat ngram size: `3`
- max generation tokens: `120`

Reranking score:

```text
0.55 * POEMetric
+ 0.15 * form_accuracy
+ 0.20 * Shakespearean rhyme pair score
+ 0.10 * final couplet rhyme
+ 0.10 * prompt-continuation theme overlap
+ 0.05 * lexical diversity
+ 0.05 * non-repetition
+ 0.05 * candidate centrality
+ 0.10 * Sonnet-or-Not pass bonus
```

## Dev Result

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dpo_single` | 42.7768 | 0.0000 | 0.5613 | 0.9194 | 0.5428 | 0.2403 | 0.5971 |
| `poemetric_reranking` | 42.0672 | 0.1667 | 0.5804 | 0.9610 | 0.5731 | 0.3307 | 0.6359 |

## Test Result

Test set에는 gold reference가 없으므로 `chrF`는 계산하지 않았다.

| model | chrF | Sonnet-or-Not | form | lexical diversity | overall quality | Theme | POEMetric |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dpo_single` |  | 0.0000 | 0.5571 | 0.9178 | 0.5455 | 0.3349 | 0.6105 |
| `poemetric_reranking` |  | 0.1667 | 0.5871 | 0.9631 | 0.5922 | 0.3672 | 0.6496 |

## Interpretation

- Reranking은 dev `chrF`를 `42.7768`에서 `42.0672`로 약간 낮췄다.
- 대신 dev/test 모두에서 `Sonnet-or-Not`, form accuracy, lexical diversity, overall quality, theme overlap, POEMetric이 상승했다.
- 특히 test `POEMetric`은 `0.6105`에서 `0.6496`으로 상승했다.
- 따라서 POEMetric reranking은 reference similarity 최고점을 노리는 방식이라기보다, sonnet 형식과 reference-free poetic quality를 강화하는 후처리로 해석하는 것이 적절하다.

## Outputs

| split | file |
|---|---|
| dev prediction | `sonnet_project/experiments/dpo_reranking/predictions/dev_reranked.txt` |
| test prediction | `sonnet_project/experiments/dpo_reranking/predictions/test_reranked.txt` |
| dev candidates | `sonnet_project/experiments/dpo_reranking/candidates/dev_candidates.jsonl` |
| test candidates | `sonnet_project/experiments/dpo_reranking/candidates/test_candidates.jsonl` |
| dev evaluation | `sonnet_project/experiments/dpo_reranking/eval/dev/all_summary_metrics.md` |
| test evaluation | `sonnet_project/experiments/dpo_reranking/eval/test/all_summary_metrics.md` |
