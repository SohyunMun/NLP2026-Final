# DPO-only rerun summary

## What changed

- Policy/ref model initialization: started both from the best MSK SFT checkpoint.
- Reference model: frozen.
- Policy model: trained only LoRA adapters.
- DPO pairs: original sonnet continuation as chosen; mismatched/corrupted continuations as rejected.
- Training: 10 epochs, batch size 4, learning rate 2e-4, beta 0.05.
- Model selection: saved both best DPO validation-loss checkpoint and best dev chrF checkpoint.
- Reported score: dev chrF from the best dev chrF checkpoint.

## Main results

| data group | start SFT checkpoint | best dev chrF | best chrF epoch | best DPO val-loss epoch | note |
|---|---|---:|---:|---:|---|
| basic | `trial10_best_loss_4gpu/basic__msk_sft_weighted_mbr/best_10-1e-05-sonnet.pt` | 40.7847 | 4 | 9 | Did not beat the previous MSK SFT result. |
| basic_plus_extra | `trial10_best_loss_4gpu/basic_plus_extra__msk_sft_weighted_mbr/best_10-1e-05-sonnet.pt` | 42.5315 | 3 | 3 | Best result among the checked DPO/SFT runs so far. |

## Comparison with previous trial10 references

| condition | method | dev chrF |
|---|---|---:|
| basic | previous MSK SFT + weighted loss + MBR/rerank | 41.2564 |
| basic | previous HUJ DPO | 40.7974 |
| basic | new MSK SFT-initialized LoRA-DPO | 40.7847 |
| basic_plus_extra | new MSK SFT-initialized LoRA-DPO | 42.5315 |
| basic_plus_extra | previous MSK SFT + weighted loss + MBR/rerank | 41.4811 |
| basic_plus_extra | previous HUJ LoRA + DPO | 40.0744 |
| basic_plus_extra | previous HUJ LoRA + Prefix PEFT + DPO | 39.6026 |

## Output paths

- Basic summary: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic/SUMMARY.md`
- Basic best checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic/best_chrf_lora_dpo.pt`
- Basic dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic/predictions/dev_best_chrf.txt`
- Basic test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic/predictions/test_best_chrf.txt`
- Basic+extra summary: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic_plus_extra/SUMMARY.md`
- Basic+extra best checkpoint: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic_plus_extra/best_chrf_lora_dpo.pt`
- Basic+extra dev prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic_plus_extra/predictions/dev_best_chrf.txt`
- Basic+extra test prediction: `/home/msko021220/nlp2026-final-MSK/experiments/dpo_msk_sft_lora_10epoch/basic_plus_extra/predictions/test_best_chrf.txt`

## Interpretation

The modified DPO setup helps most when extra training data is included. In the basic-only setting, the DPO preference signal is too small or too synthetic to improve over the already strong SFT model. In the basic+extra setting, the same DPO objective appears to refine the SFT model toward more benchmark-aligned sonnet continuations, improving dev chrF by +1.0504 over the previous MSK SFT result.
