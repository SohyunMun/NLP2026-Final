# Trial 1 Training Summary

| run | group | method | status | chrF | seconds |
|---|---|---|---|---:|---:|
| basic__huj_baseline_guided | basic | HUJ guided baseline | ok | 39.7731 | 45.8 |
| basic__huj_dpo | basic | HUJ DPO | ok | 40.0357 | 65.7 |
| basic__msk_sft_weighted_mbr | basic | MSK SFT + weighted loss + MBR/rerank generation | ok | 41.4234 | 114.0 |
| basic__shm_baseline_full_ft | basic | SHM baseline full fine-tuning | ok | 26.4208 | 68.8 |
| basic_plus_extra__huj_lora_dpo | basic_plus_extra | HUJ LoRA + DPO | ok | 39.9761 | 73.4 |
| basic_plus_extra__huj_peft_dpo | basic_plus_extra | HUJ LoRA + Prefix PEFT + DPO | ok | 38.5251 | 69.3 |
| basic_plus_extra__msk_sft_weighted_mbr | basic_plus_extra | MSK SFT + weighted loss + MBR/rerank generation | ok | 40.9529 | 94.4 |
| basic_plus_extra__shm_baseline_full_ft | basic_plus_extra | SHM baseline full fine-tuning | ok | 30.1896 | 74.1 |
| basic_plus_extra__shm_dapt_lora | basic_plus_extra | SHM DAPT + LoRA fine-tuning | ok | 27.7949 | 90.6 |
| basic_plus_extra__shm_lora | basic_plus_extra | SHM LoRA fine-tuning | ok | 27.9739 | 47.9 |
| basic_plus_extra__shm_prefix | basic_plus_extra | SHM DAPT + LoRA + Prefix tuning | ok | 39.0785 | 133.9 |
