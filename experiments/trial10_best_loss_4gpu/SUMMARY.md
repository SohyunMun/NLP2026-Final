# Trial 1 Training Summary

| run | group | method | status | selected epoch | loss source | best loss | chrF | seconds |
|---|---|---|---|---:|---|---:|---:|---:|
| basic__huj_baseline_guided | basic | HUJ guided baseline | ok | 9 | val | 4.1120 | 40.9968 | 206.4 |
| basic__huj_dpo | basic | HUJ DPO | ok | 0 | val | 0.0000 | 40.7974 | 257.3 |
| basic__msk_sft_weighted_mbr | basic | MSK SFT + weighted loss + MBR/rerank generation | ok | 8 | train | 4.2540 | 41.2564 | N/A |
| basic__shm_baseline_full_ft | basic | SHM baseline full fine-tuning | ok | 9 | train | 3.8150 | 33.7047 | 266.0 |
| basic_plus_extra__huj_lora_dpo | basic_plus_extra | HUJ LoRA + DPO | ok | 0 | val | 0.0000 | 40.0744 | 352.8 |
| basic_plus_extra__huj_peft_dpo | basic_plus_extra | HUJ LoRA + Prefix PEFT + DPO | ok | 1 | val | 0.0000 | 39.6026 | 317.4 |
| basic_plus_extra__msk_sft_weighted_mbr | basic_plus_extra | MSK SFT + weighted loss + MBR/rerank generation | ok | 9 | train | 4.1370 | 41.4811 | N/A |
| basic_plus_extra__shm_baseline_full_ft | basic_plus_extra | SHM baseline full fine-tuning | ok | 9 | train | 3.5650 | 36.1996 | 570.0 |
| basic_plus_extra__shm_dapt_lora | basic_plus_extra | SHM DAPT + LoRA fine-tuning | ok | 8 | train | 4.0340 | 31.9858 | 501.3 |
| basic_plus_extra__shm_lora | basic_plus_extra | SHM LoRA fine-tuning | ok | 9 | train | 4.0820 | 33.0272 | 375.4 |
| basic_plus_extra__shm_prefix | basic_plus_extra | SHM DAPT + LoRA + Prefix tuning | ok | 9 | train | 4.1040 | 29.3678 | 495.3 |
