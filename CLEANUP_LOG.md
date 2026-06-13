# Cleanup Log

정리일: 2026-06-13

## Data Organization

Sonnet generation 전용 데이터는 `sonnet_data/`로 정리했다.

현재 권장 데이터:

- basic group: `sonnet_data/basic/`
- strict extra 497 group: `sonnet_data/strict_497/`

세부 설명은 `sonnet_data/README.md` 참고.

## Removed

명백한 임시/중복 산출물을 삭제했다.

- Python cache directories: `__pycache__/`
- smoke-test experiment directories:
  - `experiments/dpo_sft_lora_smoke/`
  - `experiments/dpo_form_rhyme_smoke/`
  - `experiments/dpo_line_rhyme_smoke/`
- superseded 1-epoch experiment directories:
  - `experiments/trial1/`
  - `experiments/trial1_parallel_4gpu/`
- large checkpoint files under:
  - `experiments/trial10_best_loss_4gpu/**/*.pt`
- legacy non-strict data directories:
  - `trial1_data/basic_plus_extra/`
  - `trial1_data/extra_clean/`
  - `trial1_data/MANIFEST.md`
- superseded evaluation/data-preparation files:
  - `evaluate_non_chrf_metrics.py`
  - `prepare_trial1_data.py`
  - `experiments/non_chrf_eval/`
  - `experiments/non_chrf_eval_form_rhyme/`

## Preserved

재확인과 비교에 필요한 파일은 보존했다.

- current DPO experiment folders:
  - `experiments/dpo_msk_sft_lora_10epoch/`
  - `experiments/dpo_form_rhyme_rerank_10epoch/`
  - `experiments/dpo_line_rhyme_rerank_10epoch/`
- trial10 summary, logs, and predictions:
  - `experiments/trial10_best_loss_4gpu/`
- archived summaries from removed old trials:
  - `experiments/archive_summaries/`
- unified evaluation outputs:
  - `experiments/unified_sonnet_eval/`
- compatibility data mirror:
  - `trial1_data/basic/`
  - `trial1_data/basic_plus_extra_strict/`
  - `trial1_data/extra_strict/`

## Repointed Links

Old experiment `data` symlinks were repointed to the canonical data folders.

- `experiments/trial10_best_loss_4gpu/basic__*/data` -> `sonnet_data/basic/`
- `experiments/trial10_best_loss_4gpu/basic_plus_extra__*/data` -> `sonnet_data/strict_497/`

## Size Change

Before cleanup, the project folder was about 52GB. After cleanup, it is about 3.9GB.
