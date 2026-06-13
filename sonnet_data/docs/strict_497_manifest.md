# Strict 497 Data Manifest

## Group

### basic_plus_extra_strict
- train: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic_plus_extra_strict/sonnets_train_plus_extra_strict_497.txt`
- external-only train: `/home/msko021220/nlp2026-final-MSK/trial1_data/extra_strict/poetryeval_poemetric_sonnets_strict_497.txt`
- validation prompt: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic_plus_extra_strict/sonnets_held_out_dev.txt`
- validation gold: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic_plus_extra_strict/TRUE_sonnets_held_out_dev.txt`
- test prompt: `/home/msko021220/nlp2026-final-MSK/trial1_data/basic_plus_extra_strict/sonnets_held_out.txt`

## Cleaning Summary

- source external blocks: `519`
- dropped held-out line overlaps: `4` ids `['51', '52', '53', '54']`
- dropped official-train duplicates: `18` ids `['56', '93', '100', '122', '125', '132', '137', '146', '160', '161', '166', '169', '180', '188', '190', '196', '197', '199']`
- strict external blocks: `497`
- combined train blocks: `628`

## Source Hashes

- source extra: `d4179a334cc484fa3b8a91553cb2ac666b91222dfe77978b32aaaa194d67da60`
- official train: `722836995bea43c112897546c7d09a81458b740bb510596f7e81104ca3ab624e`
- dev prompt: `309ae8da81a24de61a261c291c1299f5c6bfbf4e6b7d1d5b50bd15b6c1770afd`
- dev gold: `fb41ee87e70107f554b658c8de07a6ae905298db09da84794dfaa2a61e892a1f`
- test prompt: `41417eafa263e822f1bb9e52fbdbe0ff910982d2f7663231afd3b7d6b3174ae6`

## Notes

- This split preserves the same dev/test files as the earlier groups.
- It removes punctuation-insensitive held-out line overlaps and official-train duplicates from the previous 519 external-sonnet source.
