# Strict 497 Data Manifest

## Group

### basic_plus_extra_strict
- train: `sonnet_project/data/strict_497/train_official_131_plus_extra_497_total_628.txt`
- external-only train: `sonnet_project/data/strict_497/extra_train_strict_497.txt`
- official train: `sonnet_project/data/strict_497/official_train_131.txt`
- validation prompt: `sonnet_project/data/strict_497/dev_prompts_12.txt`
- validation gold: `sonnet_project/data/strict_497/dev_gold_12.txt`
- test prompt: `sonnet_project/data/strict_497/test_prompts_12.txt`

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
