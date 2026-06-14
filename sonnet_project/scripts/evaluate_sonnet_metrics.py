#!/usr/bin/env python3
"""Unified sonnet generation evaluator.

The script is intentionally lightweight: it does not load models and does not
touch GPU libraries. It reads numbered sonnet blocks, computes reference
similarity when gold references are available, and always computes form,
diversity, repetition, prompt-faithfulness, POEMetric-style proxy, and leakage
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from sacrebleu.metrics import BLEU, CHRF
except Exception:  # pragma: no cover - fallback keeps the script portable.
    BLEU = None
    CHRF = None

try:
    import pronouncing
except Exception:  # pragma: no cover - rhyme fallback is used instead.
    pronouncing = None


RHYME_PAIRS = (
    (0, 2),    # A
    (1, 3),    # B
    (4, 6),    # C
    (5, 7),    # D
    (8, 10),   # E
    (9, 11),   # F
    (12, 13),  # G, final couplet
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is",
    "it", "its", "me", "my", "no", "nor", "not", "of", "on", "or",
    "our", "she", "so", "that", "the", "thee", "their", "them", "then",
    "there", "they", "this", "thou", "thy", "to", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "yet", "you", "your",
}

THEME_GROUPS = {
    "love": {"love", "loving", "beloved", "lover", "lovers", "dear", "heart", "desire"},
    "beauty": {"beauty", "beautiful", "fair", "sweet", "lovely", "grace", "glory"},
    "time": {"time", "age", "old", "hours", "day", "days", "night", "season", "summer", "winter"},
    "mortality": {"death", "die", "dead", "grave", "tomb", "decay", "wither", "lost"},
    "truth": {"truth", "true", "false", "lie", "lies", "faith", "swear", "forsworn"},
    "poetry": {"verse", "rhyme", "line", "lines", "muse", "song", "praise", "write"},
}

IMAGERY_WORDS = {
    "sun", "moon", "star", "stars", "heaven", "earth", "eye", "eyes",
    "face", "rose", "flower", "flowers", "summer", "winter", "night",
    "day", "light", "shadow", "fire", "water", "sea", "wind", "storm",
    "blood", "gold", "glass", "breath", "scent", "beauty", "spring",
    "rain", "cloud", "sky", "bright", "dark", "black", "white",
}

LITERARY_MARKERS = {
    "like", "as", "than", "image", "shadow", "mirror", "glass", "muse",
    "rhyme", "verse", "song", "metaphor", "simile", "compare", "praise",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def tokenize(text: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    ]


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower()).strip("'")


def normalize_line(line: str) -> str:
    text = line.strip().lower()
    text = re.sub(r"[^a-z0-9'\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def text_from_lines(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def load_numbered_blocks(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^\s*(\d+)\s*$", text))
    if not matches:
        body = text.strip()
        return [("0", body)] if body else []

    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            blocks.append((match.group(1), body))
    return blocks


def align_blocks(
    prediction_blocks: list[tuple[str, str]],
    other_blocks: list[tuple[str, str]] | None,
    mode: str,
) -> list[tuple[str, str] | None]:
    if not other_blocks:
        return [None for _ in prediction_blocks]

    if mode == "auto":
        other_ids = {block_id for block_id, _ in other_blocks}
        matched = sum(1 for block_id, _ in prediction_blocks if block_id in other_ids)
        use_id = matched >= max(1, int(0.8 * len(prediction_blocks)))
    else:
        use_id = mode == "id"

    if use_id:
        by_id = {block_id: body for block_id, body in other_blocks}
        return [
            (block_id, by_id[block_id]) if block_id in by_id else None
            for block_id, _ in prediction_blocks
        ]

    aligned: list[tuple[str, str] | None] = []
    for idx in range(len(prediction_blocks)):
        aligned.append(other_blocks[idx] if idx < len(other_blocks) else None)
    return aligned


def split_prediction(
    pred_body: str,
    prompt_body: str | None,
    prompt_line_count: int,
    prediction_mode: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    prompt_lines = nonempty_lines(prompt_body or "")[:prompt_line_count]
    raw_prediction_lines = nonempty_lines(pred_body)

    if prediction_mode == "continuation":
        full_lines = prompt_lines + raw_prediction_lines
        predicted_prompt_lines = prompt_lines
        continuation_lines = raw_prediction_lines
    else:
        full_lines = raw_prediction_lines
        expected_prompt_count = len(prompt_lines) if prompt_lines else prompt_line_count
        predicted_prompt_lines = full_lines[:expected_prompt_count]
        continuation_lines = full_lines[expected_prompt_count:]

    return full_lines, continuation_lines, prompt_lines, predicted_prompt_lines


def split_gold(
    gold_body: str | None,
    prompt_line_count: int,
) -> tuple[list[str] | None, list[str] | None]:
    if gold_body is None:
        return None, None
    lines = nonempty_lines(gold_body)
    return lines, lines[prompt_line_count:]


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[idx:idx + n]) for idx in range(0, len(tokens) - n + 1))


def char_ngrams(text: str, n: int) -> Counter[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) < n:
        return Counter()
    return Counter(normalized[idx:idx + n] for idx in range(0, len(normalized) - n + 1))


def fallback_corpus_bleu(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    pred_len = sum(len(tokenize(pred)) for pred in preds)
    ref_len = sum(len(tokenize(ref)) for ref in refs)
    if pred_len == 0:
        return 0.0

    precisions = []
    for n in range(1, 5):
        overlap = 0
        total = 0
        for pred, ref in zip(preds, refs):
            pred_counts = ngrams(tokenize(pred), n)
            ref_counts = ngrams(tokenize(ref), n)
            overlap += sum((pred_counts & ref_counts).values())
            total += sum(pred_counts.values())
        precisions.append((overlap + 1.0) / (total + 1.0))

    brevity = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    return 100.0 * brevity * math.exp(sum(math.log(p) for p in precisions) / 4.0)


def fallback_sentence_bleu(pred: str, ref: str) -> float:
    return fallback_corpus_bleu([pred], [ref])


def fallback_corpus_chrf(preds: list[str], refs: list[str], beta: float = 2.0) -> float:
    if not preds:
        return 0.0
    precision_parts = []
    recall_parts = []
    for n in range(1, 7):
        pred_total = 0
        ref_total = 0
        overlap = 0
        for pred, ref in zip(preds, refs):
            pred_counts = char_ngrams(pred, n)
            ref_counts = char_ngrams(ref, n)
            pred_total += sum(pred_counts.values())
            ref_total += sum(ref_counts.values())
            overlap += sum((pred_counts & ref_counts).values())
        precision_parts.append(overlap / pred_total if pred_total else 0.0)
        recall_parts.append(overlap / ref_total if ref_total else 0.0)

    precision = statistics.mean(precision_parts)
    recall = statistics.mean(recall_parts)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    return 100.0 * (1.0 + beta2) * precision * recall / (beta2 * precision + recall)


def fallback_sentence_chrf(pred: str, ref: str) -> float:
    return fallback_corpus_chrf([pred], [ref])


def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    if BLEU is not None:
        return BLEU(effective_order=True).corpus_score(preds, [refs]).score
    return fallback_corpus_bleu(preds, refs)


def sentence_bleu(pred: str, ref: str) -> float:
    if BLEU is not None:
        return BLEU(effective_order=True).sentence_score(pred, [ref]).score
    return fallback_sentence_bleu(pred, ref)


def corpus_chrf(preds: list[str], refs: list[str]) -> float:
    if CHRF is not None:
        return CHRF(word_order=2).corpus_score(preds, [refs]).score
    return fallback_corpus_chrf(preds, refs)


def sentence_chrf(pred: str, ref: str) -> float:
    if CHRF is not None:
        return CHRF(word_order=2).sentence_score(pred, [ref]).score
    return fallback_sentence_chrf(pred, ref)


def lcs_len(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        curr = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    lcs = lcs_len(pred_tokens, ref_tokens)
    if not pred_tokens or not ref_tokens or lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def token_f1(pred: str, ref: str) -> float:
    pred_counts = Counter(tokenize(pred))
    ref_counts = Counter(tokenize(ref))
    overlap = sum((pred_counts & ref_counts).values())
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    if not pred_total or not ref_total or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2.0 * precision * recall / (precision + recall)


def mattr(text: str, window: int = 50) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    scores = []
    for idx in range(0, len(tokens) - window + 1):
        span = tokens[idx:idx + window]
        scores.append(len(set(span)) / window)
    return statistics.mean(scores)


def distinct_n(texts: list[str], n: int) -> float:
    all_ngrams = []
    for text in texts:
        tokens = tokenize(text)
        all_ngrams.extend(tuple(tokens[idx:idx + n]) for idx in range(0, len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def corpus_repetition_rate(texts: list[str]) -> float:
    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def line_count_score(lines: list[str], target_lines: int = 14) -> float:
    if target_lines <= 0:
        return 0.0
    return clamp(1.0 - abs(len(lines) - target_lines) / target_lines)


def line_length_score(lines: list[str]) -> float:
    if not lines:
        return 0.0
    scores = []
    for line in lines[:14]:
        token_count = len(tokenize(line))
        if 8 <= token_count <= 12:
            scores.append(1.0)
        elif 6 <= token_count <= 14:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return statistics.mean(scores) if scores else 0.0


def last_word(line: str) -> str:
    tokens = tokenize(line)
    return normalize_word(tokens[-1]) if tokens else ""


def fallback_rhyme_tail(word: str) -> str:
    word = normalize_word(word)
    if not word:
        return ""
    match = re.search(r"[aeiouy][a-z']*$", word)
    if match:
        return match.group(0)
    return word[-3:]


def pronouncing_rhyme_parts(word: str) -> set[str]:
    if pronouncing is None:
        return set()
    parts = set()
    for phones in pronouncing.phones_for_word(word):
        try:
            part = pronouncing.rhyming_part(phones)
        except Exception:
            part = ""
        if part:
            parts.add(part)
    return parts


def rhyme_match_score(left_word: str, right_word: str) -> float:
    left = normalize_word(left_word)
    right = normalize_word(right_word)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_parts = pronouncing_rhyme_parts(left)
    right_parts = pronouncing_rhyme_parts(right)
    if left_parts and right_parts and left_parts & right_parts:
        return 1.0

    left_tail = fallback_rhyme_tail(left)
    right_tail = fallback_rhyme_tail(right)
    if left_tail and left_tail == right_tail:
        return 0.8
    if len(left) >= 3 and len(right) >= 3 and left[-3:] == right[-3:]:
        return 0.7
    if len(left) >= 2 and len(right) >= 2 and left[-2:] == right[-2:]:
        return 0.4
    return 0.0


def rhyme_scores(lines: list[str]) -> tuple[float, float, str]:
    if len(lines) < 14:
        return 0.0, 0.0, ""
    endings = [last_word(line) for line in lines[:14]]
    pair_scores = [rhyme_match_score(endings[left], endings[right]) for left, right in RHYME_PAIRS]
    pair_score = statistics.mean(pair_scores) if pair_scores else 0.0
    final_couplet = pair_scores[-1] if pair_scores else 0.0
    return pair_score, final_couplet, " ".join(endings)


def prompt_preservation_score(predicted_prompt: list[str], prompt: list[str]) -> float | None:
    if not prompt:
        return None
    expected = [normalize_line(line) for line in prompt]
    generated = [normalize_line(line) for line in predicted_prompt[:len(prompt)]]
    return 1.0 if generated == expected else 0.0


def theme_groups_present(tokens: list[str]) -> set[str]:
    groups = set()
    for group_name, lexicon in THEME_GROUPS.items():
        if any(token in lexicon for token in tokens):
            groups.add(group_name)
    return groups


def prompt_theme_overlap(prompt_text: str, continuation_text: str) -> float | None:
    prompt_tokens = tokenize(prompt_text)
    continuation_tokens = tokenize(continuation_text)
    if not prompt_tokens:
        return None

    prompt_content = {
        token for token in prompt_tokens
        if token not in STOPWORDS and len(token) > 2
    }
    continuation_content = {
        token for token in continuation_tokens
        if token not in STOPWORDS and len(token) > 2
    }
    lexical_overlap = (
        len(prompt_content & continuation_content) / min(len(prompt_content), 8)
        if prompt_content else 0.0
    )

    prompt_themes = theme_groups_present(prompt_tokens)
    continuation_themes = theme_groups_present(continuation_tokens)
    if prompt_themes:
        group_overlap = len(prompt_themes & continuation_themes) / len(prompt_themes)
    else:
        group_overlap = 0.0

    return clamp(0.45 * lexical_overlap + 0.55 * group_overlap)


def coverage_score(tokens: list[str], lexicon: set[str], target_unique: int) -> float:
    if target_unique <= 0:
        return 0.0
    matches = {token for token in tokens if token in lexicon}
    return clamp(len(matches) / target_unique)


def alliteration_score(lines: list[str]) -> float:
    line_scores = []
    for line in lines:
        content_words = [token for token in tokenize(line) if len(token) > 2]
        if len(content_words) < 4:
            continue
        initials = [word[0] for word in content_words]
        repeated_adjacent = sum(
            1 for idx in range(1, len(initials))
            if initials[idx] == initials[idx - 1]
        )
        line_scores.append(clamp(repeated_adjacent / 2.0))
    return statistics.mean(line_scores) if line_scores else 0.0


def imagery_literary_device_score(text: str, lines: list[str]) -> float:
    tokens = tokenize(text)
    imagery = coverage_score(tokens, IMAGERY_WORDS, target_unique=6)
    markers = coverage_score(tokens, LITERARY_MARKERS, target_unique=4)
    alliteration = alliteration_score(lines)
    return 0.55 * imagery + 0.25 * markers + 0.20 * alliteration


def line_key(line: str, min_tokens: int) -> str | None:
    tokens = tokenize(line)
    if len(tokens) < min_tokens:
        return None
    key = normalize_line(line)
    return key if key else None


def token_ngram_set(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) < n:
        return set()
    return {
        tuple(tokens[idx:idx + n])
        for idx in range(0, len(tokens) - n + 1)
    }


def build_leakage_source(path: Path, ngram_n: int, min_line_tokens: int) -> dict[str, object]:
    blocks = load_numbered_blocks(path)
    line_set: set[str] = set()
    ngram_set: set[tuple[str, ...]] = set()
    for _, body in blocks:
        lines = nonempty_lines(body)
        for line in lines:
            key = line_key(line, min_line_tokens)
            if key:
                line_set.add(key)
        ngram_set |= token_ngram_set(body, ngram_n)
    return {"lines": line_set, "ngrams": ngram_set}


def evaluate_leakage(
    lines: list[str],
    text: str,
    sources: dict[str, dict[str, object]],
    ngram_n: int,
    min_line_tokens: int,
) -> dict[str, object]:
    candidate_lines = [key for key in (line_key(line, min_line_tokens) for line in lines) if key]
    candidate_ngrams = token_ngram_set(text, ngram_n)
    result: dict[str, object] = {}

    best_line_rate = 0.0
    best_ngram_rate = 0.0
    best_source = ""
    total_line_overlaps = 0
    total_ngram_overlaps = 0

    for source_name, source in sources.items():
        source_lines = source["lines"]
        source_ngrams = source["ngrams"]
        line_overlap = sum(1 for key in candidate_lines if key in source_lines)
        ngram_overlap = len(candidate_ngrams & source_ngrams)
        line_rate = line_overlap / len(candidate_lines) if candidate_lines else 0.0
        ngram_rate = ngram_overlap / len(candidate_ngrams) if candidate_ngrams else 0.0

        result[f"leakage_line_overlap_count_{source_name}"] = line_overlap
        result[f"leakage_line_overlap_rate_{source_name}"] = line_rate
        result[f"leakage_ngram_overlap_count_{source_name}"] = ngram_overlap
        result[f"leakage_ngram_overlap_rate_{source_name}"] = ngram_rate

        total_line_overlaps += line_overlap
        total_ngram_overlaps += ngram_overlap
        if line_rate > best_line_rate or ngram_rate > best_ngram_rate:
            best_line_rate = max(best_line_rate, line_rate)
            best_ngram_rate = max(best_ngram_rate, ngram_rate)
            best_source = source_name

    result["leakage_line_overlap_count"] = total_line_overlaps
    result["leakage_ngram_overlap_count"] = total_ngram_overlaps
    result["leakage_line_overlap_rate"] = best_line_rate
    result["leakage_ngram_overlap_rate"] = best_ngram_rate
    result["leakage_best_source"] = best_source
    return result


def parse_source_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_")
    return clean_name or Path(path).stem, Path(path)


def row_to_csv(row: dict[str, object]) -> dict[str, object]:
    clean = {}
    for key, value in row.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, float):
            clean[key] = round(value, 6)
        else:
            clean[key] = value
    return clean


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv(row))


def write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# Sonnet evaluation summary: {summary['model_name']}",
        "",
        f"- prediction file: `{summary['prediction_file']}`",
        f"- evaluated sonnets: {summary['count']}",
        f"- scoring part for reference/diversity: `{summary['score_part']}`",
        "",
        "| category | metric | value | direction |",
        "|---|---:|---:|---|",
    ]
    display_rows = [
        ("Reference similarity", "chrF", "higher"),
        ("Reference similarity", "BLEU", "higher"),
        ("Reference similarity", "ROUGE-L", "higher"),
        ("Reference similarity", "token-F1", "higher"),
        ("Sonnet form", "exact_14_lines_rate", "higher"),
        ("Sonnet form", "line_count_score", "higher"),
        ("Sonnet form", "line_length_score", "higher"),
        ("Sonnet form", "shakespearean_rhyme_pair_score", "higher"),
        ("Sonnet form", "final_couplet_rhyme", "higher"),
        ("Diversity", "MATTR", "higher"),
        ("Diversity", "distinct_1", "higher"),
        ("Diversity", "distinct_2", "higher"),
        ("Repetition", "repetition_rate", "lower"),
        ("Prompt faithfulness", "prompt_preservation", "higher"),
        ("Theme", "prompt_continuation_theme_overlap", "higher"),
        ("POEMetric proxy", "imagery_literary_device_score", "higher"),
        ("Leakage check", "leakage_any_line_overlap_rate", "lower"),
        ("Leakage check", "leakage_ngram_overlap_rate", "lower"),
    ]
    for category, metric, direction in display_rows:
        value = summary.get(metric)
        value_text = "" if value is None else f"{float(value):.6f}"
        lines.append(f"| {category} | `{metric}` | {value_text} | {direction} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    pred_path = Path(args.pred).resolve()
    gold_path = Path(args.gold).resolve() if args.gold else None
    prompt_path = Path(args.prompts).resolve() if args.prompts else None

    prediction_blocks = load_numbered_blocks(pred_path)
    gold_blocks = load_numbered_blocks(gold_path) if gold_path else None
    prompt_blocks = load_numbered_blocks(prompt_path) if prompt_path else None

    aligned_gold = align_blocks(prediction_blocks, gold_blocks, args.align)
    aligned_prompts = align_blocks(prediction_blocks, prompt_blocks, args.align)

    leakage_specs: list[tuple[str, Path]] = []
    if args.train_file:
        leakage_specs.append(("train", Path(args.train_file)))
    if args.dev_file:
        leakage_specs.append(("dev", Path(args.dev_file)))
    if args.test_file:
        leakage_specs.append(("test", Path(args.test_file)))
    for spec in args.leakage_source or []:
        leakage_specs.append(parse_source_spec(spec))

    leakage_sources = {
        source_name: build_leakage_source(path.resolve(), args.ngram_n, args.min_leakage_line_tokens)
        for source_name, path in leakage_specs
        if path.exists()
    }

    rows: list[dict[str, object]] = []
    scored_preds: list[str] = []
    scored_refs: list[str] = []
    scored_texts_for_diversity: list[str] = []

    for idx, (pred_id, pred_body) in enumerate(prediction_blocks):
        gold_block = aligned_gold[idx]
        prompt_block = aligned_prompts[idx]
        gold_id = gold_block[0] if gold_block else ""
        prompt_id = prompt_block[0] if prompt_block else ""
        gold_body = gold_block[1] if gold_block else None
        prompt_body = prompt_block[1] if prompt_block else None

        full_lines, continuation_lines, prompt_lines, predicted_prompt_lines = split_prediction(
            pred_body,
            prompt_body,
            args.prompt_lines,
            args.prediction_mode,
        )
        gold_full_lines, gold_continuation_lines = split_gold(gold_body, args.prompt_lines)

        full_text = text_from_lines(full_lines)
        continuation_text = text_from_lines(continuation_lines)
        prompt_text = text_from_lines(prompt_lines)
        score_text = continuation_text if args.score_part == "continuation" else full_text
        scored_texts_for_diversity.append(score_text)

        gold_score_text = None
        if gold_body is not None:
            if args.score_part == "continuation" and gold_continuation_lines is not None:
                gold_score_text = text_from_lines(gold_continuation_lines)
            elif gold_full_lines is not None:
                gold_score_text = text_from_lines(gold_full_lines)

        pair_score, final_couplet_score, endings = rhyme_scores(full_lines)
        prompt_preservation = prompt_preservation_score(predicted_prompt_lines, prompt_lines)
        theme_overlap = prompt_theme_overlap(prompt_text, continuation_text)
        imagery_score = imagery_literary_device_score(continuation_text, continuation_lines)

        row: dict[str, object] = {
            "row_index": idx,
            "pred_id": pred_id,
            "gold_id": gold_id,
            "prompt_id": prompt_id,
            "line_count": len(full_lines),
            "exact_14_lines": 1.0 if len(full_lines) == args.target_lines else 0.0,
            "line_count_score": line_count_score(full_lines, args.target_lines),
            "line_length_score": line_length_score(full_lines),
            "shakespearean_rhyme_pair_score": pair_score,
            "final_couplet_rhyme": final_couplet_score,
            "MATTR": mattr(score_text, args.mattr_window),
            "repetition_rate": repetition_rate(score_text),
            "prompt_preservation": prompt_preservation,
            "prompt_continuation_theme_overlap": theme_overlap,
            "imagery_literary_device_score": imagery_score,
            "ending_words": endings,
        }

        if gold_score_text is not None:
            row.update({
                "chrF": sentence_chrf(score_text, gold_score_text),
                "BLEU": sentence_bleu(score_text, gold_score_text),
                "ROUGE_L": rouge_l_f1(score_text, gold_score_text),
                "token_F1": token_f1(score_text, gold_score_text),
            })
            scored_preds.append(score_text)
            scored_refs.append(gold_score_text)
        else:
            row.update({"chrF": None, "BLEU": None, "ROUGE_L": None, "token_F1": None})

        leakage_lines = continuation_lines if args.leakage_part == "continuation" else full_lines
        leakage_text = text_from_lines(leakage_lines)
        row.update(
            evaluate_leakage(
                leakage_lines,
                leakage_text,
                leakage_sources,
                args.ngram_n,
                args.min_leakage_line_tokens,
            )
        )
        rows.append(row)

    summary: dict[str, object] = {
        "model_name": args.name,
        "prediction_file": str(pred_path),
        "gold_file": str(gold_path) if gold_path else "",
        "prompt_file": str(prompt_path) if prompt_path else "",
        "count": len(rows),
        "score_part": args.score_part,
        "prediction_mode": args.prediction_mode,
        "chrF": corpus_chrf(scored_preds, scored_refs) if scored_preds else None,
        "BLEU": corpus_bleu(scored_preds, scored_refs) if scored_preds else None,
        "ROUGE-L": mean(row.get("ROUGE_L") for row in rows),
        "token-F1": mean(row.get("token_F1") for row in rows),
        "ROUGE_L": mean(row.get("ROUGE_L") for row in rows),
        "token_F1": mean(row.get("token_F1") for row in rows),
        "exact_14_lines_rate": mean(row.get("exact_14_lines") for row in rows),
        "line_count_score": mean(row.get("line_count_score") for row in rows),
        "line_length_score": mean(row.get("line_length_score") for row in rows),
        "shakespearean_rhyme_pair_score": mean(row.get("shakespearean_rhyme_pair_score") for row in rows),
        "final_couplet_rhyme": mean(row.get("final_couplet_rhyme") for row in rows),
        "MATTR": mean(row.get("MATTR") for row in rows),
        "distinct_1": distinct_n(scored_texts_for_diversity, 1),
        "distinct_2": distinct_n(scored_texts_for_diversity, 2),
        "repetition_rate": corpus_repetition_rate(scored_texts_for_diversity),
        "prompt_preservation": mean(row.get("prompt_preservation") for row in rows),
        "prompt_continuation_theme_overlap": mean(row.get("prompt_continuation_theme_overlap") for row in rows),
        "imagery_literary_device_score": mean(row.get("imagery_literary_device_score") for row in rows),
        "leakage_any_line_overlap_rate": mean(
            1.0 if float(row.get("leakage_line_overlap_count", 0)) > 0 else 0.0
            for row in rows
        ),
        "leakage_line_overlap_rate": mean(row.get("leakage_line_overlap_rate") for row in rows),
        "leakage_any_ngram_overlap_rate": mean(
            1.0 if float(row.get("leakage_ngram_overlap_count", 0)) > 0 else 0.0
            for row in rows
        ),
        "leakage_ngram_overlap_rate": mean(row.get("leakage_ngram_overlap_rate") for row in rows),
        "ngram_n": args.ngram_n,
        "leakage_sources": ",".join(source_name for source_name, _ in leakage_specs),
    }
    return rows, summary


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sonnet generation outputs with unified metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred", required=True, help="Generated sonnet file in numbered-block format.")
    parser.add_argument("--gold", default="", help="Gold sonnet file. Omit for test-set evaluation without references.")
    parser.add_argument("--prompts", default="", help="Prompt file with the given first lines.")
    parser.add_argument("--name", default="model", help="Model/run name used in output filenames.")
    parser.add_argument("--out_dir", default="experiments/sonnet_metric_eval", help="Output directory.")
    parser.add_argument("--align", choices=["auto", "index", "id"], default="auto", help="How predictions align to gold/prompts.")
    parser.add_argument("--prediction_mode", choices=["full", "continuation"], default="full", help="Whether predictions include prompt lines.")
    parser.add_argument("--score_part", choices=["full", "continuation"], default="full", help="Text span used for reference/diversity metrics.")
    parser.add_argument("--leakage_part", choices=["full", "continuation"], default="continuation", help="Text span used for leakage checks.")
    parser.add_argument("--target_lines", type=int, default=14)
    parser.add_argument("--prompt_lines", type=int, default=3)
    parser.add_argument("--mattr_window", type=int, default=50)
    parser.add_argument("--ngram_n", type=int, default=5, help="Token n-gram size for leakage overlap.")
    parser.add_argument("--min_leakage_line_tokens", type=int, default=4, help="Ignore shorter lines in line-overlap leakage checks.")
    parser.add_argument("--train_file", default="", help="Training corpus used as leakage source label `train`.")
    parser.add_argument("--dev_file", default="", help="Development corpus used as leakage source label `dev`.")
    parser.add_argument("--test_file", default="", help="Test corpus used as leakage source label `test`.")
    parser.add_argument(
        "--leakage_source",
        action="append",
        default=[],
        help="Additional leakage source as NAME=PATH. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = evaluate(args)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.name).strip("_") or "model"

    per_sonnet_path = out_dir / f"{safe_name}_per_sonnet_metrics.csv"
    summary_csv_path = out_dir / f"{safe_name}_summary_metrics.csv"
    summary_json_path = out_dir / f"{safe_name}_summary_metrics.json"
    summary_md_path = out_dir / f"{safe_name}_summary_metrics.md"

    write_csv(per_sonnet_path, rows)
    write_csv(summary_csv_path, [summary])
    summary_json_path.write_text(
        json.dumps({key: rounded(value) if isinstance(value, float) else value for key, value in summary.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_markdown(summary_md_path, summary)

    print(f"Wrote per-sonnet metrics: {per_sonnet_path}")
    print(f"Wrote summary CSV: {summary_csv_path}")
    print(f"Wrote summary JSON: {summary_json_path}")
    print(f"Wrote summary Markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
