#!/usr/bin/env python3
"""Evaluate generated sonnets for the NLP2026 sonnet-generation experiments.

Core assignment metric:
- chrF via sacrebleu, matching the starter evaluation.

Additional diagnostics:
- Sonnet or Not, Bot?-style poetic-form detection: an offline fixed-form check
  for whether a poem satisfies sonnet-like structure.
- POEMetric-style dimensions requested for the report: form accuracy, theme
  alignment, lexical diversity, and overall quality.

The POEMetric form checker follows the released rule-based algorithm structure:
https://github.com/Bingru-Li/POEMetric/blob/main/POEMetric_rule_based_algorithm.py
For Sonnet or Not, Bot?, the released repo is prompt/classification oriented;
this script uses the same target task idea, poetic form recognition, but keeps it
offline and deterministic for this assignment.
"""

import argparse
import csv
import json
import re
import statistics
import string
from pathlib import Path

from sacrebleu.metrics import CHRF

try:
    from nltk.corpus import cmudict
    CMU = cmudict.dict()
except Exception:
    CMU = {}

SONNET_RHYME = "ABABCDCDEFEFGG"
SONNET_RHYME_PAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in", "is",
    "it", "its", "me", "my", "nor", "not", "of", "on", "or", "our", "ours",
    "she", "so", "that", "the", "their", "theirs", "them", "then", "there",
    "these", "they", "this", "thou", "thy", "thine", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "yet", "you", "your",
    "shall", "doth", "did", "do", "can", "could", "would", "should", "may", "might",
}


def parse_sonnets(path):
    text = Path(path).read_text(encoding="utf-8")
    parts = re.split(r"\n\s*(\d+)\s*\n", text)
    sonnets = []
    for i in range(1, len(parts), 2):
        sid = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            sonnets.append((sid, body))
    return sonnets


def clean_lines(poem):
    return [line.strip() for line in poem.splitlines() if line.strip()]


def words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def content_words(text):
    return [w for w in words(text) if len(w) > 2 and w not in STOPWORDS]


def poem2words(poem):
    # Mirrors the released POEMetric preprocessing: line split, hyphen split,
    # possessive simplification, alphabetic tokens only.
    word_lists = []
    for line in clean_lines(poem):
        line = line.replace("-", " ").replace("'s ", " ")
        toks = re.findall(r"[A-Za-z]+", line)
        toks = [tok for tok in toks if tok.lower() != "s"]
        if toks:
            word_lists.append(toks)
    return word_lists


def fallback_rhyme_foot(word):
    word = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not word:
        return ""
    match = re.search(r"[aeiouy][a-z']*$", word)
    return match.group(0)[-4:] if match else word[-3:]


def extract_rhyme_foot(word):
    key = word.lower().strip(string.punctuation)
    if key in CMU:
        phones = CMU[key][0]
        vowels = re.compile(r"[AEIOU]")
        rhyme_parts = []
        for phone in reversed(phones):
            if vowels.search(phone):
                rhyme_parts.append(phone)
                break
            rhyme_parts.append(phone)
        return "".join(reversed(rhyme_parts))
    return fallback_rhyme_foot(key)


def rhymes_similar(foot1, foot2):
    if not foot1 or not foot2:
        return False
    if foot1 == foot2:
        return True
    f1 = re.sub(r"\d", "", foot1)
    f2 = re.sub(r"\d", "", foot2)
    if f1 == f2:
        return True
    if len(f1) == len(f2):
        return sum(1 for a, b in zip(f1, f2) if a != b) == 1
    if abs(len(f1) - len(f2)) == 1:
        longer, shorter = (f1, f2) if len(f1) > len(f2) else (f2, f1)
        return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(shorter) + 1))
    return False


def poem2rhyme(poem_words):
    last_words = [line_words[-1] for line_words in poem_words if line_words]
    feet = [extract_rhyme_foot(word) for word in last_words]
    rhyme_mapping = {}
    rhyme_letters = []
    rhyme_counter = 0
    for foot in feet:
        found = None
        for existing_foot, letter in rhyme_mapping.items():
            if rhymes_similar(existing_foot, foot):
                found = letter
                break
        if found:
            letter = found
        else:
            rhyme_counter += 1
            letter = chr(64 + rhyme_counter) if rhyme_counter <= 26 else f"Z{rhyme_counter}"
            rhyme_mapping[foot] = letter
        rhyme_letters.append(letter)
    return list(zip(last_words, feet, rhyme_letters))


def transform_rhyme_string(s):
    char_map = {}
    result = []
    next_ord = ord("A")
    for char in s:
        if char not in char_map:
            char_map[char] = chr(next_ord)
            next_ord += 1
        result.append(char_map[char])
    return "".join(result)


def poemetric_sonnet_rule(poem):
    # Close adaptation of POEMetric.check_poem(..., form='sonnet', rhyme='ABABCDCDEFEFGG').
    poem_words = poem2words(poem)
    rhyme_analysis = poem2rhyme(poem_words)
    poem_rhyme = "".join(r[-1] for r in rhyme_analysis)
    if not poem_rhyme:
        return False, "No rhyme analysis", ""
    if len(poem_rhyme) % len(SONNET_RHYME) != 0:
        return False, "False number of lines", poem_rhyme
    n_line_per_group = len(SONNET_RHYME)
    n_group = len(poem_rhyme) // len(SONNET_RHYME)
    n_matched = 0
    normalized_groups = []
    for group_idx in range(n_group):
        group = poem_rhyme[group_idx * n_line_per_group:(group_idx + 1) * n_line_per_group]
        normalized = transform_rhyme_string(group)
        normalized_groups.append(normalized)
        for predicted, expected in zip(normalized, SONNET_RHYME):
            if predicted == expected:
                n_matched += 1
    ratio = n_matched / len(poem_rhyme)
    if ratio > 0.7:
        return True, "", "".join(normalized_groups)
    return False, "False rhyme pattern", "".join(normalized_groups)


def line_count_score(lines):
    if not lines:
        return 0.0
    return max(0.0, 1.0 - abs(len(lines) - 14) / 14.0)


def line_length_score(lines):
    if not lines:
        return 0.0
    per_line = []
    for line in lines[:14]:
        length = len(words(line))
        per_line.append(max(0.0, 1.0 - abs(length - 10) / 10.0))
    return sum(per_line) / len(per_line) if per_line else 0.0


def pairwise_rhyme_score(lines):
    if len(lines) < 14:
        return 0.0
    feet = [extract_rhyme_foot(poem2words(line)[0][-1]) if poem2words(line) else "" for line in lines[:14]]
    matches = 0
    for left, right in SONNET_RHYME_PAIRS:
        if rhymes_similar(feet[left], feet[right]):
            matches += 1
    return matches / len(SONNET_RHYME_PAIRS)


def sonnet_form_diagnostics(poem):
    lines = clean_lines(poem)
    is_correct, reason, detected_rhyme = poemetric_sonnet_rule(poem)
    count = line_count_score(lines)
    rhyme_pairs = pairwise_rhyme_score(lines)
    length = line_length_score(lines)
    # Diagnostic partial score for ranking models when exact rule correctness is sparse.
    partial = 0.45 * count + 0.40 * rhyme_pairs + 0.15 * length
    return {
        "line_count": len(lines),
        "detected_rhyme_pattern": detected_rhyme,
        "poemetric_rule_form_correct": is_correct,
        "poemetric_rule_form_reason": reason,
        "line_count_score": count,
        "rhyme_pair_score": rhyme_pairs,
        "line_length_score": length,
        "form_partial_raw": partial,
    }


def mattr(text, window=50):
    toks = words(text)
    if not toks:
        return 0.0
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    scores = []
    for i in range(0, len(toks) - window + 1):
        span = toks[i:i + window]
        scores.append(len(set(span)) / window)
    return sum(scores) / len(scores)


def jaccard(a, b):
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def theme_alignment(generated, gold=None, theme=None):
    # The public POEMetric repo does not release a local theme judge. If explicit
    # themes are supplied, use target-theme keyword recall; otherwise use the
    # gold held-out poem as the theme/content reference.
    generated_words = content_words(generated)
    if theme:
        target = content_words(theme)
        return len(set(target) & set(generated_words)) / len(set(target)) if target else None
    if gold:
        return jaccard(generated_words, content_words(gold))
    return None


def load_themes(path):
    if not path:
        return {}
    theme_path = Path(path)
    if theme_path.suffix.lower() == ".json":
        data = json.loads(theme_path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}
    themes = {}
    with theme_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("id") or row.get("sid") or row.get("sonnet_id")
            theme = row.get("theme") or row.get("Theme")
            if sid is not None and theme:
                themes[str(sid)] = theme
    return themes


def scale_5(value):
    if value is None:
        return None
    return max(0.0, min(5.0, 5.0 * value))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def evaluate_file(pred_path, gold_path, themes):
    generated = parse_sonnets(pred_path)
    gold = parse_sonnets(gold_path) if gold_path else []
    gold_by_index = {str(i): text for i, (_, text) in enumerate(gold)}
    gold_by_sid = {sid: text for sid, text in gold}

    max_len = min(len(generated), len(gold)) if gold else len(generated)
    generated_for_chrf = [text for _, text in generated[:max_len]]
    gold_for_chrf = [text for _, text in gold[:max_len]]
    chrf_score = None
    if gold_for_chrf and generated_for_chrf:
        chrf_score = float(CHRF().corpus_score(generated_for_chrf, [gold_for_chrf]).score)

    rows = []
    for idx, (sid, poem) in enumerate(generated):
        gold_text = gold_by_sid.get(sid, gold_by_index.get(str(idx)))
        form = sonnet_form_diagnostics(poem)
        lex = mattr(poem)
        theme = theme_alignment(poem, gold=gold_text, theme=themes.get(sid) or themes.get(str(idx)))
        # POEMetric GitHub rule-based form is a correctness ratio; keep that exact
        # signal, and use partial form only for overall-quality diagnostics.
        form_exact_raw = 1.0 if form["poemetric_rule_form_correct"] else 0.0
        overall_parts = [form["form_partial_raw"], lex]
        if theme is not None:
            overall_parts.append(theme)
        overall = mean(overall_parts)
        rows.append({
            "id": sid,
            "sonnet_or_not_bot_form_label": "sonnet" if form["poemetric_rule_form_correct"] else "not_sonnet",
            "sonnet_or_not_bot_is_sonnet": form_exact_raw,
            "poemetric_form_accuracy": scale_5(form_exact_raw),
            "poemetric_form_partial": scale_5(form["form_partial_raw"]),
            "poemetric_theme_alignment": scale_5(theme),
            "poemetric_lexical_diversity": scale_5(lex),
            "poemetric_overall_quality": scale_5(overall),
            "theme_alignment_raw": theme,
            "lexical_diversity_mattr_raw": lex,
            **form,
        })

    summary = {
        "file": str(pred_path),
        "count": len(generated),
        "chrF": chrf_score,
        "sonnet_or_not_bot_accuracy": mean([r["sonnet_or_not_bot_is_sonnet"] for r in rows]),
        "poemetric_form_accuracy": mean([r["poemetric_form_accuracy"] for r in rows]),
        "poemetric_form_partial": mean([r["poemetric_form_partial"] for r in rows]),
        "poemetric_theme_alignment": mean([r["poemetric_theme_alignment"] for r in rows]),
        "poemetric_lexical_diversity": mean([r["poemetric_lexical_diversity"] for r in rows]),
        "poemetric_overall_quality": mean([r["poemetric_overall_quality"] for r in rows]),
        "cmudict_available": bool(CMU),
    }
    return summary, rows


def prediction_files(path):
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*_generated_sonnets.txt"))
    return [p]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="predictions", help="Prediction file or directory")
    parser.add_argument("--gold", default="data/TRUE_sonnets_held_out_dev.txt")
    parser.add_argument("--themes", default=None, help="Optional JSON or CSV mapping id/theme")
    parser.add_argument("--out_dir", default="evaluation_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    themes = load_themes(args.themes)

    summaries = []
    for pred in prediction_files(args.pred):
        summary, rows = evaluate_file(pred, args.gold, themes)
        summaries.append(summary)
        stem = pred.stem.replace("_generated_sonnets", "")
        detail_path = out_dir / f"{stem}_details.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
            writer.writeheader()
            writer.writerows(rows)

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summaries[0].keys()) if summaries else ["file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved summary: {summary_csv}")
    print(f"Saved summary: {summary_json}")
    for summary in summaries:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
