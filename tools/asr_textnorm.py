#!/usr/bin/env python3
"""Optional formatting normalization for ASR scoring.

The strict normalizer in wer_bench.py is the headline: casefold, strip
punctuation, collapse whitespace. It deliberately leaves everything else alone.

This module adds an opt-in second profile for *formatting* disagreements -- the
same words written differently -- so a model is not punished for writing "10"
where the reference writes "ten". DictatorFlow's own corpus has exactly this:
real_numbers.wav is referenced as "Testing 1 2 3 4 5 6 7 8 9 10", so any model
that spells the digits out scores near 100% on that clip while being perfectly
correct.

The hard line: this fixes how a word is *written*, never which word was *heard*.
"jumps" -> "dumps" is a real recognition error and must keep counting. Anything
that would map two genuinely different words together belongs nowhere near this
file, because a normalizer that hides errors makes every downstream number a
lie.

Both reference and hypothesis always go through the same profile.
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# Spelling variants that are the same word. Kept to genuine orthographic
# variants -- never near-homophones.
SPELLING = {
    "okay": "ok",
    "colour": "color",
    "colours": "colors",
    "favourite": "favorite",
    "organise": "organize",
    "organised": "organized",
    "recognise": "recognize",
    "recognised": "recognized",
    "analyse": "analyze",
    "behaviour": "behavior",
    "licence": "license",
    "grey": "gray",
}

# Contractions expanded on both sides so "don't" and "do not" agree.
CONTRACTIONS = {
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "cant": "can not", "cannot": "can not", "wont": "will not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not",
    "werent": "were not", "havent": "have not", "hasnt": "has not",
    "hadnt": "had not", "shouldnt": "should not", "wouldnt": "would not",
    "couldnt": "could not", "its": "it is", "thats": "that is",
    "lets": "let us", "im": "i am", "ive": "i have", "ill": "i will",
    "id": "i would", "youre": "you are", "youve": "you have",
    "theyre": "they are", "theres": "there is", "hes": "he is",
    "shes": "she is", "were_": "we are", "weve": "we have",
}


def _base_value(tokens: list[str], i: int) -> tuple[int | None, int]:
    """Parse one value below 100: "five", "twenty", "twenty five"."""
    if i >= len(tokens):
        return None, 0
    tok = tokens[i]
    if tok in TENS:
        value, consumed = TENS[tok], 1
        # "twenty five" is one number; "twenty hundred" is not, so only absorb a
        # following unit.
        if i + 1 < len(tokens) and tokens[i + 1] in UNITS and UNITS[tokens[i + 1]] < 10:
            value += UNITS[tokens[i + 1]]
            consumed += 1
        return value, consumed
    if tok in UNITS:
        return UNITS[tok], 1
    return None, 0


def words_to_number(tokens: list[str], start: int) -> tuple[int | None, int]:
    """Parse exactly one spelled-out number starting at `start`.

    Returns (value, tokens_consumed).

    The subtlety that matters for dictation: a run of bare units is a *sequence
    of digits*, not a sum. "one two three" is 1 2 3 (someone reading out digits),
    while "twenty five" is 25 and "three hundred" is 300. An earlier version
    accumulated greedily and turned "one two three" into 6, which would have
    quietly mangled the exact clip this profile exists to fix.

    So a number only extends past its base value through an explicit scale word
    (hundred / thousand), never by butting two units together.
    """
    value, consumed = _base_value(tokens, start)
    if value is None:
        return None, 0
    i = start + consumed

    # Scale words multiply what came before and may be followed by a remainder:
    # "three hundred and five", "two thousand five hundred".
    while i < len(tokens) and tokens[i] in ("hundred", "thousand"):
        scale = 100 if tokens[i] == "hundred" else 1000
        value = max(value, 1) * scale
        i += 1
        consumed = i - start

        j = i
        if j < len(tokens) and tokens[j] == "and":
            j += 1
        remainder, used = _base_value(tokens, j)
        if remainder is None:
            continue
        # "two thousand five hundred": the remainder itself takes a scale.
        if j + used < len(tokens) and tokens[j + used] in ("hundred", "thousand"):
            inner_scale = 100 if tokens[j + used] == "hundred" else 1000
            if inner_scale < scale:
                value += remainder * inner_scale
                i = j + used + 1
                consumed = i - start
                continue
            break
        value += remainder
        i = j + used
        consumed = i - start

    return value, consumed


def normalize_numbers(tokens: list[str]) -> list[str]:
    """Rewrite spelled-out numbers as digits so both spellings compare equal."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        value, consumed = words_to_number(tokens, i)
        if value is not None and consumed > 0:
            out.append(str(value))
            i += consumed
            continue
        out.append(tokens[i])
        i += 1
    return out


def split_alnum(token: str) -> list[str]:
    """Split letter/digit runs: "mp3" -> "mp 3", so "MP 3" and "mp3" agree."""
    parts = re.findall(r"[a-z]+|\d+", token)
    return parts if len(parts) > 1 else [token]


def lenient_tokens(text: str, strict_tokens) -> list[str]:
    """Apply the formatting profile on top of the strict tokenizer.

    strict_tokens is passed in rather than imported so this module stays
    independent of wer_bench's import path.
    """
    tokens = strict_tokens(text)

    expanded: list[str] = []
    for tok in tokens:
        tok = tok.replace("'", "")
        tok = SPELLING.get(tok, tok)
        if tok in CONTRACTIONS:
            expanded.extend(CONTRACTIONS[tok].split())
            continue
        expanded.extend(split_alnum(tok))

    return normalize_numbers(expanded)
