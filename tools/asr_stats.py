#!/usr/bin/env python3
"""Statistics and system-combination for ASR benchmarking.

Three layers that a raw WER table cannot give you:

1. **Confidence intervals.** A single WER on a small corpus is a point estimate
   with a wide interval around it. `bootstrap_wer_ci` resamples *clips* (the
   independent unit) to show how wide.

2. **Paired significance.** "Model A beat model B" is only a claim if the
   difference survives resampling. `paired_bootstrap` scores both systems on the
   same resampled clips, so the shared difficulty of the corpus cancels out --
   far more sensitive than comparing two independent intervals.

3. **Agreement / combination.** `rover_combine` merges several systems into a
   confusion network and votes per slot, which typically lands below the best
   single system. `disagreement_spans` uses the same network without any
   reference to point at words the systems argue about, which is where errors
   concentrate.

Nothing here needs numpy or scipy; the gate has to run in a bare container.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

# NULL slot marker in a confusion network (a system that said nothing here).
NULL = "@"


# ---- alignment with a backtrace ---------------------------------------------

# Edit operations, as (op, ref_index, hyp_index).
MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


def align_path(reference: Sequence[str], hypothesis: Sequence[str]) -> list[tuple[str, int, int]]:
    """Levenshtein alignment returning the operation path.

    wer_bench.align only needs counts; combination needs to know *which* words
    line up, so this keeps the backtrace.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    path: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                path.append((MATCH if cost == 0 else SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            path.append((DEL, i - 1, -1))
            i -= 1
            continue
        path.append((INS, -1, j - 1))
        j -= 1
    path.reverse()
    return path


# ---- bootstrap ---------------------------------------------------------------


@dataclass(frozen=True)
class ClipScore:
    """One clip's contribution to corpus WER."""

    errors: int
    reference_words: int


def corpus_wer(scores: Iterable[ClipScore]) -> float:
    errors = sum(s.errors for s in scores)
    words = sum(s.reference_words for s in scores)
    if words == 0:
        return 0.0
    return errors / words


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap_wer_ci(
    scores: Sequence[ClipScore],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for corpus WER.

    Clips are resampled with replacement because clips -- not words -- are the
    independent observations. Resampling words would treat every word as its own
    experiment and report an interval several times too narrow.
    """
    point = corpus_wer(scores)
    if not scores:
        return Interval(point, point, point)
    rng = random.Random(seed)
    n = len(scores)
    draws = []
    for _ in range(resamples):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        draws.append(corpus_wer(sample))
    draws.sort()
    lo_idx = int((1 - confidence) / 2 * resamples)
    hi_idx = min(resamples - 1, int((1 + confidence) / 2 * resamples))
    return Interval(point, draws[lo_idx], draws[hi_idx])


@dataclass(frozen=True)
class Comparison:
    delta: float  # wer_a - wer_b; negative means A is better
    low: float
    high: float
    p_value: float
    significant: bool

    def verdict(self, name_a: str, name_b: str, alpha: float) -> str:
        if not self.significant:
            return f"{name_a} vs {name_b}: no significant difference (p={self.p_value:.3f})"
        better, worse = (name_a, name_b) if self.delta < 0 else (name_b, name_a)
        return f"{better} beats {worse} by {abs(self.delta):.4f} WER (p={self.p_value:.3f})"


def paired_bootstrap(
    a: Sequence[ClipScore],
    b: Sequence[ClipScore],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap on the WER difference between two systems.

    Both systems are scored on the *same* resampled clips, so corpus difficulty
    cancels and only the difference between systems is resampled. This is why a
    paired test can call a 2-error difference significant when the two systems'
    individual intervals overlap heavily.

    p is two-sided: the proportion of resamples whose difference lands on the
    opposite side of zero from the observed difference, doubled.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same clips in the same order")
    observed = corpus_wer(a) - corpus_wer(b)
    if not a:
        return Comparison(0.0, 0.0, 0.0, 1.0, False)

    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(corpus_wer([a[i] for i in idx]) - corpus_wer([b[i] for i in idx]))
    deltas.sort()

    lo_idx = int((1 - confidence) / 2 * resamples)
    hi_idx = min(resamples - 1, int((1 + confidence) / 2 * resamples))
    low, high = deltas[lo_idx], deltas[hi_idx]

    # Proportion of resamples on the null side of zero.
    if observed < 0:
        tail = sum(1 for d in deltas if d >= 0)
    elif observed > 0:
        tail = sum(1 for d in deltas if d <= 0)
    else:
        tail = resamples // 2
    p = min(1.0, 2 * tail / resamples)
    # The interval and the p-value answer the same question; require both to
    # agree before calling a result significant.
    return Comparison(observed, low, high, p, p < (1 - confidence) and not (low <= 0 <= high))


# ---- ROVER-style combination -------------------------------------------------


def build_confusion_network(hypotheses: Sequence[Sequence[str]]) -> list[Counter]:
    """Align N hypotheses into a confusion network of voting slots.

    Simplified ROVER: the first hypothesis seeds the network, then each further
    hypothesis is aligned against the network's current consensus. Matches and
    substitutions vote in the aligned slot, deletions vote NULL, and insertions
    open a new slot. Real ROVER aligns against the whole network rather than its
    consensus and can weight by confidence; this keeps the useful part without
    needing per-word posteriors, which most of these backends do not expose.
    """
    if not hypotheses:
        return []
    network: list[Counter] = [Counter([w]) for w in hypotheses[0]]

    for hyp in hypotheses[1:]:
        consensus = [slot.most_common(1)[0][0] for slot in network]
        # Skip NULL-only consensus slots when aligning, then map back.
        positions = [i for i, w in enumerate(consensus) if w != NULL]
        compact = [consensus[i] for i in positions]

        path = align_path(compact, list(hyp))
        voted: set[int] = set()
        pending_inserts: dict[int, list[str]] = {}

        for op, ri, hi in path:
            if op in (MATCH, SUB):
                slot = positions[ri]
                network[slot][hyp[hi]] += 1
                voted.add(slot)
            elif op == DEL:
                slot = positions[ri]
                network[slot][NULL] += 1
                voted.add(slot)
            else:  # insertion: attach before the next aligned reference slot
                anchor = ri if ri >= 0 else _next_ref(path, hi)
                pending_inserts.setdefault(anchor, []).append(hyp[hi])

        # Slots this hypothesis never touched are deletions from its point of view.
        for i, slot in enumerate(network):
            if i not in voted:
                slot[NULL] += 1

        # Splice inserted words in as new slots, back to front so indices hold.
        for anchor in sorted(pending_inserts, reverse=True):
            at = positions[anchor] if 0 <= anchor < len(positions) else len(network)
            for word in reversed(pending_inserts[anchor]):
                fresh = Counter({word: 1, NULL: len(network[0]) if network else 0})
                network.insert(at, fresh)
    return network


def _next_ref(path: Sequence[tuple[str, int, int]], hyp_index: int) -> int:
    """Reference slot that follows an inserted hypothesis word."""
    for op, ri, hi in path:
        if ri >= 0 and hi >= hyp_index:
            return ri
    return -1


def rover_combine(hypotheses: Sequence[Sequence[str]], tie_break_order: bool = True) -> list[str]:
    """Vote a single word sequence out of several hypotheses.

    Ties fall back to the earliest system's word when tie_break_order is set,
    which makes the combiner deterministic and lets a known-good system act as
    the default voice.
    """
    network = build_confusion_network(hypotheses)
    out: list[str] = []
    first = list(hypotheses[0]) if hypotheses else []
    for slot in network:
        best_count = max(slot.values())
        winners = [w for w, c in slot.items() if c == best_count]
        if len(winners) > 1 and tie_break_order:
            preferred = next((w for w in first if w in winners), None)
            word = preferred if preferred is not None else sorted(winners)[0]
        else:
            word = winners[0] if len(winners) == 1 else sorted(winners)[0]
        if word != NULL:
            out.append(word)
    return out


def disagreement_spans(hypotheses: Sequence[Sequence[str]]) -> list[dict]:
    """Slots where the systems do not agree, with their vote split.

    Needs no reference, so it works on production traffic: low agreement is a
    usable confidence signal for flagging a transcript for review.
    """
    network = build_confusion_network(hypotheses)
    spans = []
    for i, slot in enumerate(network):
        total = sum(slot.values())
        top, count = slot.most_common(1)[0]
        if count < total:
            spans.append(
                {
                    "slot": i,
                    "winner": top,
                    "agreement": round(count / total, 3),
                    "votes": dict(slot),
                }
            )
    return spans


def agreement_rate(hypotheses: Sequence[Sequence[str]]) -> float:
    """Mean per-slot agreement across systems; 1.0 means unanimous everywhere."""
    network = build_confusion_network(hypotheses)
    if not network:
        return 1.0
    total = 0.0
    for slot in network:
        votes = sum(slot.values())
        total += slot.most_common(1)[0][1] / votes if votes else 1.0
    return total / len(network)
