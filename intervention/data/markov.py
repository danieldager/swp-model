"""Attested-bigram chain over the training lexicon, plus one constrained sampler.

``sample_sequence`` draws a phoneme sequence whose every bigram is attested, holding a
set of positions fixed. Both places that need "random but word-like phonemes" are that
one call with different clamps:

    edit    -> clamp everything outside the edited window, resample the window
    source  -> clamp the window the counterfactual must carry, resample the rest

Sampling is left-to-right with one-step lookahead (a candidate must be reachable from
the previous phoneme and able to reach the next clamped one), which is what keeps the
yield high enough to be usable; callers still gate the result on the content filters.
"""
from __future__ import annotations

import numpy as np

Chain = dict[str, object]


def build_chain(sequences: list[list[int]], special_tokens: set[int] | None = None) -> Chain:
    """Attested transitions and word-initial phonemes from the training words."""
    special = special_tokens or set()
    succ: dict[int, set[int]] = {}
    pred: dict[int, set[int]] = {}
    starts: set[int] = set()

    for seq in sequences:
        clean = [t for t in seq if t not in special]
        if not clean:
            continue
        starts.add(clean[0])
        for a, b in zip(clean, clean[1:]):
            succ.setdefault(a, set()).add(b)
            pred.setdefault(b, set()).add(a)

    return {
        "succ": {a: np.fromiter(sorted(bs), dtype=np.int64) for a, bs in succ.items()},
        "pred": pred,
        "starts": np.fromiter(sorted(starts), dtype=np.int64),
    }


def sample_sequence(
    length: int,
    clamped: dict[int, int],
    chain: Chain,
    rng: np.random.Generator,
    max_attempts: int = 8,
) -> list[int] | None:
    """A ``length``-phoneme sequence with every bigram attested and ``clamped`` honoured.

    Returns ``None`` when the constraints leave no candidate within ``max_attempts``.
    """
    succ, pred, starts = chain["succ"], chain["pred"], chain["starts"]

    for _ in range(max_attempts):
        out: list[int] = []
        for i in range(length):
            fixed = clamped.get(i)
            if fixed is not None:
                if i and out[-1] not in pred.get(fixed, ()):
                    break  # the clamped phoneme is unreachable from what we drew
                out.append(fixed)
                continue

            cands = starts if i == 0 else succ.get(out[-1], np.empty(0, dtype=np.int64))
            nxt = clamped.get(i + 1)
            if nxt is not None:
                allowed = pred.get(nxt, set())
                cands = np.fromiter((c for c in cands if c in allowed), dtype=np.int64)
            if len(cands) == 0:
                break
            out.append(int(rng.choice(cands)))
        else:
            return out
    return None
