"""Describe a pair of CDR3s with the 42 columns the model reads.

Everything is derived from the two CDR3 strings (no V or J gene features), so the
model runs on any pair of beta-chain CDR3s. It assumes the CDR3s are aligned to start
with N-terminal C and end with C-terminal F.

The alignment is vectorized. A pair is described by the
cheapest edit script between its two CDR3s, and computing that one pair at a time costs tens of
microseconds. Instead, pairs are bucketed by length, each bucket is held as a `uint8` matrix, and the
substitution positions for a whole bucket come out of one array comparison. Indel pairs are
handled by deleting each position of the longer side in turn and keeping the best alignment.

The supported edit space is 0-1 indels and 1-2 substitutions, which is what the model was
trained on. A pair outside this space - with identical CDR3s, two indels, three or more substitutions
is marked unsupported and scores NaN.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

#: Positions are counted into bins 0..POSITION_CLIP from each end, the last bin pooling everything
#: beyond. Must match the bins the coefficient table was fitted with.
POSITION_CLIP = 9

#: The Atchley factors the chemistry columns use, in the order the fitted columns expect.
CHEMICAL_FACTORS = ("polarity", "molecular_size", "charge")

MAX_INDELS, MAX_SWAPS = 1, 2

#: CDR3s outside this band are treated as malformed input, not as unusual receptors. Real beta
#: CDR3s run to roughly these lengths; the training set spanned 8 to 21, with 98% between 10 and
#: 16, so the band is deliberately wider than the data. Scores near its edges extrapolate -
#: length enters the model standardized on a mean of 13.2 and an SD of 1.33, so it moves the
#: logit a long way, and an unguarded 55-mer scored 1.0000 before this existed.
MIN_LENGTH, MAX_LENGTH = 7, 25
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@lru_cache(maxsize=1)
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BLOSUM62 and the Atchley factors as arrays indexed by residue code, plus the code map.

    `codes` maps a byte value (the ASCII code of a residue) to its row in the other two, and -1
    for anything not one of the twenty standard amino acids.
    """
    blosum = pd.read_csv(DATA / "blosum62.csv", index_col=0).loc[list(_ALPHABET), list(_ALPHABET)]
    atchley = pd.read_csv(DATA / "atchley_factors.csv", index_col=0).loc[list(_ALPHABET),
                                                                         list(CHEMICAL_FACTORS)]
    codes = np.full(256, -1, dtype=np.int8)
    for index, residue in enumerate(_ALPHABET):
        codes[ord(residue)] = index
    return blosum.to_numpy(float), atchley.to_numpy(float), codes


def _encode(sequences: np.ndarray, length: int) -> np.ndarray:
    """A (n, length) array of residue codes, -1 where the residue is not a standard amino acid.

    Non-ASCII characters are replaced rather than raising, so that an unusable CDR3 becomes an
    unsupported pair instead of an exception from deep inside the encoder.
    """
    raw = np.frombuffer("".join(sequences).encode("ascii", "replace"),
                        dtype=np.uint8).reshape(-1, length)
    return _tables()[2][raw]


def _lengths(sequences: np.ndarray) -> np.ndarray:
    """CDR3 lengths, or -1 for anything that is not a non-empty string (None, NaN, a float)."""
    def measure(value):
        return len(value) if isinstance(value, str) and value else -1

    if not len(sequences):
        return np.zeros(0, dtype=int)
    return np.frompyfunc(measure, 1, 1)(sequences).astype(int)


def align(cdr3_a, cdr3_b) -> dict[str, np.ndarray]:
    """The cheapest edit script for every pair, computed a length-bucket at a time.

    Returns arrays over the input, all of length n:

    - `n_indels`, `n_swaps` - the edit counts, with `n_swaps` = -1 where the pair is unsupported,
      which includes anything this module cannot describe at all.
    - `supported` - within 0-1 indels and 1-2 substitutions.
    - `length` - the common frame, which is the shorter CDR3's length.
    - `swap_a`, `swap_b` - (n, MAX_SWAPS) residue codes at each substitution, -1 where unused.
    - `swap_position` - (n, MAX_SWAPS) position in the common frame, -1 where unused.
    - `gap_position` - where the gap sits in the common frame, -1 on an equal-length pair.
    - `gap_ambiguity` - how many placements the gap admits; 0 without a gap.
    """
    a = np.asarray(cdr3_a, dtype=object)
    b = np.asarray(cdr3_b, dtype=object)
    if len(a) != len(b):
        raise ValueError(f"{len(a):,} and {len(b):,} CDR3s; the two sides must be the same length")
    n = len(a)
    length_a, length_b = _lengths(a), _lengths(b)
    in_band = lambda lengths: (lengths >= MIN_LENGTH) & (lengths <= MAX_LENGTH)
    usable = in_band(length_a) & in_band(length_b)

    out = {
        "n_indels": np.zeros(n, dtype=np.int8),
        "n_swaps": np.full(n, -1, dtype=np.int8),
        "length": np.where(usable, np.minimum(length_a, length_b), 0).astype(np.int16),
        "swap_a": np.full((n, MAX_SWAPS), -1, dtype=np.int8),
        "swap_b": np.full((n, MAX_SWAPS), -1, dtype=np.int8),
        "swap_position": np.full((n, MAX_SWAPS), -1, dtype=np.int16),
        "gap_position": np.full(n, -1, dtype=np.int16),
        "gap_ambiguity": np.zeros(n, dtype=np.int16),
    }
    difference = np.where(usable, length_a - length_b, 99)
    out["n_indels"][np.abs(difference) == 1] = 1

    # the longer side is deleted from, so orient every indel pair the same way
    swap_sides = difference < 0
    longer = np.where(swap_sides, b, a)
    shorter = np.where(swap_sides, a, b)

    _fill_equal_length(out, np.flatnonzero(difference == 0), a, b, length_a)
    _fill_one_indel(out, np.flatnonzero(np.abs(difference) == 1), longer, shorter,
                    np.minimum(length_a, length_b))
    out["supported"] = (usable & (out["n_swaps"] >= 0) & (out["n_indels"] <= MAX_INDELS)
                        & (out["n_indels"] + out["n_swaps"] > 0))
    return out


def _fill_equal_length(out, rows, a, b, lengths) -> None:
    """Substitution positions for the equal-length pairs, one length-bucket at a time."""
    for length in np.unique(lengths[rows]) if len(rows) else ():
        bucket = rows[lengths[rows] == length]
        left, right = _encode(a[bucket], length), _encode(b[bucket], length)
        # an unknown residue has no BLOSUM row, so the pair cannot be described at all
        known = ((left >= 0) & (right >= 0)).all(axis=1)
        differs = (left != right) & known[:, None]
        counts = differs.sum(axis=1)
        usable = known & (counts <= MAX_SWAPS)
        out["n_swaps"][bucket[usable]] = counts[usable].astype(np.int8)
        for slot in range(min(MAX_SWAPS, length)):
            # the slot-th differing position of each row, where there is one
            order = np.argsort(~differs, axis=1, kind="stable")[:, slot]
            has = usable & (counts > slot)
            if not has.any():
                continue
            take, position = bucket[has], order[has]
            out["swap_position"][take, slot] = position
            out["swap_a"][take, slot] = left[has, position]
            out["swap_b"][take, slot] = right[has, position]


def _fill_one_indel(out, rows, longer, shorter, common) -> None:
    """Best single-deletion alignment for the indel pairs, one common-length bucket at a time.

    Deleting each position of the longer side in turn and keeping the fewest substitutions is the
    cheapest edit script. Ties are the gap's placement ambiguity, which the model uses.
    """
    for length in np.unique(common[rows]) if len(rows) else ():
        bucket = rows[common[rows] == length]
        long_codes = _encode(longer[bucket], length + 1)
        short_codes = _encode(shorter[bucket], length)
        known = (long_codes >= 0).all(axis=1) & (short_codes >= 0).all(axis=1)
        swaps = np.empty((len(bucket), length + 1), dtype=np.int16)
        for position in range(length + 1):
            keep = np.delete(np.arange(length + 1), position)
            swaps[:, position] = (long_codes[:, keep] != short_codes).sum(axis=1)
        best = swaps.min(axis=1)
        usable = known & (best <= MAX_SWAPS)
        out["n_swaps"][bucket[usable]] = best[usable].astype(np.int8)
        out["gap_ambiguity"][bucket] = (swaps == best[:, None]).sum(axis=1)
        out["gap_position"][bucket] = swaps.argmin(axis=1)
        if not usable.any():
            continue
        # re-derive the substitutions under the chosen placement
        chosen = swaps.argmin(axis=1)
        for position in np.unique(chosen[usable]):
            here = usable & (chosen == position)
            keep = np.delete(np.arange(length + 1), position)
            left, right = long_codes[np.ix_(here, keep)], short_codes[here]
            differs = left != right
            counts = differs.sum(axis=1)
            take = bucket[here]
            for slot in range(min(MAX_SWAPS, length)):
                order = np.argsort(~differs, axis=1, kind="stable")[:, slot]
                has = counts > slot
                if not has.any():
                    continue
                rows_here, at = take[has], order[has]
                out["swap_position"][rows_here, slot] = at
                out["swap_a"][rows_here, slot] = left[has, at]
                out["swap_b"][rows_here, slot] = right[has, at]


def columns(cdr3_a, cdr3_b, alignment: dict | None = None) -> dict[str, np.ndarray]:
    """Every named feature column, as a dict of arrays. Unsupported pairs are filled with zeros.

    The model selects the columns it needs by name, so this returns the full set and does not
    care about their order.
    """
    alignment = align(cdr3_a, cdr3_b) if alignment is None else alignment
    n = len(alignment["n_indels"])
    built: dict[str, np.ndarray] = {
        "cdr3len": alignment["length"].astype(float),
        "n_swaps": alignment["n_swaps"].clip(min=0).astype(float),
        "n_indels": alignment["n_indels"].astype(float),
        "gap_ambiguity": alignment["gap_ambiguity"].astype(float),
    }

    rows = np.arange(n)
    position, length = alignment["swap_position"], alignment["length"]
    for prefix, distance in (("n_swaps_at_start", position),
                             ("n_swaps_at_end", length[:, None] - position - 1)):
        binned = np.minimum(distance, POSITION_CLIP)
        for index in range(POSITION_CLIP + 1):
            built[f"{prefix}_{index}"] = ((position >= 0) & (binned == index)).sum(axis=1).astype(float)

    gap, gapped = alignment["gap_position"], alignment["gap_position"] >= 0
    gap_from_end = np.where(gapped, np.minimum(length - gap, POSITION_CLIP), -1)
    for index in range(POSITION_CLIP + 1):
        built[f"gap_at_end_{index}"] = (gap_from_end == index).astype(float)

    blosum, atchley, _ = _tables()
    left, right = alignment["swap_a"], alignment["swap_b"]
    present = left >= 0
    counts = present.sum(axis=1)
    safe_left, safe_right = np.where(present, left, 0), np.where(present, right, 0)

    per_swap = {"swap_blosum62": blosum[safe_left, safe_right]}
    for index, factor in enumerate(CHEMICAL_FACTORS):
        a_value, b_value = atchley[safe_left, index], atchley[safe_right, index]
        per_swap[f"swap_{factor}_change"] = np.abs(a_value - b_value)
        per_swap[f"swap_{factor}_mean"] = (a_value + b_value) / 2
    for stem, values in per_swap.items():
        total = np.where(present, values, 0.0).sum(axis=1)
        built[f"{stem}_total"] = total
        built[f"{stem}_mean"] = np.divide(total, counts, out=np.zeros(n), where=counts > 0)

    unsupported = ~alignment["supported"]
    if unsupported.any():
        for value in built.values():
            value[unsupported] = 0.0
    return built


def edit_class(cdr3_a, cdr3_b) -> tuple[np.ndarray, np.ndarray]:
    """Counts of (indels, substitutions) per pair. Substitutions are -1 where the pair is unsupported."""
    alignment = align(cdr3_a, cdr3_b)
    return alignment["n_indels"], alignment["n_swaps"]
