"""Input that cannot be described must come back NaN - never a score, never an exception.

The dangerous case is not the crash, it is the near-miss: before these were handled, an unknown
residue at a substituted position was counted by the position bins and skipped by the chemistry
aggregates, so the pair scored from a feature vector no real pair could produce.
"""

import numpy as np
import pytest

from tcrsmalldist import edit_class, score

VALID = ("CASSLYGGARYEQYF", "CASSLYGGARYEQFF")

UNUSABLE = [
    ("unknown residue at the swapped position", "CASSXGDYEQYF", "CASSLGDYEQYF"),
    ("unknown residue elsewhere in the pair", "CASSXGDYEQYF", "CASSXGDYEQWF"),
    ("two different unknown residues", "CASSXGDYEQYF", "CASSZGDYEQYF"),
    ("unknown residue in an indel pair", "CASSXLAGGNEQYF", "CASSXAGGNEQYF"),
    ("lowercase", "casslygdyeqyf", "casslygdyeqwf"),
    ("a digit", "CASS1GDYEQYF", "CASS2GDYEQYF"),
    ("non-ascii", "CASSÅGDYEQYF", "CASSLGDYEQYF"),
    ("empty against non-empty", "C", ""),
    ("both empty", "", ""),
    ("None", None, "CASSLGDYEQYF"),
    ("a float nan", float("nan"), "CASSLGDYEQYF"),
    ("not a string at all", 12345, "CASSLGDYEQYF"),
    ("shorter than the length band", "CASSLF", "CASSLY"),
    ("longer than the length band", "C" + "A" * 24 + "F", "C" + "A" * 23 + "GF"),
]


@pytest.mark.parametrize("label,a,b", UNUSABLE, ids=[case[0] for case in UNUSABLE])
def test_unusable_pairs_score_nan(label, a, b):
    assert np.isnan(score([a], [b])[0])


@pytest.mark.parametrize("label,a,b", UNUSABLE, ids=[case[0] for case in UNUSABLE])
def test_unusable_pairs_report_minus_one_swaps(label, a, b):
    _, swaps = edit_class([a], [b])
    assert swaps[0] == -1, "-1 is how an undescribable pair is distinguished from a 0-swap one"


@pytest.mark.parametrize("length", [7, 25])
def test_the_length_band_is_inclusive(length):
    a, b = "C" + "A" * (length - 2) + "F", "C" + "A" * (length - 3) + "GF"
    assert np.isfinite(score([a], [b])[0])


def test_one_bad_pair_does_not_poison_its_neighbours():
    a = [VALID[0], "CASSXGDYEQYF", VALID[0]]
    b = [VALID[1], "CASSLGDYEQYF", VALID[1]]
    got = score(a, b)
    assert np.isnan(got[1])
    assert got[0] == got[2] == pytest.approx(score([VALID[0]], [VALID[1]])[0])
