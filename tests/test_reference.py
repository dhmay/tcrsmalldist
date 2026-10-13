"""The contract: this implementation reproduces the research code it was ported from.

`tests/data/reference_scores.tsv` is 2,000 ImmuneCODE pairs, 400 from each edit class, scored by
`tcrsmalldist-devel`'s `portable.score_pairs_lr` - a straightforward per-pair implementation. The
alignment here is vectorized by length bucket instead, which is a different algorithm reaching
the same answer. If it ever stops reaching the same answer, this test is how you find out.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcrsmalldist import edit_class, score

REFERENCE = Path(__file__).parent / "data" / "reference_scores.tsv"


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE, sep="\t")


def test_scores_match_the_research_implementation(reference):
    ours = score(reference.cdr3_a.to_numpy(), reference.cdr3_b.to_numpy())
    assert np.isfinite(ours).all(), "every reference pair is inside the supported edit space"
    assert np.abs(ours - reference.reference_score.to_numpy()).max() < 1e-10


def test_every_edit_class_is_covered(reference):
    indels, swaps = edit_class(reference.cdr3_a.to_numpy(), reference.cdr3_b.to_numpy())
    found = {f"{i} indel + {s} swap{'' if s == 1 else 's'}" for i, s in zip(indels, swaps)}
    assert found == set(reference.edit_class)
