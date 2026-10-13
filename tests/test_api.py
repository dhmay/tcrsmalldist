"""Behaviour at the edges: what is supported, what comes back NaN, what the shapes are."""

import numpy as np
import pandas as pd
import pytest

from tcrsmalldist import Model, edit_class, features, score, score_dataframe

SUPPORTED = ("CASSLYGGARYEQYF", "CASSLYGGARYEQFF")


def test_unsupported_pairs_are_nan_not_extrapolated():
    a = [SUPPORTED[0], "CASSLTGDYEQYF", "CASSLTGDYEQYF", "CASSLTGDYEQYF"]
    b = [SUPPORTED[1], "CASSLTGDYEQYF", "CAWWWWWWWEQYF", "CASSLTGDYEQ"]
    got = score(a, b)
    assert np.isfinite(got[0])
    assert np.isnan(got[1:]).all(), "identical, three swaps and two indels are all unsupported"


def test_edit_class_counts_the_cheapest_script():
    indels, swaps = edit_class(["CASSLTGDYEQYF", "CASSLLAGGNEQYF"],
                               ["CASSLTGDWEQYF", "CASSLAGGNEQYF"])
    assert list(indels) == [0, 1]
    assert list(swaps) == [1, 0]


def test_features_match_the_models_column_order():
    matrix = features([SUPPORTED[0]], [SUPPORTED[1]])
    assert matrix.shape == (1, len(Model.load().features))


def test_score_dataframe_keeps_the_index():
    frame = pd.DataFrame({"x": [SUPPORTED[0]], "y": [SUPPORTED[1]]}, index=[42])
    got = score_dataframe(frame, "x", "y")
    assert got.index.tolist() == [42] and got.name == "tcrsmalldist"


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        score(["CASSLTGDYEQYF"], ["CASSLTGDWEQYF", "CASSLTGDWEQYF"])


def test_empty_input_is_fine():
    assert score([], []).shape == (0,)
