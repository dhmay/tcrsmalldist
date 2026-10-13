"""Build the model from a table of coefficients and standardization parameters.
The model is a simple linear transformation of standardized feature columns,
and as such doesn't need to depend on scikit-learn.

`data/lr_coefficients.tsv` has one row per feature, carrying the standardizing mean and
scale beside the coefficient, plus an `(intercept)` row. 
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .features import align, columns

DATA = Path(__file__).parent / "data"
DEFAULT_TABLE = DATA / "lr_coefficients.tsv"


@dataclass(frozen=True)
class Model:
    """A fitted linear model over named, standardized feature columns."""

    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_table(cls, path: str | Path) -> "Model":
        table = pd.read_csv(path, sep="\t")
        is_intercept = table.feature == "(intercept)"
        weights = table[~is_intercept]
        return cls(features=tuple(weights.feature),
                   mean=weights["mean"].to_numpy(float),
                   scale=weights["scale"].to_numpy(float),
                   coef=weights.coef.to_numpy(float),
                   intercept=float(table.loc[is_intercept, "coef"].iloc[0]))

    @classmethod
    def load(cls) -> "Model":
        """The shipped model."""
        return _shipped()

    def score_columns(self, built: dict[str, np.ndarray]) -> np.ndarray:
        """Probabilities from already-built feature columns, selected in the model's own order.

        The columns are accumulated into the logit one at a time rather than stacked into a
        matrix first. Stacking would copy every column into a second array, roughly doubling
        peak memory for no benefit - the dot product never needs them contiguous.
        """
        missing = [name for name in self.features if name not in built]
        if missing:
            raise KeyError(f"missing {len(missing)} feature columns: {', '.join(missing[:4])}")
        logit = np.full(len(built[self.features[0]]), self.intercept, dtype=float)
        for name, mean, scale, coef in zip(self.features, self.mean, self.scale, self.coef):
            term = built[name] - mean
            term /= scale
            term *= coef
            logit += term
        return 1 / (1 + np.exp(-logit))


@lru_cache(maxsize=1)
def _shipped() -> Model:
    return Model.from_table(DEFAULT_TABLE)


#: Pairs per internal batch. The feature columns cost a few hundred bytes a pair while they
#: exist, so scoring tens of millions in one call would otherwise need tens of gigabytes.
#: Batching bounds peak memory by the batch rather than by the input.
BATCH = 500_000


def score(cdr3_a, cdr3_b, model: Model | None = None, batch: int = BATCH) -> np.ndarray:
    """Same-peptide probability for each pair of CDR3s. NaN where the pair is unsupported.

    Unsupported means outside the edit space the model was fitted on, e.g., identical CDR3s, more than
    one indel, or more than two substitutions. Those are returned as NaN.

    Input of any size is accepted: scoring runs in batches of `batch` pairs, so peak memory is
    set by the batch rather than by the number of pairs.
    """
    model = _shipped() if model is None else model
    a, b = np.asarray(cdr3_a, dtype=object), np.asarray(cdr3_b, dtype=object)
    if len(a) != len(b):
        raise ValueError(f"{len(a):,} and {len(b):,} CDR3s; the two sides must be the same length")

    out = np.empty(len(a), dtype=float)
    for start in range(0, len(a), batch):
        stop = min(start + batch, len(a))
        piece_a, piece_b = a[start:stop], b[start:stop]
        alignment = align(piece_a, piece_b)
        probability = model.score_columns(columns(piece_a, piece_b, alignment))
        out[start:stop] = np.where(alignment["supported"], probability, np.nan)
    return out


def features(cdr3_a, cdr3_b, model: Model | None = None) -> np.ndarray:
    """Return the feature matrix on a pair, with its columns in the order the model reads them.

    For inspecting what a pair looks like to the model, or for refitting on the same encoding.
    """
    model = _shipped() if model is None else model
    built = columns(cdr3_a, cdr3_b)
    return np.column_stack([built[name] for name in model.features])


def score_dataframe(frame: pd.DataFrame, cdrcol1: str = "cdr3b_i", cdrcol2: str = "cdr3b_j",
                model: Model | None = None) -> pd.Series:
    """`score` over two cdr3 columns of a DataFrame, returned as a Series on the frame's index."""
    return pd.Series(score(frame[cdrcol1].to_numpy(), frame[cdrcol2].to_numpy(), model=model),
                     index=frame.index, name="tcrsmalldist")
