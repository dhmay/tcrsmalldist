"""A calibrated same-peptide probability for TCR pairs that TCRdist calls close.

    from tcrsmalldist import score
    score(["CASSLYGGARYEQYF"], ["CASSLYGGARYEQFF"])

See the README for what the model predicts, which pairs it supports, and how to call it on
TCRdist output in bulk. The code lives in `features.py` (encoding and alignment) and `model.py`
(the coefficient table and scoring); this module only re-exports the public names.
"""

from .features import align, columns, edit_class
from .model import Model, features, score, score_dataframe

__all__ = ["score", "score_dataframe", "features", "edit_class", "Model", "align", "columns"]
__version__ = "0.1.0"
