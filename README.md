# tcrsmalldist

A calibrated same-peptide probability for TCR pairs that TCRdist calls close together (tcrdist <= 24).

TCRdist returns a number for any pair of TCRs, but at short range it has little to say: two pairs
whose CDR3s differ by the same amino acid substitution(s) get the same distance regardless of where
in the CDR3 the substitution(s) sit(s), and an indel is charged a flat penalty regardless of
where it sits. 

`tcrsmalldist` provides a way to score these "small" distances relative to each other.
For a pair separated by one indel and/or one or two amino-acid substitutions (most but not all
the pairs with tcrdist <= 24), it returns the probability that the two receptors bind the same peptide.

The model is a logistic regression fitted on TCR pairs with tcrdist <= 24 from the Class I ImmuneCODE
MIRA data. Its 42 features encode the swap position(s) if any, the gap position if any, and the
amino acid swap or swaps (using BLOSUM62 and Atchley factors).

## Install

```bash
pip install tcrsmalldist
```

## Use it

```python
from tcrsmalldist import score

score(["CASSLYGGARYEQYF", "CASSPQLAGGNEQYF"],
      ["CASSLYGGARYEQFF", "CASSPQLAGGNEQY"])
# array([0.87, 0.31])
```

On a table of pairs, which is what TCRdist gives you:

```python
import pandas as pd
from tcrsmalldist import score_dataframe

pairs = pd.read_csv("tcrdist_output.tsv", sep="\t")
pairs["tcrsmalldist"] = score_dataframe(pairs, "cdr3b_i", "cdr3b_j")
```

Or from the command line:

```bash
tcrsmalldist score tcrdist_output.tsv -o scored.tsv --cdrcol1 cdr3b_i --cdrcol2 cdr3b_j
```

The CLI streams in chunks, so the input tsv doesn't have to fit in memory.

## Supported pairs

Scoring is defined for 0 or 1 indels and 0, 1 or 2 substitutions, requiring at least one
of those two types of difference. Anything else (identical CDR3s, two indels, 3+ substitutions)
returns `NaN`.

An unknown residue (`X`, lowercase, anything non-standard) will cause  the model to return `NaN`. 
Both CDR3s must also be strings of 7 to 25 standard amino acids, or the pair returns `NaN`.
The model was fitted on CDR3s of 8 to 21 residues, 98% of them
between 10 and 16, so scores near the edges of the allowed range are untested extrapolations.

```python
from tcrsmalldist import edit_class
edit_class(["CASSLTGDYEQYF"], ["CASSLTGDWEQYF"])   # (indels, substitutions) per pair
```

One edge case: a pair whose indel sits within four residues of the CDR3's
C-terminal end is scored as if it had no gap, because the model has no feature for a gap that
close to the end (there were no such pairs in the training data).

## Refining TCRdist output at scale

The core API takes and returns arrays, with no file I/O and no DataFrame in the hot path, so it
can drop into a data processing pipeline:

```python
import numpy as np
from tcrsmalldist import score

close = distances <= 24                       # whatever your neighbour search returns
refined = np.full(len(distances), np.nan)
refined[close] = score(cdr3_a[close], cdr3_b[close])
```

Alignment is vectorized by CDR3 length for efficiency on large inputs.

## The model

`src/tcrsmalldist/data/lr_coefficients.tsv` represents the model: one row per feature with its
standardizing mean, scale and coefficient, plus an intercept row.

The features are CDR3 length; substitution counts binned by distance from each end of the CDR3;
BLOSUM62 and Atchley-factor aggregates over the substituted residues; the gap's position binned
from the CDR3 end; and the gap's placement ambiguity.

```python
from tcrsmalldist import Model, features
Model.load().coef          # the fitted weights
features(a, b)             # the 42 columns, for inspection or refitting
```

## Provenance of third-party tables

Two third-party tables are redistributed in `src/tcrsmalldist/data/`:

- **`blosum62.csv`** - Henikoff & Henikoff 1992, from the matrix
  [NCBI distributes with BLAST](https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62).
- **`atchley_factors.csv`** - the five factors of Atchley et al. 2005, *PNAS* 102:6395-6400,
  table 2. Taken from a
  [machine-readable reproduction](https://raw.githubusercontent.com/vadimnazarov/kidera-atchley/master/atchley.txt)
  rather than from the publisher, and checked against the values quoted in the paper. This
  model uses three of the five factors: polarity, molecular size and charge.

## Acknowlegments

Software implementation and documentation were developed with assistance from Claude (Anthropic). The authors designed the analyses, verified all code and outputs, and take full responsibility for the work.