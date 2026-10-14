@~/code/claude-scaffold/working-agreement.md

# tcrsmalldist

The public package: run the shipped model, and nothing else. The analysis behind the model -
feature selection, validation, the decks - lives in the sibling `tcrsmalldist-devel` repo, which
is also where `benchmarks/benchmark.ipynb` is turned into slides.

- `src/tcrsmalldist/` - `features.py` builds the 42 columns, vectorized by CDR3 length bucket;
  `model.py` reads `data/lr_coefficients.tsv` and scores; `cli.py` streams a TSV.
- `tests/test_reference.py` is the contract: 2,000 pairs scored by the research implementation,
  matched to 1e-10. Regenerate it from `tcrsmalldist-devel` if the model is ever refitted.
- Dependencies are numpy and pandas only. No scikit-learn, no pickle - that is why the model
  ships as a coefficient table carrying its own standardizing mean and scale.
