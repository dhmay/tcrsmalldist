"""Command line: score a table of CDR3 pairs.

    tcrsmalldist score pairs.tsv -o scored.tsv --cdrcol1 cdr3b_i --cdrcol2 cdr3b_j

Reads in chunks and writes as it goes, so a TCRdist export of tens of millions of rows does not
have to fit in memory.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import __version__
from .model import Model, score


def _chunks(path: str, separator: str, size: int):
    reader = pd.read_csv(path, sep=separator, chunksize=size)
    for chunk in reader:
        yield chunk


def run(path: str, out: str, cdrcol1: str, cdrcol2: str, separator: str = "\t",
        chunk_size: int = 1_000_000, max_tcrdist: float | None = None,
        tcrdist_column: str = "tcrdist") -> dict[str, int]:
    """Score `path` into `out`, adding a `tcrsmalldist` column. Returns a small tally."""
    model = Model.load()
    tally = {"rows": 0, "scored": 0, "unsupported": 0, "skipped": 0}
    header = True
    with open(out, "w") as handle:
        for chunk in _chunks(path, separator, chunk_size):
            for column in (cdrcol1, cdrcol2):
                if column not in chunk.columns:
                    raise SystemExit(f"column {column!r} is not in {path}; "
                                     f"found {', '.join(map(str, chunk.columns[:8]))}")
            tally["rows"] += len(chunk)
            wanted = np.ones(len(chunk), dtype=bool)
            if max_tcrdist is not None:
                if tcrdist_column not in chunk.columns:
                    raise SystemExit(f"--max-tcrdist needs a {tcrdist_column!r} column")
                wanted = (chunk[tcrdist_column] <= max_tcrdist).to_numpy()
                tally["skipped"] += int((~wanted).sum())
            values = np.full(len(chunk), np.nan)
            if wanted.any():
                values[wanted] = score(chunk[cdrcol1].to_numpy()[wanted],
                                       chunk[cdrcol2].to_numpy()[wanted], model=model)
            tally["scored"] += int(np.isfinite(values).sum())
            tally["unsupported"] += int(wanted.sum() - np.isfinite(values).sum())
            chunk.assign(tcrsmalldist=values).to_csv(handle, sep=separator, index=False,
                                                     header=header)
            header = False
    return tally


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcrsmalldist", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    scorer = sub.add_parser("score", help="add a tcrsmalldist column to a table of CDR3 pairs")
    scorer.add_argument("input")
    scorer.add_argument("-o", "--out", required=True)
    scorer.add_argument("--cdrcol1", default="cdr3b_i",
                        help="first CDR3 column (default: cdr3b_i)")
    scorer.add_argument("--cdrcol2", default="cdr3b_j",
                        help="second CDR3 column (default: cdr3b_j)")
    scorer.add_argument("--sep", default="\t", help="field separator (default: tab)")
    scorer.add_argument("--chunk-size", type=int, default=1_000_000)
    scorer.add_argument("--max-tcrdist", type=float,
                        help="skip pairs above this distance, leaving them unscored")
    scorer.add_argument("--tcrdist-column", default="tcrdist")

    args = parser.parse_args(argv)
    tally = run(args.input, args.out, args.cdrcol1, args.cdrcol2, args.sep, args.chunk_size,
                args.max_tcrdist, args.tcrdist_column)
    print(f"{tally['rows']:,} rows: {tally['scored']:,} scored, "
          f"{tally['unsupported']:,} outside the supported edit space"
          + (f", {tally['skipped']:,} above the distance cutoff" if tally["skipped"] else ""),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
