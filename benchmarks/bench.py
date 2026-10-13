"""Speed and peak memory for the scoring path, across input sizes and edit mixes.

Run as `uv run python benchmarks/bench.py`, which writes `benchmarks/results.tsv`.

**Every measurement runs in its own subprocess.** Peak RSS is a high-water mark that never falls,
so measuring several configurations in one process would report the largest of them for all of
them. Each child builds its input, warms the caches, measures, prints one row and exits.

**The synthetic input is drawn from a pool of a few thousand distinct CDR3 pairs**, so generating
sixteen million pairs costs a sample rather than sixteen million string constructions. Pairs
therefore repeat, and the object arrays hold references to a small set of strings - which makes
the *input* side of memory optimistic relative to real data. The scoring work is unaffected,
since nothing caches on sequence identity. Point `--input` at a real TSV to check.

Reported per row: pairs per second, and peak RSS above the post-warmup baseline, so the figure is
the cost of scoring rather than of the interpreter and the model table.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results.tsv"
ALPHABET = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
#: bytes on macOS, kilobytes on Linux
RSS_SCALE = 1e6 if sys.platform == "darwin" else 1e3
POOL = 20_000


def make_pairs(n: int, mix: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n pairs of the requested edit mix, sampled from a pool of distinct pairs."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(11, 17, size=POOL)
    stems = ["CASS" + "".join(rng.choice(ALPHABET, length - 5)) + "F" for length in lengths]
    left, right = [], []
    for index, stem in enumerate(stems):
        kind = mix if mix != "mixed" else ("substitution", "indel")[index % 2]
        cut = 6 + index % 4
        if kind == "substitution":
            partner = stem[:cut] + rng.choice(ALPHABET) + stem[cut + 1:]
        else:
            partner = stem[:cut] + stem[cut + 1:]
        left.append(stem)
        right.append(partner)
    take = rng.integers(0, POOL, size=n)
    return np.asarray(left, dtype=object)[take], np.asarray(right, dtype=object)[take]


def naive_score(cdr3_a, cdr3_b) -> np.ndarray:
    """A per-pair reference, to quantify what the vectorized path buys. Equal-length only."""
    from tcrsmalldist.features import columns

    from tcrsmalldist import Model

    model = Model.load()
    out = np.empty(len(cdr3_a))
    for index, (a, b) in enumerate(zip(cdr3_a, cdr3_b)):
        out[index] = model.score_columns(columns([a], [b]))[0]
    return out


def child(spec: dict) -> None:
    """One measurement. Prints a JSON row on stdout."""
    from tcrsmalldist import score, score_dataframe

    n, mix, style = spec["n"], spec["mix"], spec["style"]
    left, right = make_pairs(n, mix, spec.get("seed", 0))
    score(left[:16], right[:16])                     # warm the model table and the lookup arrays

    if style == "dataframe":
        import pandas as pd
        frame = pd.DataFrame({"cdr3b_i": left, "cdr3b_j": right})
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start = time.perf_counter()
    if style == "array":
        values = score(left, right, batch=spec["batch"])
    elif style == "unbatched":
        values = score(left, right, batch=max(n, 1))
    elif style == "dataframe":
        values = score_dataframe(frame).to_numpy()
    elif style == "naive":
        values = naive_score(left, right)
    else:
        raise SystemExit(f"unknown style {style!r}")
    elapsed = time.perf_counter() - start

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({**spec, "seconds": elapsed, "pairs_per_second": n / elapsed,
                      "peak_mb": (peak - baseline) / RSS_SCALE,
                      "bytes_per_pair": (peak - baseline) / RSS_SCALE * 1e6 / n,
                      "scored": float(np.isfinite(values).mean())}))


def stages(n: int = 2_000_000) -> list[dict]:
    """Where the time goes, measured in one process since only timing is wanted."""
    from tcrsmalldist.features import align, columns

    from tcrsmalldist import Model

    left, right = make_pairs(n, "mixed")
    model = Model.load()
    rows, marks = [], {}
    start = time.perf_counter()
    alignment = align(left, right)
    marks["align"] = time.perf_counter() - start
    start = time.perf_counter()
    built = columns(left, right, alignment)
    marks["columns"] = time.perf_counter() - start
    start = time.perf_counter()
    model.score_columns(built)
    marks["score_columns"] = time.perf_counter() - start
    for stage, seconds in marks.items():
        rows.append({"measurement": "stage", "stage": stage, "n": n, "seconds": seconds,
                     "share": seconds / sum(marks.values())})
    return rows


#: The ladder. The largest sizes are limited by the input arrays, not by scoring: a pair of
#: object arrays is 16 bytes of pointers per pair before any work happens, so an in-memory call
#: of tens of millions is impractical whatever the scorer does. The CLI streams instead.
LADDER = [100_000, 500_000, 2_000_000, 8_000_000, 16_000_000]


def plan(batch: int) -> list[dict]:
    specs = [{"measurement": "scale", "n": n, "mix": "mixed", "style": "array", "batch": batch}
             for n in LADDER]
    specs += [{"measurement": "mix", "n": 2_000_000, "mix": mix, "style": "array",
               "batch": batch} for mix in ("substitution", "indel")]
    specs += [{"measurement": "batching", "n": n, "mix": "mixed", "style": "unbatched",
               "batch": 0} for n in (2_000_000, 8_000_000)]
    specs += [{"measurement": "style", "n": 2_000_000, "mix": "mixed", "style": "dataframe",
               "batch": batch}]
    specs += [{"measurement": "naive", "n": 20_000, "mix": "substitution", "style": "naive",
               "batch": batch}]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, default=500_000)
    args = parser.parse_args()

    if args.run:
        child(json.loads(args.run))
        return

    rows = []
    for spec in plan(args.batch):
        label = f"{spec['measurement']}/{spec['style']}/{spec['mix']}/{spec['n']:,}"
        finished = subprocess.run([sys.executable, __file__, "--run", json.dumps(spec)],
                                  capture_output=True, text=True)
        if finished.returncode != 0:
            print(f"  {label:48s} FAILED\n{finished.stderr[-400:]}", file=sys.stderr)
            continue
        row = json.loads(finished.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"  {label:48s} {row['pairs_per_second']:>10,.0f} pairs/s   "
              f"{row['peak_mb']:>7.0f} MB peak   {row['bytes_per_pair']:>5.0f} B/pair",
              flush=True)

    rows += stages()
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS, sep="\t", index=False)
    print(f"\nwrote {RESULTS.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
