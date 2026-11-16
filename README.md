# Wanted but Out of Print: Screening Open Library for Books Entering the Public Domain in 2027 (BDCC Lab 2)

This project estimates which Open Library works are already in the public
domain, or enter it on January 1, 2027, in the United States
(publication-based term) and in the Philippines (life of the author plus
50 years). It then measures reader demand from reading-log shelves and
ratings and current supply from recent editions, publishers, digitized
copies and Filipino-language editions, and it produces a 2027 watchlist
of high-demand, low-supply titles together with a list of wanted titles
that are already free.

Rights classifications are estimates from catalogue metadata, not legal
advice.

## Data

The report reads one fixed folder on Jojie,
`/mnt/data/public/openlibrary/20221007`, which holds Open Library's
export of September 30, 2022 (47 GiB, seven files; all seven are used).
It is the only snapshot on Jojie with the reading-log and ratings dumps
and with separate works, editions and authors dumps, since the later
snapshots hold only the revision history (`ol_cdump`). Section 2.2 of the
report explains the choice. No data file is part of the submission.

## Files

| File | Purpose |
|---|---|
| `FINAL REPORT.ipynb` / `FINAL REPORT.html` | Technical report; runs top to bottom and regenerates every table and chart |
| `helper_code.py` | All processing, analysis and plotting functions (PySpark), with docstrings |
| `TEST.ipynb` / `TEST.html` | 73 unit and end-to-end tests on hand-made miniature dumps |
| `images/du_sch_terminal.png` | Screenshot of `du -sch` on the raw input files (Figure 1 of the report) |

## Running on Jojie

1. Put `FINAL REPORT.ipynb`, `helper_code.py`, `TEST.ipynb` and the
   `images/` folder in one folder and open the report.
2. Choose *Kernel > Restart Kernel and Run All Cells*. Run only **one**
   copy at a time, because a lock file stops a second run from writing
   the same `parquet/` folder.
3. The first run builds `parquet/` next to the notebook (about 5 GB) and
   takes two to three hours on a 4-CPU allowance. The five record dumps
   are decompressed to a scratch folder in `/tmp` (about 58 GB, deleted
   afterwards; set `OL_STAGING_DIR` to move it) and parsed in parallel,
   while the all-types dump and the 27.7 GB revision history are read
   straight from gzip, one Spark task per file (about 30 minutes for the
   largest). Later runs reuse the tables and take about half an hour.
4. If Spark runs out of memory, set `OL_SPARK_MEMORY` (default `10g`)
   before the Spark cell.
5. `parquet/manifest.json` records the snapshot and date rules the tables
   were built with, and a run with different settings refuses to reuse
   them. Delete `parquet/` to rebuild from scratch, which is also needed
   after changing `helper_code.py`.
6. Run `TEST.ipynb` to check the helper functions; it needs no data.
7. Do not submit or commit the `parquet/` folder.

Requirements: Python 3.10+, Java 17+, `pyspark pandas numpy matplotlib
pyarrow jupyter pycodestyle`.
