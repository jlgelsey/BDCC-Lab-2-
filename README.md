# Open Library Public Domain Opportunity Radar (BDCC Lab 2)

Estimates which Open Library works are, or will become on 1 January 2027,
public domain in the United States (publication-based terms) and the
Philippines (life + 50 years), measures reader demand (reading-log
shelves, ratings, Want-to-Read trends) and current supply (recent
editions, publishers, digitized copies, Filipino-language editions), and
produces a 2027 watchlist of high-demand, low-supply works.

Rights classifications are estimates from catalogue metadata, not legal
advice.

## Files

| File | Purpose |
|---|---|
| `FINAL REPORT.ipynb` / `FINAL REPORT.html` | Technical report; runs top to bottom and regenerates every table and chart |
| `helper_code.py` | All processing, analysis and plotting functions (PySpark), with docstrings |
| `TEST.ipynb` / `TEST.html` | 49 unit and end-to-end tests on hand-made miniature dumps |
| `images/` | Spark UI screenshots (ingestion and analysis runs) |

## Running

1. Install Python 3.10+, Java 17+, and
   `pip install pyspark pandas numpy matplotlib pyarrow jupyter pycodestyle`.
2. Download the October 2026 dumps (works, editions, authors, wikidata,
   reading-log, ratings) from <https://openlibrary.org/developers/dumps>
   (mirrored at <https://archive.org/details/ol_dump_2026-10-31>) into
   `data/`, or point `OL_DATA_DIR` at the folder that holds them.
3. Open `FINAL REPORT.ipynb` and choose *Restart & Run All*. The first
   run parses the dumps into `parquet/` (about an hour on 4 cores); later
   runs reuse those tables and take minutes.
4. Run `TEST.ipynb` to check the helper functions; it needs no dumps.

Neither raw nor processed data is committed (`data/` and `parquet/` are
git-ignored).
