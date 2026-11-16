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
| `TEST.ipynb` / `TEST.html` | 52 unit and end-to-end tests on hand-made miniature dumps |
| `images/` | Spark UI screenshots (ingestion and analysis runs) |

## Running on Jojie (the course JupyterHub)

1. Upload `FINAL REPORT.ipynb`, `helper_code.py`, `TEST.ipynb` and the
   `images/` folder into **one folder** in your Jojie JupyterLab.
2. Open `FINAL REPORT.ipynb` and choose *Kernel > Restart Kernel and Run
   All Cells*. The notebook finds the Open Library dumps under
   `/mnt/data/public` on its own. If it cannot, run
   `ls /mnt/data/public` in a terminal to find the folder and set it in
   a new cell at the top with `import os` and
   `os.environ["OL_DATA_DIR"] = "<folder>"`, then Run All again.
3. The first run parses the dumps into `parquet/` next to the notebook
   (about an hour, roughly 7 GB). Later runs reuse those tables and take
   minutes. If Spark runs out of memory, set
   `os.environ["OL_SPARK_MEMORY"] = "6g"` before the Spark cell; if `/tmp`
   fills up, set `OL_SPARK_LOCAL_DIR` to a folder in your home directory.
4. Check the line printed under the Spark cell: the narrative quotes the
   31 October 2026 dumps, so a warning there means Jojie holds a different
   month and the numbers in the text must be updated.
5. Run `TEST.ipynb` to check the helper functions; it needs no data.
6. Do not submit or commit the `parquet/` folder.

## Running elsewhere

Install Python 3.10+, Java 17+ and
`pip install pyspark pandas numpy matplotlib pyarrow jupyter pycodestyle`,
download the six October 2026 dumps from
<https://openlibrary.org/developers/dumps> (mirrored at
<https://archive.org/details/ol_dump_2026-10-31>) into `data/`, and run
the notebook as above.

Neither raw nor processed data is committed (`data/` and `parquet/` are
git-ignored).
