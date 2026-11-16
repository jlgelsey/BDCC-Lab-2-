"""
Open Library Public Domain Opportunity Radar (Parts 1-10)
BDCC Lab 2 Core Module

This module holds every function the technical report calls. The report
notebook (FINAL REPORT.ipynb) only orchestrates these functions and writes
the narrative; the test notebook (TEST.ipynb) checks them on small,
hand-written examples whose correct answers are known in advance.

Install pyspark (with a Java 17+ runtime), pandas, numpy, matplotlib and
pyarrow first. Importing this module does not start Spark, read any dump,
or write any file.

Pipeline at a glance
--------------------
Part 1  Spark session, dump discovery and file inventory
Part 2  Parse the six monthly dumps and save cleaned Parquet tables
Part 3  Data-quality profile (coverage of the fields the analysis needs)
Part 4  Reconcile author death years (Open Library text vs Wikidata)
Part 5  Estimate copyright status per work (US, Philippines, life+70)
Part 6  Demand: reading-log shelves, ratings and Want-to-Read trends
Part 7  Supply: editions, recent editions, publishers, scans, Filipino
Part 8  2027 watchlist and potentially-public-domain list
Part 9  Validation: independent recounts of the published numbers
Part 10 Charts

Example
-------
>>> spark = create_spark()
>>> dumps = discover_dumps(resolve_data_dir())
>>> paths = build_parquet_tables(spark, dumps, resolve_parquet_dir())
>>> tables = load_tables(spark, paths)
>>> deaths = reconcile_author_deaths(tables["authors"], tables["wikidata"])
>>> status = estimate_copyright_status(tables["works"], tables["editions"],
...                                    deaths["author_deaths"])

Rights classifications are estimates computed from catalogue metadata.
They are not legal advice and must be checked against primary records
(for example US renewal records) before any reprint decision.
"""

from datetime import date
from pathlib import Path
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pyspark is imported lazily inside functions that need it, so that the
# pure-Python rule functions can be imported and read without Java.

SEED = 42

# The analysis date: rights are evaluated as they will stand on this day.
REFERENCE_DATE = date(2027, 1, 1)
REFERENCE_YEAR = REFERENCE_DATE.year

# The dump analysed: Open Library's export of 30 September 2022, which is
# the copy stored on Jojie in /mnt/data/public/openlibrary/20221007 (the
# folder name is the date it was downloaded). Every date-dependent setting
# below follows from this one date, so another dump only needs it changed.
DUMP_DATE = date(2022, 9, 30)

# A publication year after the dump year cannot be real.
MAX_VALID_YEAR = DUMP_DATE.year
MIN_VALID_YEAR = 1450  # printing with movable type in Europe

# "Recent" supply window: the ten calendar years ending with the dump year
# (2013-2022 for this dump).
RECENT_FROM_YEAR = DUMP_DATE.year - 9

# Filipino-language codes used by Open Library (MARC language codes).
FILIPINO_LANGUAGE_KEYS = ("/languages/tgl", "/languages/fil")

# Wikidata items that matter here.
WIKIDATA_PHILIPPINES = "Q928"

# Dumps the proposal lists. Covers, lists and revision history are excluded.
DUMP_NAMES = ("works", "editions", "authors", "wikidata", "reading-log",
              "ratings")

# The Jojie folder is listed first and by its full path: Jojie's shared
# drive is slow to list (about 15 s for this folder tree), so going straight
# to the right folder keeps the first cell fast.
JOJIE_DUMP_DIR = "/mnt/data/public/openlibrary/20221007"

DATA_DIR_CANDIDATES = (
    JOJIE_DUMP_DIR,
    "/mnt/data/public/openlibrary",
    "/mnt/data/public/open-library",
    "/mnt/data/public/open_library",
    "data",
)
# Shared data root on Jojie, the course's JupyterHub server. If none of the
# folders above holds the dumps, every folder up to two levels below this
# root is searched as well.
SHARED_DATA_ROOT = "/mnt/data/public"

# The dump whose numbers the report's narrative quotes, and the Jojie
# folder date of the same files. Running on another dump works, but the
# text would then need updating.
REPORT_DUMP_DATE = DUMP_DATE.isoformat()
REPORT_DUMP_ALIASES = (REPORT_DUMP_DATE, "2022-10-07")

# Dumps the analysis can run without. Jojie has no Wikidata dump, so
# without it every death year comes from Open Library's own records.
OPTIONAL_DUMPS = ("wikidata",)

DUMP_FILE_REGEX = (r"^ol_dump_{name}_(\d{{4}}-\d{{2}}-\d{{2}}|latest)"
                   r"\.txt(\.gz)?$")

SHELVES = ("Want to Read", "Currently Reading", "Already Read")

# Words that only describe a publisher's legal form. Removing them lets
# "Penguin Books Ltd." and "Penguin Books" count as one publisher.
PUBLISHER_LEGAL_SUFFIXES = (
    "inc", "incorporated", "ltd", "limited", "llc", "co", "corp",
    "corporation", "company", "gmbh", "sa", "plc", "pty",
)


# ---------------------------------------------------------------------------
# PART 1: SPARK SESSION, DUMP DISCOVERY AND FILE INVENTORY
# ---------------------------------------------------------------------------


def create_spark(app_name="ol-pd-radar", driver_memory=None,
                 shuffle_partitions=64, local_dir=None):
    """Start (or reuse) a local Spark session sized for one machine.

    Parameters
    ----------
    app_name : str
        Name shown in the Spark UI.
    driver_memory : str, optional
        JVM heap for the driver; in local mode the driver does all work.
        Defaults to the OL_SPARK_MEMORY environment variable, else "10g".
        Lower it (for example "6g") on a server with a smaller memory
        allowance per user, such as a shared JupyterHub.
    shuffle_partitions : int
        Number of partitions after joins and group-bys. 64 suits 4-16 cores.
    local_dir : str, optional
        Scratch folder for shuffle files; defaults to the OL_SPARK_LOCAL_DIR
        environment variable, else Spark's own choice (usually /tmp). Large
        joins write several GB here, so point it at a roomy disk.

    Returns
    -------
    SparkSession
    """
    from pyspark.sql import SparkSession

    driver_memory = driver_memory or os.environ.get("OL_SPARK_MEMORY", "10g")
    local_dir = local_dir or os.environ.get("OL_SPARK_LOCAL_DIR")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        # Parquet files are compressed column by column; zstd is compact.
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if local_dir:
        builder = builder.config("spark.local.dir", str(local_dir))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _has_dumps(path):
    """True if a folder directly contains Open Library dump files.

    os.scandir stops at the first matching name, so a folder that does
    hold the dumps is recognised without listing all of it.
    """
    try:
        with os.scandir(path) as entries:
            return any(e.name.startswith("ol_dump_") and ".txt" in e.name
                       for e in entries)
    except OSError:          # missing or unreadable folder
        return False


def _newest_dump_folder(path):
    """Return `path` if it holds dumps, else its newest subfolder that does.

    On Jojie the dumps are stored one level down, in folders named after
    the download date (201809, 20240420, 20260424, ...). Those names sort
    in date order, so the subfolders are tried from the last name back,
    and the newest month that actually holds dump files wins.
    """
    path = Path(path)
    if _has_dumps(path):
        return path
    try:
        subfolders = sorted((e.path for e in os.scandir(path)
                             if e.is_dir()), reverse=True)
    except OSError:
        return None
    for sub in subfolders:
        if _has_dumps(sub):
            return Path(sub)
    return None


def resolve_data_dir(custom_dir=None, shared_root=SHARED_DATA_ROOT,
                     verbose=True):
    """Find the folder that holds the ol_dump_* files.

    The search order is: an explicit argument, the OL_DATA_DIR environment
    variable, the folders in DATA_DIR_CANDIDATES, and finally the folders
    under `shared_root` (the class data root on Jojie) whose names look
    like Open Library, plus their direct subfolders. Only those folders
    are opened: listing every dataset on a shared server can take many
    minutes, because some folders hold millions of files. An explicit
    folder that does not exist raises an error instead of silently
    falling back to another one.

    Returns
    -------
    Path
    """
    if custom_dir:
        path = Path(custom_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Data folder not found: {path}")
        return path
    env_dir = os.environ.get("OL_DATA_DIR")
    candidates = ([env_dir] if env_dir else []) + list(DATA_DIR_CANDIDATES)
    for candidate in candidates:
        found = _newest_dump_folder(candidate)
        if found is not None:
            return found
    root = Path(shared_root)
    looks_like_ol = re.compile(r"open.?lib|ol.?dump|^ol$|^ol[_-]", re.I)
    tried = []
    if root.is_dir():
        try:
            names = sorted(entry.name for entry in os.scandir(root)
                           if entry.is_dir())
        except OSError:
            names = []
        for name in names:
            if not looks_like_ol.search(name):
                continue
            folder = root / name
            tried.append(str(folder))
            if verbose:
                print(f"Looking for dumps in {folder} ...")
            found = _newest_dump_folder(folder)
            if found is not None:
                return found
    raise FileNotFoundError(
        "No Open Library dumps (ol_dump_*.txt.gz) found"
        + (f" in {', '.join(tried)}" if tried else "")
        + ". Set OL_DATA_DIR to the folder that holds them, or place them "
        "in ./data"
    )


def resolve_parquet_dir(custom_dir=None):
    """Return the folder for cleaned Parquet tables (created if needed).

    Uses an explicit argument, then OL_PARQUET_DIR, then ./parquet.
    """
    path = Path(custom_dir or os.environ.get("OL_PARQUET_DIR", "parquet"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_dumps(data_dir, names=DUMP_NAMES, required=True):
    """Pick the newest dump file for each requested dump type.

    File names follow ol_dump_<name>_<YYYY-MM-DD>.txt.gz; the ISO date sorts
    correctly as text, so the newest dated file is chosen. Files named
    ol_dump_<name>_latest.txt(.gz) and uncompressed .txt copies are also
    accepted, because shared servers sometimes store the dumps that way.

    Parameters
    ----------
    data_dir : str or Path
    names : iterable of str
    required : bool
        If True, a missing dump raises FileNotFoundError. If False it is
        simply left out; build_parquet_tables then only complains when the
        table has not already been saved as Parquet.

    Returns
    -------
    dict
        {name: Path}
    """
    data_dir = Path(data_dir)
    found = {}
    for name in names:
        pattern = re.compile(DUMP_FILE_REGEX.format(name=re.escape(name)))
        matches = [p for p in data_dir.iterdir() if pattern.match(p.name)]
        # newest dated file wins; a "latest" file is used only if no dated
        # file exists, and a .gz file beats an uncompressed copy
        matches.sort(key=lambda p: (pattern.match(p.name).group(1) != "latest",
                                    pattern.match(p.name).group(1),
                                    p.name.endswith(".gz")))
        if matches:
            found[name] = matches[-1]
        elif required:
            raise FileNotFoundError(f"No {name} dump in {data_dir}")
    return found


def _folder_size(path):
    """Total size in bytes of all files under a folder (0 if absent)."""
    path = Path(path)
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def dump_inventory(dumps, parquet_dir=None, names=DUMP_NAMES):
    """Tabulate each dump: file, date, compressed size and Parquet status.

    No dump is read; only file names and sizes are inspected. A raw dump
    may be absent once its Parquet table exists (the raw files can be
    deleted after parsing to save disk space).

    Returns
    -------
    DataFrame
        One row per dump name with dump_date, raw_gb, parquet_gb and
        parquet_ready.
    """
    rows = []
    for name in names:
        path = dumps.get(name)
        row = {
            "dump": name,
            "file": path.name if path else "(not present)",
            "dump_date": dump_date_of(path) if path else None,
            "raw_gb": round(path.stat().st_size / 1e9, 3) if path else None,
        }
        if parquet_dir is not None:
            folder = Path(parquet_dir) / name.replace("-", "_")
            row["parquet_gb"] = round(_folder_size(folder) / 1e9, 3)
            row["parquet_ready"] = (folder / "_SUCCESS").is_file()
        rows.append(row)
    return pd.DataFrame(rows)


def dump_date_of(path):
    """Date of a dump file as YYYY-MM-DD, or None if it cannot be told.

    Dated files carry the date in their name. Files named "latest" (as on
    Jojie) take it from their folder when the folder is named YYYYMMDD.
    """
    path = Path(path)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    folder = path.parent.name
    if re.fullmatch(r"\d{8}", folder):
        return f"{folder[:4]}-{folder[4:6]}-{folder[6:]}"
    return None


def check_dump_date(dumps, expected=REPORT_DUMP_ALIASES):
    """Compare the dumps' dates with the dump the report's text describes.

    Returns a short message: confirmation when every dump matches one of
    `expected` (the export date, or the Jojie folder date of the same
    files), otherwise a warning naming the dates found.
    """
    dates = sorted({dump_date_of(path) or "unknown"
                    for path in dumps.values()})
    if dates and all(d in expected for d in dates):
        return (f"Dumps dated {', '.join(dates)}: the {expected[0]} export "
                "described in the text.")
    return (
        f"WARNING: dumps dated {', '.join(dates) or 'unknown'} were found, "
        f"but the report's text quotes the {expected[0]} dump. The tables "
        "and charts will reflect the new data; update the numbers in the "
        "narrative before submitting."
    )


# ---------------------------------------------------------------------------
# PART 2: PARSING THE DUMPS INTO CLEANED PARQUET TABLES
# ---------------------------------------------------------------------------
#
# Every dump except wikidata, reading-log and ratings is a tab-separated text
# file with five columns: type, key, revision, last_modified, JSON record.
# We read each line as text, split on tabs, and parse only the JSON fields we
# need with an explicit schema. An explicit schema is faster than letting
# Spark guess, and it guarantees the same columns every month.
#
# A .gz file cannot be split, so Spark reads each dump in a single task.
# That first pass is the slow part; once the Parquet files exist, every
# later step reads them in parallel and in seconds.

YEAR_REGEX = r"(?<![0-9])(1[4-9][0-9]{2}|20[0-9]{2})(?![0-9])"


def parse_year(text, min_year=MIN_VALID_YEAR, max_year=MAX_VALID_YEAR):
    """Pure-Python twin of year_from_text, used by the tests.

    Takes the FIRST four-digit year between 1400 and 2099 in a free-text
    date such as "February 27, 1997" or "c1931", then rejects years outside
    [min_year, max_year]. Returns an int or None.
    """
    if text is None:
        return None
    match = re.search(YEAR_REGEX, str(text))
    if not match:
        return None
    year = int(match.group(1))
    return year if min_year <= year <= max_year else None


def year_from_text(column, min_year=MIN_VALID_YEAR, max_year=MAX_VALID_YEAR):
    """Spark expression: first plausible four-digit year in a text column.

    Mirrors parse_year exactly; values outside the valid range become null.
    """
    from pyspark.sql import functions as F

    raw = F.regexp_extract(column, YEAR_REGEX, 1)
    year = F.when(raw != "", raw.cast("int"))
    return F.when((year >= min_year) & (year <= max_year), year)


def parse_life_year(text):
    """Pure-Python twin of life_year_from_text (birth or death dates).

    Author dates are messy ("1901.7.", "1963)", "c. 1850 or 1851",
    "384 BC"). We take the LARGEST 3-4 digit number, because a later death
    year means longer protection, so the estimate errs towards "still in
    copyright". Text containing "BC"/"B.C." gives a negative year.
    Returns an int or None.
    """
    if text is None:
        return None
    text = str(text)
    numbers = [int(n) for n in re.findall(r"(?<![0-9])([0-9]{3,4})(?![0-9])",
                                          text)]
    numbers = [n for n in numbers if n <= MAX_VALID_YEAR]
    if not numbers:
        return None
    if re.search(r"\bB\.?\s?C\.?(?![a-z])", text, flags=re.IGNORECASE):
        return -min(numbers)
    return max(numbers)


def life_year_from_text(column):
    """Spark expression mirroring parse_life_year."""
    from pyspark.sql import functions as F

    numbers = F.filter(
        F.transform(
            F.regexp_extract_all(
                column, F.lit(r"(?<![0-9])([0-9]{3,4})(?![0-9])"), 1
            ),
            lambda n: n.cast("int"),
        ),
        lambda n: n <= MAX_VALID_YEAR,
    )
    is_bc = F.coalesce(
        column.rlike(r"(?i)\bB\.?\s?C\.?(?![a-z])"), F.lit(False)
    )
    return F.when(F.size(numbers) > 0, F.when(
        is_bc, -F.array_min(numbers)
    ).otherwise(F.array_max(numbers)))


def normalize_publisher(name):
    """Pure-Python twin of publisher_key: a comparable publisher name.

    Lowercases, turns punctuation into spaces, drops legal-form words such
    as "Ltd." or "Inc.", and collapses spaces. Returns "" for empty input.
    """
    if name is None:
        return ""
    text = re.sub(r"[^0-9a-z]+", " ", str(name).lower())
    words = [w for w in text.split() if w not in PUBLISHER_LEGAL_SUFFIXES]
    return " ".join(words)


def publisher_key(column):
    """Spark expression mirroring normalize_publisher."""
    from pyspark.sql import functions as F

    words = F.split(
        F.trim(F.regexp_replace(F.lower(column), r"[^0-9a-z]+", " ")), " "
    )
    kept = F.filter(
        words,
        lambda w: (w != "") & ~w.isin(*PUBLISHER_LEGAL_SUFFIXES),
    )
    return F.array_join(kept, " ")


def read_record_dump(spark, path, schema):
    """Read a five-column Open Library dump and parse its JSON column.

    Parameters
    ----------
    spark : SparkSession
    path : str or Path
        A works, editions or authors .txt.gz file.
    schema : StructType
        Only the JSON fields to keep; all other fields are ignored.

    Returns
    -------
    DataFrame
        Columns record_type, key, revision, last_modified, plus `j`
        (the parsed struct) and `json_ok` (False if the JSON was malformed).
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    # Every record carries its own "key". With PERMISSIVE parsing Spark
    # returns a struct of nulls (not a null) for malformed JSON, so a
    # missing key is the reliable sign that the record did not parse.
    schema = T.StructType(
        list(schema.fields) + [T.StructField("key", T.StringType())]
    )
    lines = spark.read.text(str(path))
    parts = F.split(F.col("value"), "\t", 5)
    parsed = lines.select(
        parts.getItem(0).alias("record_type"),
        parts.getItem(1).alias("key"),
        parts.getItem(2).cast("int").alias("revision"),
        parts.getItem(3).alias("last_modified"),
        F.from_json(parts.getItem(4), schema,
                    {"mode": "PERMISSIVE"}).alias("j"),
    )
    return parsed.withColumn("json_ok", F.col("j.key").isNotNull())


def works_schema():
    """JSON fields kept from the works dump."""
    from pyspark.sql import types as T

    author_role = T.StructType([
        T.StructField("author", T.StructType([
            T.StructField("key", T.StringType())
        ])),
    ])
    return T.StructType([
        T.StructField("title", T.StringType()),
        T.StructField("first_publish_date", T.StringType()),
        T.StructField("authors", T.ArrayType(author_role)),
        T.StructField("subjects", T.ArrayType(T.StringType())),
    ])


def editions_schema():
    """JSON fields kept from the editions dump."""
    from pyspark.sql import types as T

    key_struct = T.StructType([T.StructField("key", T.StringType())])
    return T.StructType([
        T.StructField("title", T.StringType()),
        T.StructField("publish_date", T.StringType()),
        T.StructField("publishers", T.ArrayType(T.StringType())),
        T.StructField("languages", T.ArrayType(key_struct)),
        T.StructField("works", T.ArrayType(key_struct)),
        T.StructField("ocaid", T.StringType()),
    ])


def authors_schema():
    """JSON fields kept from the authors dump."""
    from pyspark.sql import types as T

    return T.StructType([
        T.StructField("name", T.StringType()),
        T.StructField("birth_date", T.StringType()),
        T.StructField("death_date", T.StringType()),
        T.StructField("entity_type", T.StringType()),
        T.StructField("remote_ids", T.StructType([
            T.StructField("wikidata", T.StringType()),
        ])),
    ])


def wikidata_schema():
    """Wikidata statements kept: P569 birth, P570 death, P648 Open Library
    author ID, P27 country of citizenship. Each statement keeps its rank so
    deprecated (known-wrong) values can be dropped."""
    from pyspark.sql import types as T

    def statements(content_type):
        return T.ArrayType(T.StructType([
            T.StructField("rank", T.StringType()),
            T.StructField("value", T.StructType([
                T.StructField("content", content_type),
            ])),
        ]))

    time_type = T.StructType([
        T.StructField("time", T.StringType()),
        T.StructField("precision", T.IntegerType()),
    ])
    return T.StructType([
        T.StructField("id", T.StringType()),
        T.StructField("statements", T.StructType([
            T.StructField("P569", statements(time_type)),
            T.StructField("P570", statements(time_type)),
            T.StructField("P648", statements(T.StringType())),
            T.StructField("P27", statements(T.StringType())),
        ])),
    ])


def parse_works(spark, path):
    """Clean works table: one row per /type/work record.

    Keeps the author key list (in credit order), the work-level
    first_publish_date text and year, and the subject count. Records whose
    JSON fails to parse are kept with json_ok = False so they can be
    counted in the quality profile instead of disappearing.
    """
    from pyspark.sql import functions as F

    df = read_record_dump(spark, path, works_schema())
    df = df.filter(F.col("record_type") == "/type/work")
    author_keys = F.filter(
        F.transform(F.col("j.authors"), lambda a: a["author"]["key"]),
        lambda k: k.isNotNull(),
    )
    return df.select(
        F.col("key").alias("work_key"),
        F.col("j.title").alias("title"),
        F.array_distinct(F.coalesce(author_keys, F.array())).alias(
            "author_keys"
        ),
        F.col("j.first_publish_date").alias("first_publish_date"),
        year_from_text(F.col("j.first_publish_date")).alias("work_year"),
        F.coalesce(F.size(F.col("j.subjects")), F.lit(0)).alias(
            "n_subjects"
        ),
        F.when(
            F.exists(
                F.coalesce(F.col("j.subjects"), F.array()),
                lambda s: F.lower(s).rlike(r"\bfiction\b"),
            ),
            True,
        ).otherwise(False).alias("is_fiction"),
        "json_ok",
    )


def parse_editions(spark, path):
    """Clean editions table: one row per /type/edition record.

    publish_year is the first plausible year in publish_date. Editions
    can point to several works; we keep the first, which is the one Open
    Library displays. Publisher names are kept raw and as a normalized key.
    """
    from pyspark.sql import functions as F

    df = read_record_dump(spark, path, editions_schema())
    df = df.filter(F.col("record_type") == "/type/edition")
    language_keys = F.transform(F.col("j.languages"), lambda x: x["key"])
    publishers = F.coalesce(F.col("j.publishers"), F.array())
    return df.select(
        F.col("key").alias("edition_key"),
        F.col("j.works").getItem(0).getField("key").alias("work_key"),
        F.col("j.publish_date").alias("publish_date"),
        year_from_text(F.col("j.publish_date")).alias("publish_year"),
        publishers.alias("publishers"),
        F.array_distinct(
            F.filter(F.transform(publishers, publisher_key),
                     lambda p: p != "")
        ).alias("publisher_keys"),
        F.coalesce(language_keys, F.array()).alias("language_keys"),
        F.col("j.ocaid").alias("ocaid"),
        "json_ok",
    )


def parse_authors(spark, path):
    """Clean authors table with parsed birth and death years."""
    from pyspark.sql import functions as F

    df = read_record_dump(spark, path, authors_schema())
    df = df.filter(F.col("record_type") == "/type/author")
    return df.select(
        F.col("key").alias("author_key"),
        F.col("j.name").alias("name"),
        F.col("j.entity_type").alias("entity_type"),
        F.col("j.birth_date").alias("birth_date"),
        F.col("j.death_date").alias("death_date"),
        life_year_from_text(F.col("j.birth_date")).alias("ol_birth_year"),
        life_year_from_text(F.col("j.death_date")).alias("ol_death_year"),
        F.col("j.remote_ids.wikidata").alias("wikidata_id"),
        "json_ok",
    )


def parse_wikidata_time_year(time_text):
    """Pure-Python twin of the Wikidata year rule, used by the tests.

    Wikidata times look like "+1931-05-03T00:00:00Z" or
    "-0384-00-00T00:00:00Z" (BCE). Returns the signed year or None.
    """
    if time_text is None:
        return None
    match = re.match(r"^([+-])0*([0-9]+)-", str(time_text))
    if not match:
        return None
    year = int(match.group(2))
    return -year if match.group(1) == "-" else year


def wikidata_years(statements, min_precision=9):
    """Spark expression: list of signed years from Wikidata time statements.

    Statements ranked "deprecated" are dropped, and so are values less
    precise than a year (precision < 9 means decade, century, ...).
    """
    from pyspark.sql import functions as F

    kept = F.filter(
        F.coalesce(statements, F.array()),
        lambda s: (F.coalesce(s["rank"], F.lit("normal")) != "deprecated")
        & (s["value"]["content"]["precision"] >= min_precision),
    )

    def to_year(s):
        time_text = s["value"]["content"]["time"]
        sign = F.regexp_extract(time_text, r"^([+-])", 1)
        digits = F.regexp_extract(time_text, r"^[+-]0*([0-9]+)-", 1)
        year = F.when(digits != "", digits.cast("int"))
        return F.when(sign == "-", -year).otherwise(year)

    return F.filter(F.transform(kept, to_year), lambda y: y.isNotNull())


def parse_wikidata(spark, path):
    """Clean Wikidata table: one row per Wikidata item in the dump.

    The dump has three tab-separated columns: Q-id, the entity JSON wrapped
    in CSV-style quotes (inner quotes doubled), and a timestamp. We unwrap
    the quotes and parse the four statements we need. Open Library author
    IDs are P648 values ending in "A" (edition and work IDs end in M / W).
    Items without such an ID are kept, because an Open Library author
    record can still point to them through remote_ids.wikidata.
    """
    from pyspark.sql import functions as F

    lines = spark.read.text(str(path))
    parts = F.split(F.col("value"), "\t", 3)
    raw_json = parts.getItem(1)
    unwrapped = F.regexp_replace(
        F.regexp_replace(raw_json, r'^"|"$', ""), '""', '"'
    )
    df = lines.select(
        parts.getItem(0).alias("qid"),
        F.from_json(unwrapped, wikidata_schema()).alias("j"),
    )

    def not_deprecated(s):
        return F.coalesce(s["rank"], F.lit("normal")) != "deprecated"

    ol_ids = F.filter(
        F.transform(
            F.filter(F.coalesce(F.col("j.statements.P648"), F.array()),
                     not_deprecated),
            lambda s: s["value"]["content"],
        ),
        lambda v: v.rlike(r"^OL[0-9]+A$"),
    )
    countries = F.transform(
        F.filter(F.coalesce(F.col("j.statements.P27"), F.array()),
                 not_deprecated),
        lambda s: s["value"]["content"],
    )
    birth_years = wikidata_years(F.col("j.statements.P569"))
    death_years = wikidata_years(F.col("j.statements.P570"))
    return df.select(
        "qid",
        F.array_distinct(ol_ids).alias("ol_author_ids"),
        F.when(F.size(birth_years) > 0, F.array_max(birth_years)).alias(
            "wd_birth_year"
        ),
        F.when(F.size(death_years) > 0, F.array_max(death_years)).alias(
            "wd_death_year"
        ),
        F.when(F.size(death_years) > 0, F.array_min(death_years)).alias(
            "wd_death_year_min"
        ),
        F.array_contains(countries, WIKIDATA_PHILIPPINES).alias(
            "wd_philippine_citizen"
        ),
        # malformed JSON gives a struct of nulls, so test a field
        F.col("j.id").isNotNull().alias("json_ok"),
    )


def parse_reading_log(spark, path):
    """Reading-log events: work, edition, shelf and the date it was logged.

    The dump has no user column, so each row is one shelf entry by one
    (anonymous) reader. Only the three standard shelves are kept.
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    schema = T.StructType([
        T.StructField("work_key", T.StringType()),
        T.StructField("edition_key", T.StringType()),
        T.StructField("shelf", T.StringType()),
        T.StructField("logged_date", T.StringType()),
    ])
    df = spark.read.csv(str(path), sep="\t", schema=schema,
                        nullValue="\\N", quote="")
    return df.select(
        "work_key",
        "edition_key",
        F.trim("shelf").alias("shelf"),
        F.to_date("logged_date").alias("logged_date"),
    ).filter(F.col("shelf").isin(*SHELVES) & F.col("work_key").isNotNull())


def parse_ratings(spark, path):
    """Ratings: work, edition, star rating (1-5) and the date it was given."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    schema = T.StructType([
        T.StructField("work_key", T.StringType()),
        T.StructField("edition_key", T.StringType()),
        T.StructField("rating", T.StringType()),
        T.StructField("rated_date", T.StringType()),
    ])
    df = spark.read.csv(str(path), sep="\t", schema=schema,
                        nullValue="\\N", quote="")
    rating = F.col("rating").cast("int")
    return df.select(
        "work_key",
        "edition_key",
        F.when(rating.between(1, 5), rating).alias("rating"),
        F.to_date("rated_date").alias("rated_date"),
    ).filter(F.col("work_key").isNotNull())


def wikidata_output_schema():
    """Columns of the cleaned Wikidata table, used for an empty table when
    the Wikidata dump is not available."""
    from pyspark.sql import types as T

    return T.StructType([
        T.StructField("qid", T.StringType()),
        T.StructField("ol_author_ids", T.ArrayType(T.StringType())),
        T.StructField("wd_birth_year", T.IntegerType()),
        T.StructField("wd_death_year", T.IntegerType()),
        T.StructField("wd_death_year_min", T.IntegerType()),
        T.StructField("wd_philippine_citizen", T.BooleanType()),
        T.StructField("json_ok", T.BooleanType()),
    ])


EMPTY_SCHEMAS = {"wikidata": wikidata_output_schema}

PARSERS = {
    "works": parse_works,
    "editions": parse_editions,
    "authors": parse_authors,
    "wikidata": parse_wikidata,
    "reading-log": parse_reading_log,
    "ratings": parse_ratings,
}


def build_parquet_tables(spark, dumps, parquet_dir, overwrite=False,
                         parallel=True):
    """Parse each dump once and save it as Parquet.

    A table whose folder already contains a _SUCCESS marker is reused, so a
    second "Run All" of the report takes minutes rather than hours. Pass
    overwrite=True after downloading a newer dump.

    Because each .gz file is read by a single Spark task, parsing the dumps
    one after another would leave most CPU cores idle. With parallel=True
    the dumps are submitted as concurrent Spark jobs from separate Python
    threads, so the total time is roughly that of the largest dump.

    Returns
    -------
    dict
        {name: Path to the Parquet folder}
    """
    from concurrent.futures import ThreadPoolExecutor

    parquet_dir = Path(parquet_dir)
    out = {name: parquet_dir / name.replace("-", "_") for name in DUMP_NAMES}
    todo = [
        name for name in DUMP_NAMES
        if overwrite or not (out[name] / "_SUCCESS").is_file()
    ]
    missing = [name for name in todo
               if name not in dumps and name not in OPTIONAL_DUMPS]
    if missing:
        raise FileNotFoundError(
            f"No Parquet table and no raw dump for: {missing}"
        )
    # An optional dump that is absent (Wikidata on Jojie) is saved as an
    # empty table with the usual columns, so every later join still works
    # and simply finds no matches.
    for name in [n for n in todo if n not in dumps]:
        spark.createDataFrame([], EMPTY_SCHEMAS[name]()).write.mode(
            "overwrite").parquet(str(out[name]))
    todo = [name for name in todo if name in dumps]

    def write(name):
        table = PARSERS[name](spark, dumps[name])
        # Split the output into files of at most 2 million rows without a
        # shuffle; a shuffle would copy the whole table to disk again.
        (table.write.mode("overwrite")
         .option("maxRecordsPerFile", 2_000_000)
         .parquet(str(out[name])))
        return name

    # Largest dumps first, so they start as early as possible.
    todo.sort(key=lambda n: -Path(dumps[n]).stat().st_size)
    if parallel and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            list(pool.map(write, todo))
    else:
        for name in todo:
            write(name)
    return out


def materialize(spark, df, path, overwrite=False):
    """Save an intermediate result as Parquet once, then read it back.

    Spark is lazy: without this, every later count or chart would redo the
    joins that produced `df`. Writing it once and reading the files back
    is faster than an in-memory cache for tables with tens of millions of
    rows, and it survives a kernel restart.
    """
    path = Path(path)
    if overwrite or not (path / "_SUCCESS").is_file():
        df.write.mode("overwrite").parquet(str(path))
    return spark.read.parquet(str(path))


def load_tables(spark, parquet_paths):
    """Open every Parquet table; names use underscores (reading_log)."""
    return {
        name.replace("-", "_"): spark.read.parquet(str(path))
        for name, path in parquet_paths.items()
    }


# ---------------------------------------------------------------------------
# PART 3: DATA-QUALITY PROFILE
# ---------------------------------------------------------------------------


def profile_tables(tables):
    """Count rows and the coverage of the fields this analysis depends on.

    Coverage is the share of rows where a field is present and usable
    (for example a publish_date from which a valid year could be parsed).
    One Spark pass per table computes every count at once.

    Returns
    -------
    DataFrame
        table, field, rows, non_null, coverage_pct.
    """
    from pyspark.sql import functions as F

    checks = {
        "works": {
            "json parsed": F.col("json_ok"),
            "title": F.col("title").isNotNull(),
            "at least one author key": F.size("author_keys") > 0,
            "first_publish_date text": F.col("first_publish_date")
            .isNotNull(),
            "first_publish_date valid year": F.col("work_year").isNotNull(),
        },
        "editions": {
            "json parsed": F.col("json_ok"),
            "linked to a work": F.col("work_key").isNotNull(),
            "publish_date text": F.col("publish_date").isNotNull(),
            "publish_date valid year": F.col("publish_year").isNotNull(),
            "publisher": F.size("publisher_keys") > 0,
            "language": F.size("language_keys") > 0,
            "digitized (ocaid)": F.col("ocaid").isNotNull(),
        },
        "authors": {
            "json parsed": F.col("json_ok"),
            "name": F.col("name").isNotNull(),
            "birth_date text": F.col("birth_date").isNotNull(),
            "death_date text": F.col("death_date").isNotNull(),
            "death_date valid year": F.col("ol_death_year").isNotNull(),
            "wikidata link": F.col("wikidata_id").isNotNull(),
        },
        "wikidata": {
            "json parsed": F.col("json_ok"),
            "birth year (P569)": F.col("wd_birth_year").isNotNull(),
            "death year (P570)": F.col("wd_death_year").isNotNull(),
            "Philippine citizen (P27)": F.col("wd_philippine_citizen"),
        },
        "reading_log": {
            "edition given": F.col("edition_key").isNotNull(),
            "valid date": F.col("logged_date").isNotNull(),
        },
        "ratings": {
            "valid 1-5 rating": F.col("rating").isNotNull(),
            "valid date": F.col("rated_date").isNotNull(),
        },
    }
    rows = []
    for name, fields in checks.items():
        if name not in tables:
            continue
        aggs = [F.count(F.lit(1)).alias("__rows")] + [
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"c{i}")
            for i, cond in enumerate(fields.values())
        ]
        result = tables[name].agg(*aggs).collect()[0]
        total = result["__rows"]
        for i, field in enumerate(fields):
            count = result[f"c{i}"] or 0
            rows.append({
                "table": name,
                "field": field,
                "rows": total,
                "non_null": count,
                "coverage_pct": round(100 * count / total, 2) if total else
                np.nan,
            })
    return pd.DataFrame(rows)


def find_duplicate_keys(tables):
    """Count keys that appear more than once in each record table.

    Each Open Library key should be unique in a monthly dump; a non-zero
    count would double-count editions or authors in later joins.
    """
    from pyspark.sql import functions as F

    key_columns = {"works": "work_key", "editions": "edition_key",
                   "authors": "author_key"}
    rows = []
    for name, column in key_columns.items():
        dupes = (tables[name].groupBy(column).count()
                 .filter(F.col("count") > 1).count())
        rows.append({"table": name, "duplicated_keys": dupes})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART 4: AUTHOR DEATH YEARS (OPEN LIBRARY TEXT vs WIKIDATA)
# ---------------------------------------------------------------------------


def choose_death_year(ol_year, wd_year):
    """Pure-Python twin of the death-year rule used in Spark.

    Wikidata is structured (a typed date with a precision), so it wins when
    present; otherwise we fall back to the year parsed from Open Library's
    free-text death_date. Returns (year, source) with source one of
    "wikidata", "open_library" or None.
    """
    if wd_year is not None:
        return wd_year, "wikidata"
    if ol_year is not None:
        return ol_year, "open_library"
    return None, None


def reconcile_author_deaths(authors, wikidata):
    """Attach one death year (and birth year) to every Open Library author.

    Links an author to Wikidata in BOTH directions: the author record's
    remote_ids.wikidata Q-id, and Wikidata items whose P648 lists the
    author's OL ID. When several Wikidata items match, the latest death
    year is used (the conservative choice). Agreement between the two
    sources is measured on authors that have both.

    Returns
    -------
    dict
        author_deaths (Spark DataFrame, one row per author) and
        source_summary / agreement (pandas tables for the report).
    """
    from pyspark.sql import functions as F

    wd_by_ol = (
        wikidata.select(
            F.explode("ol_author_ids").alias("ol_id"), "qid",
            "wd_birth_year", "wd_death_year", "wd_philippine_citizen",
        )
        .withColumn("author_key", F.concat(F.lit("/authors/"), "ol_id"))
        .select("author_key", "qid", "wd_birth_year", "wd_death_year",
                "wd_philippine_citizen")
    )
    wd_by_qid = (
        authors.filter(F.col("wikidata_id").isNotNull())
        .select("author_key", F.col("wikidata_id").alias("qid"))
        .join(wikidata.select("qid", "wd_birth_year", "wd_death_year",
                              "wd_philippine_citizen"), "qid")
        .select("author_key", "qid", "wd_birth_year", "wd_death_year",
                "wd_philippine_citizen")
    )
    wd_links = (
        wd_by_ol.unionByName(wd_by_qid)
        .groupBy("author_key")
        .agg(
            F.countDistinct("qid").alias("n_wikidata_items"),
            F.max("wd_birth_year").alias("wd_birth_year"),
            F.max("wd_death_year").alias("wd_death_year"),
            F.max(F.col("wd_philippine_citizen").cast("int")).cast(
                "boolean").alias("wd_philippine_citizen"),
        )
    )
    deaths = authors.select(
        "author_key", "name", "entity_type", "ol_birth_year",
        "ol_death_year",
    ).join(wd_links, "author_key", "left")
    deaths = deaths.withColumn(
        "death_year", F.coalesce("wd_death_year", "ol_death_year")
    ).withColumn(
        "death_source",
        F.when(F.col("wd_death_year").isNotNull(), "wikidata")
        .when(F.col("ol_death_year").isNotNull(), "open_library"),
    ).withColumn(
        "birth_year", F.coalesce("wd_birth_year", "ol_birth_year")
    ).withColumn(
        "wd_philippine_citizen",
        F.coalesce("wd_philippine_citizen", F.lit(False)),
    ).withColumn(
        # a death before birth, or a death after the dump, is a data error
        "life_dates_conflict",
        F.coalesce(
            (F.col("death_year") < F.col("birth_year"))
            | (F.col("death_year") - F.col("birth_year") > 120),
            F.lit(False),
        ),
    )
    deaths = deaths.cache()

    source_summary = (
        deaths.groupBy(F.coalesce("death_source", F.lit("none")).alias(
            "death_source"))
        .count().toPandas().sort_values("count", ascending=False)
    )
    total = source_summary["count"].sum()
    source_summary["share_pct"] = (100 * source_summary["count"] /
                                   total).round(2)

    both = deaths.filter(
        F.col("wd_death_year").isNotNull()
        & F.col("ol_death_year").isNotNull()
    )
    diff = F.abs(F.col("wd_death_year") - F.col("ol_death_year"))
    agreement = both.agg(
        F.count(F.lit(1)).alias("authors_with_both"),
        F.sum(F.when(diff == 0, 1).otherwise(0)).alias("exact_match"),
        F.sum(F.when(diff <= 1, 1).otherwise(0)).alias("within_one_year"),
        F.sum(F.when(diff > 10, 1).otherwise(0)).alias("differ_over_10"),
        F.sum(F.when(
            ((F.col("wd_death_year") <= REFERENCE_YEAR - 51)
             != (F.col("ol_death_year") <= REFERENCE_YEAR - 51)), 1
        ).otherwise(0)).alias("ph_status_would_flip"),
    ).toPandas()
    n = agreement.loc[0, "authors_with_both"]
    for column in ["exact_match", "within_one_year", "differ_over_10",
                   "ph_status_would_flip"]:
        agreement[f"{column}_pct"] = (
            round(100 * agreement.loc[0, column] / n, 2) if n else np.nan
        )
    return {
        "author_deaths": deaths,
        "source_summary": source_summary.reset_index(drop=True),
        "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# PART 5: COPYRIGHT STATUS ESTIMATES
# ---------------------------------------------------------------------------
#
# The rules below describe the position on 1 January 2027.
#
# United States (17 U.S.C. sections 302-304; publication-based for older
# works). A work published in year Y before 1978 is protected for 95 years,
# ending on 31 December of year Y + 95, so it is public domain from
# 1 January of Y + 96.
#   * published 1930 or earlier ............ public domain already
#   * published 1931 ....................... enters the public domain 2027
#   * published 1932-1963 .................. PD only if the copyright was
#     never renewed; renewal records are not in Open Library, so these are
#     flagged "renewal unknown"
#   * published 1964-1977 .................. protected (renewal automatic)
#   * published 1978 or later .............. protected (life + 70 years,
#     and at least to 2047 for works created earlier but published
#     1978-2002)
#
# Philippines (Republic Act 8293, sections 213-214). Protection lasts for
# the life of the author plus 50 years, counted from 1 January of the year
# after death. For several authors the last surviving author counts.
#   * last author died 1975 or earlier ..... public domain already
#   * last author died 1976 ................ enters the public domain 2027
#   * died 1977 or later ................... protected
#   * any author without a death year ...... unknown
#
# Life + 70 (EU, UK and most of the world) is reported for comparison:
# died 1955 or earlier is PD, died 1956 enters in 2027.

US_LABELS = {
    "pd": "Public domain (published 1930 or earlier)",
    "enters": "Enters public domain 1 Jan 2027 (published 1931)",
    "renewal": "Renewal unknown (published 1932-1963)",
    "protected_pre78": "Protected (published 1964-1977)",
    "protected_post78": "Protected (published 1978 or later)",
    "unknown": "Unknown (no valid publication year)",
}
PH_LABELS = {
    "pd": "Public domain (last author died 1975 or earlier)",
    "enters": "Enters public domain 1 Jan 2027 (last author died 1976)",
    "protected": "Protected (last author died 1977 or later)",
    "unknown_death": "Unknown (an author's death year is missing)",
    "unknown_author": "Unknown (no author linked)",
}
LIFE70_LABELS = {
    "pd": "Public domain (died 1955 or earlier)",
    "enters": "Enters public domain 1 Jan 2027 (died 1956)",
    "protected": "Protected (died 1957 or later)",
    "unknown": "Unknown",
}


def us_status(first_year, reference_year=REFERENCE_YEAR):
    """US status code from a first-publication year (pure Python).

    Returns one of the US_LABELS keys. The year that enters the public
    domain on 1 January of reference_year is reference_year - 96.
    """
    if first_year is None:
        return "unknown"
    entering = reference_year - 96
    if first_year < entering:
        return "pd"
    if first_year == entering:
        return "enters"
    if first_year <= 1963:
        return "renewal"
    if first_year <= 1977:
        return "protected_pre78"
    return "protected_post78"


def life_plus_status(last_death_year, n_authors, n_with_death, term,
                     reference_year=REFERENCE_YEAR):
    """Status code for a "life + term years" rule (pure Python).

    Protection ends on 31 December of death_year + term, so the year that
    enters the public domain on 1 January of reference_year is
    reference_year - term - 1. All authors must have a death year.
    """
    if n_authors == 0:
        return "unknown_author"
    if n_with_death < n_authors or last_death_year is None:
        return "unknown_death"
    entering = reference_year - term - 1
    if last_death_year < entering:
        return "pd"
    if last_death_year == entering:
        return "enters"
    return "protected"


def us_status_column(year_col, reference_year=REFERENCE_YEAR):
    """Spark expression mirroring us_status."""
    from pyspark.sql import functions as F

    entering = reference_year - 96
    return (
        F.when(year_col.isNull(), "unknown")
        .when(year_col < entering, "pd")
        .when(year_col == entering, "enters")
        .when(year_col <= 1963, "renewal")
        .when(year_col <= 1977, "protected_pre78")
        .otherwise("protected_post78")
    )


def life_plus_status_column(death_col, n_authors_col, n_death_col, term,
                            reference_year=REFERENCE_YEAR):
    """Spark expression mirroring life_plus_status."""
    from pyspark.sql import functions as F

    entering = reference_year - term - 1
    return (
        F.when(n_authors_col == 0, "unknown_author")
        .when((n_death_col < n_authors_col) | death_col.isNull(),
              "unknown_death")
        .when(death_col < entering, "pd")
        .when(death_col == entering, "enters")
        .otherwise("protected")
    )


def estimate_copyright_status(works, editions, author_deaths,
                              reference_year=REFERENCE_YEAR):
    """One row per work with its estimated first year and rights status.

    first_year is the earlier of the work's own first_publish_date year and
    the earliest valid edition year. Using the minimum matches how Open
    Library itself reports "first published", but one mis-dated edition can
    make a work look older than it is, so two checks are added:

    * year_before_birth: first_year is earlier than the last author's
      birth year + 10, which is implausible for a real first edition;
    * gap_after_death: first_year is more than 70 years after the last
      author died, suggesting the older editions are missing from the
      catalogue (so the US status is probably understated).

    Works with year_before_birth are kept but marked "flagged" and are
    excluded from the watchlist.
    """
    from pyspark.sql import functions as F

    edition_years = (
        editions.filter(F.col("work_key").isNotNull())
        .groupBy("work_key")
        .agg(
            F.min("publish_year").alias("earliest_edition_year"),
            F.count(F.lit(1)).alias("_n"),
        )
        .drop("_n")
    )
    work_authors = works.select(
        "work_key", F.explode_outer("author_keys").alias("author_key")
    )
    author_facts = (
        work_authors.join(
            author_deaths.select(
                "author_key", "death_year", "birth_year",
                "wd_philippine_citizen", "life_dates_conflict",
            ),
            "author_key", "left",
        )
        .groupBy("work_key")
        .agg(
            F.count("author_key").alias("n_authors"),
            F.count("death_year").alias("n_authors_with_death"),
            F.max("death_year").alias("last_death_year"),
            F.max("birth_year").alias("last_birth_year"),
            F.max(F.col("wd_philippine_citizen").cast("int")).cast(
                "boolean").alias("has_philippine_author"),
            F.max(F.col("life_dates_conflict").cast("int")).cast(
                "boolean").alias("author_dates_conflict"),
        )
    )
    status = (
        works.select("work_key", "title", "author_keys", "work_year",
                     "is_fiction")
        .join(edition_years, "work_key", "left")
        .join(author_facts, "work_key", "left")
        .withColumn(
            "first_year", F.least("work_year", "earliest_edition_year")
        )
        .withColumn(
            "year_source",
            F.when(F.col("first_year").isNull(), None)
            .when(F.col("work_year") == F.col("first_year"), "work record")
            .otherwise("earliest edition"),
        )
        .withColumn(
            "year_before_birth",
            F.coalesce(F.col("first_year") < F.col("last_birth_year") + 10,
                       F.lit(False)),
        )
        .withColumn(
            "gap_after_death",
            F.coalesce(F.col("first_year") - F.col("last_death_year") > 70,
                       F.lit(False)),
        )
        .fillna({"n_authors": 0, "n_authors_with_death": 0,
                 "has_philippine_author": False,
                 "author_dates_conflict": False})
    )
    status = (
        status.withColumn("us_code", us_status_column(F.col("first_year"),
                                                      reference_year))
        .withColumn("ph_code", life_plus_status_column(
            F.col("last_death_year"), F.col("n_authors"),
            F.col("n_authors_with_death"), 50, reference_year))
        .withColumn("life70_code", life_plus_status_column(
            F.col("last_death_year"), F.col("n_authors"),
            F.col("n_authors_with_death"), 70, reference_year))
        .withColumn(
            "flagged",
            F.col("year_before_birth") | F.col("author_dates_conflict"),
        )
        .withColumn(
            "potentially_pd",
            ~F.col("flagged")
            & ((F.col("us_code") == "pd") | (F.col("ph_code") == "pd")),
        )
        .withColumn(
            "enters_2027",
            ~F.col("flagged")
            & ((F.col("us_code") == "enters")
               | (F.col("ph_code") == "enters")),
        )
    )
    return status


def summarize_status(status):
    """Count works in every US x Philippines status combination.

    Returns
    -------
    dict
        us (counts by US label), ph (counts by PH label), crosstab
        (US label rows x PH label columns) and flags (data-quality flags).
    """
    from pyspark.sql import functions as F

    combos = (
        status.groupBy("us_code", "ph_code")
        .agg(F.count(F.lit(1)).alias("works"))
        .toPandas()
    )
    combos["US status"] = combos["us_code"].map(US_LABELS)
    combos["PH status"] = combos["ph_code"].map(PH_LABELS)
    total = combos["works"].sum()

    def by(column):
        table = combos.groupby(column, as_index=False)["works"].sum()
        table["share_pct"] = (100 * table["works"] / total).round(2)
        return table.sort_values("works", ascending=False).reset_index(
            drop=True)

    crosstab = combos.pivot_table(index="US status", columns="PH status",
                                  values="works", aggfunc="sum",
                                  fill_value=0)
    flags = status.agg(
        F.count(F.lit(1)).alias("works"),
        F.sum(F.col("year_before_birth").cast("int")).alias(
            "year_before_birth"),
        F.sum(F.col("author_dates_conflict").cast("int")).alias(
            "author_dates_conflict"),
        F.sum(F.col("gap_after_death").cast("int")).alias(
            "gap_after_death"),
        F.sum(F.col("potentially_pd").cast("int")).alias("potentially_pd"),
        F.sum(F.col("enters_2027").cast("int")).alias("enters_2027"),
    ).toPandas()
    return {"us": by("US status"), "ph": by("PH status"),
            "crosstab": crosstab, "flags": flags}


def status_by_regime(status):
    """Works per status group under all three regimes, side by side.

    The life + 70 rule (EU, UK and most other countries) is computed for
    every work in estimate_copyright_status but used only for comparison;
    this table is where it is reported, next to the two regimes the
    analysis relies on.

    Returns
    -------
    DataFrame
        Rows: public domain, enters 2027, renewal unknown (US only),
        protected, unknown. Columns: one per regime (works).
    """
    from pyspark.sql import functions as F

    def group(code_col, regime):
        if regime == "us":
            return (
                F.when(F.col(code_col) == "pd", "Public domain already")
                .when(F.col(code_col) == "enters", "Enters 1 Jan 2027")
                .when(F.col(code_col) == "renewal",
                      "Renewal unknown (US only)")
                .when(F.col(code_col).startswith("protected"), "Protected")
                .otherwise("Unknown (missing data)")
            )
        return (
            F.when(F.col(code_col) == "pd", "Public domain already")
            .when(F.col(code_col) == "enters", "Enters 1 Jan 2027")
            .when(F.col(code_col) == "protected", "Protected")
            .otherwise("Unknown (missing data)")
        )

    row = status.select(
        group("us_code", "us").alias("United States (publication + 95)"),
        group("ph_code", "life").alias("Philippines (life + 50)"),
        group("life70_code", "life").alias("Life + 70 (comparison)"),
    )
    order = ["Public domain already", "Enters 1 Jan 2027",
             "Renewal unknown (US only)", "Protected",
             "Unknown (missing data)"]
    tables = []
    for column in row.columns:
        counts = row.groupBy(column).count().toPandas()
        tables.append(counts.set_index(column)["count"].rename(column))
    return (pd.concat(tables, axis=1).reindex(order).fillna(0)
            .astype(int).rename_axis("Status on 1 Jan 2027"))


def first_year_histogram(status, start=1800, end=MAX_VALID_YEAR):
    """Number of works per estimated first-publication year (pandas)."""
    from pyspark.sql import functions as F

    return (
        status.filter(F.col("first_year").between(start, end))
        .groupBy("first_year").count()
        .toPandas().sort_values("first_year").reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# PART 6: DEMAND (READING LOG, RATINGS, WANT-TO-READ TRENDS)
# ---------------------------------------------------------------------------


def compute_demand(reading_log, ratings, trend_end=DUMP_DATE):
    """Per-work demand signals.

    * want_to_read, currently_reading, already_read: shelf entries;
    * wtr_last_12m / wtr_prev_12m: Want-to-Read entries logged in the 12
      months ending at trend_end, and in the 12 months before that;
    * ratings_count, ratings_mean: valid 1-5 star ratings.

    The reading-log dump only contains entries that are still on a shelf,
    so older periods are undercounted (readers move or remove books).
    Growth figures are therefore descriptive, not a true trend estimate.
    """
    from pyspark.sql import functions as F

    end = F.lit(trend_end.isoformat()).cast("date")
    last_start = F.add_months(end, -12)
    prev_start = F.add_months(end, -24)
    is_wtr = F.col("shelf") == "Want to Read"
    shelves = reading_log.groupBy("work_key").agg(
        F.sum(is_wtr.cast("int")).alias("want_to_read"),
        F.sum((F.col("shelf") == "Currently Reading").cast("int")).alias(
            "currently_reading"),
        F.sum((F.col("shelf") == "Already Read").cast("int")).alias(
            "already_read"),
        F.sum((is_wtr & (F.col("logged_date") > last_start)
               & (F.col("logged_date") <= end)).cast("int")).alias(
            "wtr_last_12m"),
        F.sum((is_wtr & (F.col("logged_date") > prev_start)
               & (F.col("logged_date") <= last_start)).cast("int")).alias(
            "wtr_prev_12m"),
    )
    stars = ratings.filter(F.col("rating").isNotNull()).groupBy(
        "work_key").agg(
        F.count(F.lit(1)).alias("ratings_count"),
        F.avg("rating").alias("ratings_mean"),
    )
    demand = shelves.join(stars, "work_key", "full_outer").fillna(
        0, subset=["want_to_read", "currently_reading", "already_read",
                   "wtr_last_12m", "wtr_prev_12m", "ratings_count"]
    ).withColumn(
        "shelf_total",
        F.col("want_to_read") + F.col("currently_reading")
        + F.col("already_read"),
    )
    return demand


def summarize_shelves(reading_log, ratings):
    """Overall shelf and rating counts (pandas) for the report."""
    from pyspark.sql import functions as F

    shelf_counts = (
        reading_log.groupBy("shelf").agg(
            F.count(F.lit(1)).alias("entries"),
            F.countDistinct("work_key").alias("distinct_works"),
        ).toPandas().sort_values("entries", ascending=False)
    )
    rating_counts = (
        ratings.groupBy("rating").count().toPandas().sort_values("rating")
    )
    return {"shelves": shelf_counts.reset_index(drop=True),
            "ratings": rating_counts.reset_index(drop=True)}


def monthly_want_to_read(reading_log, status, start="2015-01-01",
                         end=DUMP_DATE.isoformat()):
    """Monthly Want-to-Read entries split by the work's rights group.

    Groups: "US PD (<=1930)", "Enters 2027", "Renewal unknown (1932-63)"
    and "Other". Returns a pandas table: month x group counts.
    """
    from pyspark.sql import functions as F

    group = (
        F.when(F.col("enters_2027"), "Enters 2027")
        .when(F.col("us_code") == "pd", "US PD (<=1930)")
        .when(F.col("us_code") == "renewal", "Renewal unknown (1932-63)")
        .otherwise("Other or unknown")
    )
    groups = status.select("work_key", group.alias("rights_group"))
    monthly = (
        reading_log.filter(
            (F.col("shelf") == "Want to Read")
            & F.col("logged_date").between(start, end)
        )
        .join(groups, "work_key", "left")
        .fillna({"rights_group": "Other or unknown"})
        .groupBy(F.trunc("logged_date", "month").alias("month"),
                 "rights_group")
        .count()
        .toPandas()
    )
    table = monthly.pivot(index="month", columns="rights_group",
                          values="count").fillna(0).sort_index()
    table.index = pd.to_datetime(table.index)
    return table


# ---------------------------------------------------------------------------
# PART 7: SUPPLY (EDITIONS, PUBLISHERS, SCANS, FILIPINO EDITIONS)
# ---------------------------------------------------------------------------


def compute_supply(editions, recent_from=RECENT_FROM_YEAR):
    """Per-work supply signals from the editions table.

    * n_editions, n_recent_editions (publish_year >= recent_from)
    * n_publishers, n_recent_publishers: distinct normalized publishers
    * n_digitized: editions with an Internet Archive scan (ocaid)
    * n_filipino_editions: editions catalogued in Tagalog/Filipino
    * n_languages, latest_edition_year

    Editions without a valid year are counted in n_editions but can never
    be "recent", so recent counts are lower bounds.
    """
    from pyspark.sql import functions as F

    is_recent = F.coalesce(F.col("publish_year") >= recent_from,
                           F.lit(False))
    is_filipino = F.exists(
        "language_keys", lambda k: k.isin(*FILIPINO_LANGUAGE_KEYS)
    )
    per_work = (
        editions.filter(F.col("work_key").isNotNull())
        .withColumn("is_recent", is_recent)
        .groupBy("work_key")
        .agg(
            F.count(F.lit(1)).alias("n_editions"),
            F.sum(F.col("is_recent").cast("int")).alias("n_recent_editions"),
            F.size(F.array_distinct(F.flatten(F.collect_list(
                "publisher_keys")))).alias("n_publishers"),
            F.size(F.array_distinct(F.flatten(F.collect_list(
                F.when(F.col("is_recent"), F.col("publisher_keys"))
            )))).alias("n_recent_publishers"),
            F.sum(F.col("ocaid").isNotNull().cast("int")).alias(
                "n_digitized"),
            F.sum(is_filipino.cast("int")).alias("n_filipino_editions"),
            F.size(F.array_distinct(F.flatten(F.collect_list(
                "language_keys")))).alias("n_languages"),
            F.max("publish_year").alias("latest_edition_year"),
        )
    )
    return per_work


def build_work_profile(status, demand, supply):
    """Join rights status, demand and supply into one row per work.

    Works without reading-log entries or editions receive zero counts, so
    every work in the works dump appears exactly once.
    """
    zero_cols = [
        "want_to_read", "currently_reading", "already_read", "shelf_total",
        "wtr_last_12m", "wtr_prev_12m", "ratings_count", "n_editions",
        "n_recent_editions", "n_publishers", "n_recent_publishers",
        "n_digitized", "n_filipino_editions", "n_languages",
    ]
    profile = (
        status.join(demand, "work_key", "left")
        .join(supply, "work_key", "left")
        .fillna(0, subset=zero_cols)
    )
    return profile


def compare_groups(profile):
    """Demand and supply summary for each US rights group (pandas).

    Medians are computed approximately with Spark's percentile_approx
    (relative error 1e-4); means and shares are exact.
    """
    from pyspark.sql import functions as F

    def share(cond):
        return F.round(100 * F.avg(cond.cast("int")), 2)

    table = (
        profile.groupBy("us_code").agg(
            F.count(F.lit(1)).alias("works"),
            share(F.col("want_to_read") > 0).alias("pct_with_wtr"),
            F.round(F.avg("want_to_read"), 3).alias("mean_wtr"),
            F.sum("want_to_read").alias("total_wtr"),
            F.round(F.avg("n_editions"), 2).alias("mean_editions"),
            share(F.col("n_recent_editions") > 0).alias(
                "pct_with_recent_edition"),
            share(F.col("n_digitized") > 0).alias("pct_digitized"),
            share(F.col("n_filipino_editions") > 0).alias("pct_filipino"),
            F.sum("wtr_last_12m").alias("wtr_last_12m"),
            F.sum("wtr_prev_12m").alias("wtr_prev_12m"),
            F.sum("ratings_count").alias("ratings"),
            # try_divide returns null for a group with no ratings, where
            # plain division raises an error under Spark 4's ANSI mode
            F.round(F.try_divide(
                F.sum(F.col("ratings_mean") * F.col("ratings_count")),
                F.sum("ratings_count")), 2).alias("mean_rating"),
        ).toPandas()
    )
    # growth of Want-to-Read entries, last 12 months vs the 12 before
    # (undefined, so missing, when the earlier window has no entries)
    previous = table["wtr_prev_12m"].where(table["wtr_prev_12m"] > 0)
    table["wtr_growth_pct"] = (
        100 * (table["wtr_last_12m"] / previous - 1)
    ).round(1)
    table["US status"] = table["us_code"].map(US_LABELS)
    order = list(US_LABELS)
    table["order"] = table["us_code"].map(order.index)
    return table.sort_values("order").drop(columns="order").reset_index(
        drop=True)


def demand_supply_grid(profile, demand_bins=(0, 1, 10, 100, 1000),
                       recent_bins=(0, 1, 3, 10)):
    """Works by Want-to-Read band x recent-edition band, for PD groups.

    Rows are demand bands, columns are supply bands; values count works
    that are potentially PD or enter in 2027 (not flagged).
    """
    from pyspark.sql import functions as F

    def band(column, edges):
        expr = None
        labels = []
        for i, low in enumerate(edges):
            high = edges[i + 1] - 1 if i + 1 < len(edges) else None
            label = f"{low}+" if high is None else (
                f"{low}" if high == low else f"{low}-{high}")
            labels.append(label)
            cond = F.col(column) >= low
            if high is not None:
                cond = cond & (F.col(column) <= high)
            expr = (F.when(cond, label) if expr is None
                    else expr.when(cond, label))
        return expr, labels

    demand_expr, demand_labels = band("want_to_read", demand_bins)
    supply_expr, supply_labels = band("n_recent_editions", recent_bins)
    grid = (
        profile.filter(F.col("potentially_pd") | F.col("enters_2027"))
        .groupBy(demand_expr.alias("Want to Read"),
                 supply_expr.alias("Recent editions"))
        .count().toPandas()
        .pivot(index="Want to Read", columns="Recent editions",
               values="count")
        .reindex(index=demand_labels, columns=supply_labels)
        .fillna(0).astype(int)
    )
    return grid


# ---------------------------------------------------------------------------
# PART 8: 2027 WATCHLIST AND POTENTIALLY-PUBLIC-DOMAIN LIST
# ---------------------------------------------------------------------------

DISPLAY_COLUMNS = [
    "work_key", "title", "first_year", "last_death_year", "us_code",
    "ph_code", "want_to_read", "wtr_last_12m", "ratings_count",
    "ratings_mean", "n_editions", "n_recent_editions",
    "n_recent_publishers", "n_digitized", "n_filipino_editions",
    "latest_edition_year", "gap_after_death",
]


def demand_threshold(profile, quantile=0.99):
    """Want-to-Read count at a quantile among works with any WTR entry.

    Using only works with at least one entry avoids a threshold of zero:
    most catalogue works have never been shelved by anyone.
    """
    from pyspark.sql import functions as F

    value = (
        profile.filter(F.col("want_to_read") > 0)
        .agg(F.percentile_approx("want_to_read", quantile, 10000))
        .collect()[0][0]
    )
    return int(value)


def add_author_names(selected, authors):
    """Attach a readable "authors" column (names in credit order).

    Parameters
    ----------
    selected : Spark DataFrame
        Must contain work_key and author_keys; keep it small (a top-N
        list), because names are looked up for every author key.
    authors : Spark DataFrame
        The cleaned authors table.
    """
    from pyspark.sql import functions as F

    names = (
        selected.select("work_key",
                        F.posexplode("author_keys").alias("pos",
                                                          "author_key"))
        .join(authors.select("author_key", "name"), "author_key", "left")
        .groupBy("work_key")
        .agg(F.array_sort(F.collect_list(F.struct("pos", "name"))).alias(
            "_names"))
        .select("work_key", F.array_join(
            F.transform("_names", lambda x: x["name"]), ", ").alias(
            "authors"))
    )
    return selected.join(names, "work_key", "left")


def _finish_list(selected, order, top_n, authors):
    """Shared tail of the two list builders: limit, add names, sort."""
    if top_n is None:
        return selected.select(*DISPLAY_COLUMNS)
    top = selected.orderBy(*order).limit(top_n)
    if authors is not None:
        top = add_author_names(top, authors)
    # the join for names does not keep row order, so sort again in pandas
    table = top.toPandas().sort_values(
        ["want_to_read", "wtr_last_12m", "ratings_count", "work_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    # whole years read better as integers; Int64 keeps missing values
    for column in ["first_year", "last_death_year", "latest_edition_year"]:
        table[column] = table[column].astype("Int64")
    table["ratings_mean"] = table["ratings_mean"].round(2)
    columns = DISPLAY_COLUMNS[:2] + (["authors"] if authors is not None
                                     else []) + DISPLAY_COLUMNS[2:]
    return table[columns]


def build_watchlist(profile, min_want_to_read, max_recent_editions=1,
                    top_n=None, authors=None):
    """High-demand, low-supply works entering the public domain in 2027.

    Selection (all must hold):
      1. enters_2027: US first publication 1931 OR Philippine life+50 with
         the last author dying in 1976, and no data-quality flag;
      2. want_to_read >= min_want_to_read (high demand);
      3. n_recent_editions <= max_recent_editions (low recent supply).
    Ranked by Want-to-Read, then recent Want-to-Read, then ratings.

    Returns
    -------
    Spark DataFrame with DISPLAY_COLUMNS, or a pandas table of the top_n
    rows (with author names when the authors table is passed).
    """
    from pyspark.sql import functions as F

    selected = profile.filter(
        F.col("enters_2027")
        & (F.col("want_to_read") >= min_want_to_read)
        & (F.col("n_recent_editions") <= max_recent_editions)
    )
    order = [F.desc("want_to_read"), F.desc("wtr_last_12m"),
             F.desc("ratings_count"), F.col("work_key")]
    return _finish_list(selected, order, top_n, authors)


def build_pd_list(profile, min_want_to_read=1, top_n=None, authors=None,
                  max_recent_editions=None):
    """Works estimated to be public domain already (US or PH), by demand.

    Includes only unflagged works with at least min_want_to_read entries,
    so the list is about books readers actually look for. With
    max_recent_editions set, only works with at most that many editions
    since RECENT_FROM_YEAR are kept (the low-supply view).
    """
    from pyspark.sql import functions as F

    selected = profile.filter(
        F.col("potentially_pd") & (F.col("want_to_read") >= min_want_to_read)
    )
    if max_recent_editions is not None:
        selected = selected.filter(
            F.col("n_recent_editions") <= max_recent_editions
        )
    order = [F.desc("want_to_read"), F.desc("wtr_last_12m"),
             F.desc("ratings_count"), F.col("work_key")]
    return _finish_list(selected, order, top_n, authors)


def selection_funnel(profile, min_want_to_read, max_recent_editions=1):
    """Works remaining after each filtering step of the two lists.

    Every step is applied on top of the previous one, so the table shows
    exactly where works are lost: the share of the previous step and the
    share of the whole catalogue are both reported.

    Returns
    -------
    DataFrame
        list, step, works, pct_of_previous, pct_of_catalogue.
    """
    from pyspark.sql import functions as F

    unflagged = ~F.col("flagged")
    demand = F.col("want_to_read") >= min_want_to_read
    supply = F.col("n_recent_editions") <= max_recent_editions
    entering = (F.col("us_code") == "enters") | (F.col("ph_code") == "enters")
    free = (F.col("us_code") == "pd") | (F.col("ph_code") == "pd")
    steps = {
        "2027 watchlist": [
            ("all works in the catalogue", F.lit(True)),
            ("no data-quality flag", unflagged),
            ("enters the public domain on 1 Jan 2027", entering),
            (f"high demand (>= {min_want_to_read} Want-to-Read)", demand),
            (f"low supply (<= {max_recent_editions} recent edition)",
             supply),
        ],
        "public-domain list": [
            ("all works in the catalogue", F.lit(True)),
            ("no data-quality flag", unflagged),
            ("public domain already (US or PH rule)", free),
            (f"high demand (>= {min_want_to_read} Want-to-Read)", demand),
        ],
    }
    aggs, labels = [], []
    for name, conditions in steps.items():
        cumulative = F.lit(True)
        for i, (label, condition) in enumerate(conditions):
            cumulative = cumulative & condition
            aggs.append(F.sum(cumulative.cast("int")).alias(f"c{len(aggs)}"))
            labels.append((name, label))
    counts = profile.agg(*aggs).collect()[0]
    rows, previous = [], None
    for i, (name, label) in enumerate(labels):
        works = int(counts[f"c{i}"])
        if label == "all works in the catalogue":
            total, previous = works, works
        rows.append({
            "list": name, "step": label, "works": works,
            "pct_of_previous": round(100 * works / previous, 2)
            if previous else np.nan,
            "pct_of_catalogue": round(100 * works / total, 4),
        })
        previous = works
    return pd.DataFrame(rows)


def watchlist_sensitivity(profile, quantiles=(0.90, 0.95, 0.99),
                          max_recent_options=(0, 1, 2, 5)):
    """How many works the watchlist keeps under other reasonable cutoffs.

    A stable top of the list across cutoffs shows the ranking is not an
    artefact of one arbitrary threshold. Returns a pandas grid.
    """
    from pyspark.sql import functions as F

    candidates = profile.filter(F.col("enters_2027")).select(
        "want_to_read", "n_recent_editions").cache()
    rows = []
    for q in quantiles:
        threshold = demand_threshold(profile, q)
        for max_recent in max_recent_options:
            count = candidates.filter(
                (F.col("want_to_read") >= threshold)
                & (F.col("n_recent_editions") <= max_recent)
            ).count()
            rows.append({"WTR quantile": q, "WTR threshold": threshold,
                         "max recent editions": max_recent,
                         "works": count})
    candidates.unpersist()
    return pd.DataFrame(rows).pivot_table(
        index=["WTR quantile", "WTR threshold"],
        columns="max recent editions", values="works")


def entering_2027_breakdown(profile):
    """Count 2027 entrants by route (US, PH or both) and by data flags."""
    from pyspark.sql import functions as F

    route = (
        F.when((F.col("us_code") == "enters") & (F.col("ph_code") ==
                                                 "enters"), "US and PH")
        .when(F.col("us_code") == "enters", "US only (published 1931)")
        .when(F.col("ph_code") == "enters", "PH only (author died 1976)")
    )
    return (
        profile.filter((F.col("us_code") == "enters")
                       | (F.col("ph_code") == "enters"))
        .groupBy(route.alias("route"), "flagged")
        .agg(
            F.count(F.lit(1)).alias("works"),
            F.sum((F.col("want_to_read") > 0).cast("int")).alias(
                "with_wtr"),
            F.sum("want_to_read").alias("total_wtr"),
            F.sum((F.col("n_recent_editions") == 0).cast("int")).alias(
                "no_recent_edition"),
            F.sum((F.col("n_filipino_editions") > 0).cast("int")).alias(
                "with_filipino_edition"),
        )
        .toPandas().sort_values(["flagged", "works"],
                                ascending=[True, False])
        .reset_index(drop=True)
    )


def top_authors(profile, authors, condition="enters_2027", top_n=15):
    """Authors with the most Want-to-Read entries among selected works.

    Parameters
    ----------
    profile : Spark DataFrame
        Work profile from build_work_profile.
    authors : Spark DataFrame
        Cleaned authors table, used for names.
    condition : str
        Boolean profile column that selects the works (for example
        "enters_2027" or "potentially_pd").

    A work with several authors counts once for each of them, so the
    totals across authors can exceed the total for the works.
    """
    from pyspark.sql import functions as F

    per_author = (
        profile.filter(F.col(condition))
        .select(F.explode("author_keys").alias("author_key"),
                "want_to_read", "n_recent_editions", "us_code", "ph_code")
        .groupBy("author_key")
        .agg(
            F.count(F.lit(1)).alias("works"),
            F.sum("want_to_read").alias("want_to_read"),
            F.sum((F.col("n_recent_editions") == 0).cast("int")).alias(
                "works_without_recent_edition"),
            F.sum((F.col("us_code") == "enters").cast("int")).alias(
                "via_us_1931"),
            F.sum((F.col("ph_code") == "enters").cast("int")).alias(
                "via_ph_1976"),
        )
        .orderBy(F.desc("want_to_read"), "author_key")
        .limit(top_n)
    )
    table = (
        per_author.join(
            authors.select("author_key", "name"), "author_key", "left")
        .toPandas()
        .sort_values(["want_to_read", "author_key"],
                     ascending=[False, True])
        .reset_index(drop=True)
    )
    return table[["name", "works", "want_to_read",
                  "works_without_recent_edition", "via_us_1931",
                  "via_ph_1976", "author_key"]]


def list_flag_summary(profile, min_want_to_read):
    """How often the gap_after_death warning fires on high-demand lists.

    gap_after_death means the earliest catalogued edition is more than 70
    years after the last author's death. For genuine classics it only
    shows that old editions are missing; but it is also the signature of
    a work linked to the wrong (much older) author record, which would
    make a modern book look public domain. Every listed work with this
    warning needs a manual check of its author link.
    """
    from pyspark.sql import functions as F

    demand = F.col("want_to_read") >= min_want_to_read
    lists = {
        "potentially public domain, high demand":
            F.col("potentially_pd") & demand,
        "... and at most one recent edition":
            F.col("potentially_pd") & demand
            & (F.col("n_recent_editions") <= 1),
        "enters 2027, high demand": F.col("enters_2027") & demand,
    }
    rows = []
    for name, condition in lists.items():
        counts = profile.filter(condition).agg(
            F.count(F.lit(1)).alias("works"),
            F.sum(F.col("gap_after_death").cast("int")).alias(
                "gap_after_death"),
        ).collect()[0]
        works = counts["works"]
        gaps = counts["gap_after_death"] or 0
        rows.append({
            "list": name, "works": works, "gap_after_death": gaps,
            "gap_pct": round(100 * gaps / works, 1) if works else np.nan,
        })
    return pd.DataFrame(rows)


def philippine_authors_summary(profile):
    """Works by authors with Philippine citizenship (Wikidata P27 = Q928)."""
    from pyspark.sql import functions as F

    return (
        profile.filter(F.col("has_philippine_author"))
        .groupBy("ph_code")
        .agg(F.count(F.lit(1)).alias("works"),
             F.sum("want_to_read").alias("total_wtr"),
             F.sum((F.col("n_filipino_editions") > 0).cast("int")).alias(
                 "with_filipino_edition"))
        .toPandas()
        .assign(**{"PH status": lambda t: t["ph_code"].map(PH_LABELS)})
        .sort_values("works", ascending=False).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# PART 8b: ANALYTICAL CHECKS (EXPLORATION, UNCERTAINTY, ROBUSTNESS)
# ---------------------------------------------------------------------------


def parsing_rule_examples():
    """Show the parsing rules on typical messy inputs from the dumps.

    The Spark pipeline applies these rules as column expressions; the
    pure-Python twins used here give identical answers (TEST.ipynb checks
    this), so the table documents exactly what the pipeline does with
    each kind of input.

    Returns
    -------
    DataFrame
        rule, raw input, parsed output.
    """
    rows = []
    for text in ["February 27, 1997", "c1931", "1931-1932", "19--",
                 "2030"]:
        rows.append(("publication year (first plausible year)", text,
                     parse_year(text)))
    for text in ["1963)", "c. 1850 or 1851", "384 BC"]:
        rows.append(("life year (latest year, BC negative)", text,
                     parse_life_year(text)))
    for text in ["+1976-01-12T00:00:00Z", "-0384-00-00T00:00:00Z"]:
        rows.append(("Wikidata time value", text,
                     parse_wikidata_time_year(text)))
    for text in ["Penguin Books Ltd.", "Simon & Schuster, Inc."]:
        rows.append(("publisher name", text, normalize_publisher(text)))
    for ol_year, wd_year in [(1951, 1950), (1951, None), (None, None)]:
        year, source = choose_death_year(ol_year, wd_year)
        rows.append(("death year (Open Library, Wikidata)",
                     f"{ol_year}, {wd_year}", f"{year} from {source}"))
    table = pd.DataFrame(rows, columns=["rule", "raw input", "parsed"])
    table["parsed"] = table["parsed"].map(
        lambda v: "no year" if v is None else str(v))
    return table


def demand_concentration(profile):
    """How unequally Want-to-Read entries are spread over shelved works.

    Only the frequency table (how many works have exactly k entries) is
    brought to pandas, so the computation stays small even though it
    covers every shelved work.

    Returns
    -------
    dict
        lorenz (cumulative share of works vs share of entries, works
        ordered from least to most wanted), top_shares (share of entries
        held by the top 0.1%, 1% and 10% of works) and gini (0 = equal,
        1 = all entries on one work).
    """
    from pyspark.sql import functions as F

    freq = (profile.filter(F.col("want_to_read") > 0)
            .groupBy("want_to_read").count().toPandas()
            .sort_values("want_to_read").reset_index(drop=True))
    works = freq["count"].sum()
    entries = (freq["want_to_read"] * freq["count"]).sum()
    cum_works = freq["count"].cumsum() / works
    cum_entries = (freq["want_to_read"] * freq["count"]).cumsum() / entries
    lorenz = pd.DataFrame({
        "share_of_works": np.concatenate([[0.0], cum_works.to_numpy()]),
        "share_of_entries": np.concatenate([[0.0],
                                            cum_entries.to_numpy()]),
    })
    # Gini = 1 - 2 x area under the Lorenz curve (trapezoid rule)
    x, y = lorenz["share_of_works"], lorenz["share_of_entries"]
    area = float(np.sum((x.diff().fillna(0)) * (y + y.shift(1).fillna(0))
                        / 2))
    gini = 1 - 2 * area

    def top_share(fraction):
        # walk down from the most-wanted works until `fraction` of works
        # are covered, taking a partial group at the boundary
        remaining = fraction * works
        held = 0.0
        for k, n in zip(freq["want_to_read"][::-1], freq["count"][::-1]):
            take = min(n, remaining)
            held += take * k
            remaining -= take
            if remaining <= 0:
                break
        return held / entries

    top = pd.DataFrame({
        "top share of shelved works": ["0.1%", "1%", "10%"],
        "share of all Want-to-Read entries (%)": [
            round(100 * top_share(f), 1) for f in (0.001, 0.01, 0.1)],
    })
    return {"lorenz": lorenz, "top_shares": top, "gini": round(gini, 3),
            "shelved_works": int(works), "entries": int(entries)}


def demand_by_decade(profile, start=1800, end=2000):
    """Demand and supply of works grouped by decade of first publication.

    This is the bivariate view behind the whole project: if old works
    were simply unwanted, a public-domain opportunity would not exist.
    The default stops at the 2000s because a work first published from
    RECENT_FROM_YEAR on counts as having a recent edition by definition,
    which would
    make the supply column meaningless for the 2010s and 2020s.

    Returns
    -------
    DataFrame
        decade, works, pct_shelved (any Want-to-Read entry),
        wtr_per_shelved_work, mean_rating (weighted by ratings),
        pct_recent_edition.
    """
    from pyspark.sql import functions as F

    decade = (F.floor(F.col("first_year") / 10) * 10).cast("int")
    table = (
        profile.filter(F.col("first_year").between(start, end + 9))
        .groupBy(decade.alias("decade"))
        .agg(
            F.count(F.lit(1)).alias("works"),
            F.round(100 * F.avg((F.col("want_to_read") > 0).cast("int")),
                    2).alias("pct_shelved"),
            # try_divide gives null, not an error, for a decade with no
            # shelved or rated works
            F.round(F.try_divide(
                F.sum("want_to_read"),
                F.sum((F.col("want_to_read") > 0).cast("int"))),
                2).alias("wtr_per_shelved_work"),
            F.round(F.try_divide(
                F.sum(F.col("ratings_mean") * F.col("ratings_count")),
                F.sum("ratings_count")), 2).alias("mean_rating"),
            F.round(100 * F.avg((F.col("n_recent_editions") > 0)
                                .cast("int")), 2).alias(
                "pct_recent_edition"),
        )
        .toPandas().sort_values("decade").reset_index(drop=True)
    )
    return table


def demand_supply_correlation(profile, max_rows=200_000):
    """Spearman correlation between demand and supply in each US group.

    If publishers follow readers, works with more Want-to-Read entries
    should have more recent editions. Spearman's rho is used because both
    counts are heavily skewed. Each group is limited to a random sample of
    at most `max_rows` shelved works (seed fixed), which keeps the pandas
    step small; rho on 200,000 works is precise to about +/-0.005.

    Returns
    -------
    DataFrame
        US status, shelved works, sample size, rho with recent editions,
        rho with all editions, rho with digitized copies.
    """
    from pyspark.sql import functions as F

    shelved = profile.filter(F.col("want_to_read") > 0).select(
        "us_code", "want_to_read", "n_recent_editions", "n_editions",
        "n_digitized")
    sizes = {r["us_code"]: r["count"]
             for r in shelved.groupBy("us_code").count().collect()}
    rows = []
    for code in US_LABELS:
        n = sizes.get(code, 0)
        if n < 30:
            continue
        fraction = min(1.0, max_rows / n)
        sample = shelved.filter(F.col("us_code") == code).sample(
            False, fraction, SEED).toPandas()
        corr = sample.drop(columns="us_code").corr(method="spearman")
        rows.append({
            "US status": US_LABELS[code],
            "shelved works": n,
            "sample": len(sample),
            "rho: recent editions": round(
                corr.loc["want_to_read", "n_recent_editions"], 3),
            "rho: all editions": round(
                corr.loc["want_to_read", "n_editions"], 3),
            "rho: digitized copies": round(
                corr.loc["want_to_read", "n_digitized"], 3),
        })
    return pd.DataFrame(rows)


def demand_persistence(reading_log, cutoff=date(DUMP_DATE.year - 1,
                                                DUMP_DATE.month,
                                                DUMP_DATE.day),
                       quantile=0.99, months=12):
    """Hold-out check: do works wanted before a cutoff stay wanted after?

    Want-to-Read counts up to `cutoff` are treated as the "fitting"
    period, and the entries logged in the following `months` months as
    the hold-out period, which plays no part in defining the groups. If
    cumulative Want-to-Read entries were mostly noise, the works in the
    top `quantile` before the cutoff would not stand out afterwards.

    Returns
    -------
    dict
        table (one row per pre-cutoff group: works, share receiving any
        new entry, mean new entries, share of all new entries), threshold
        (the pre-cutoff count at `quantile`) and rho (Spearman between
        pre- and post-cutoff counts on a 200,000-work sample of works
        shelved before the cutoff).
    """
    from pyspark.sql import functions as F

    end = F.add_months(F.lit(cutoff.isoformat()).cast("date"), months)
    cut = F.lit(cutoff.isoformat()).cast("date")
    wtr = reading_log.filter(F.col("shelf") == "Want to Read")
    per_work = wtr.groupBy("work_key").agg(
        F.sum((F.col("logged_date") <= cut).cast("int")).alias("before"),
        F.sum(((F.col("logged_date") > cut)
               & (F.col("logged_date") <= end)).cast("int")).alias("after"),
    ).cache()
    threshold = int(per_work.filter(F.col("before") > 0).agg(
        F.percentile_approx("before", quantile, 10000)).collect()[0][0])
    group = (F.when(F.col("before") >= threshold,
                    f"top {100 - 100 * quantile:.0f}% before cutoff")
             .when(F.col("before") > 0, "other works shelved before cutoff")
             .otherwise("first shelved after cutoff"))
    table = per_work.groupBy(group.alias("group")).agg(
        F.count(F.lit(1)).alias("works"),
        F.round(100 * F.avg((F.col("after") > 0).cast("int")), 2).alias(
            "pct_with_new_entry"),
        F.round(F.avg("after"), 3).alias("mean_new_entries"),
        F.sum("after").alias("new_entries"),
    ).toPandas()
    total_new = table["new_entries"].sum()
    table["pct_of_new_entries"] = (100 * table["new_entries"]
                                   / total_new).round(1)
    order = {f"top {100 - 100 * quantile:.0f}% before cutoff": 0,
             "other works shelved before cutoff": 1,
             "first shelved after cutoff": 2}
    table = table.sort_values("group", key=lambda c: c.map(order))
    shelved = per_work.filter(F.col("before") > 0)
    n = shelved.count()
    sample = shelved.sample(False, min(1.0, 200_000 / max(n, 1)),
                            SEED).select("before", "after").toPandas()
    rho = (round(sample.corr(method="spearman").iloc[0, 1], 3)
           if len(sample) > 2 else np.nan)
    per_work.unpersist()
    return {"table": table.reset_index(drop=True), "threshold": threshold,
            "rho": rho, "cutoff": cutoff.isoformat()}


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion, in percent.

    Unlike the plain normal approximation it never leaves [0, 100] and
    stays sensible for proportions close to 0, which matters here because
    shares such as "has a Filipino edition" are around 0.01%.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def group_intervals(profile, ratings):
    """Uncertainty for the group comparisons of Sections 7.2 and 7.3.

    Shares get 95% Wilson intervals. Mean ratings get a 95% normal
    interval from the individual ratings (mean +/- 1.96 sd / sqrt(n)),
    computed by joining every rating to its work's US group.

    Returns
    -------
    DataFrame
        One row per US group with each estimate and its interval.
    """
    from pyspark.sql import functions as F

    counts = profile.groupBy("us_code").agg(
        F.count(F.lit(1)).alias("works"),
        F.sum((F.col("want_to_read") > 0).cast("int")).alias("shelved"),
        F.sum((F.col("n_recent_editions") > 0).cast("int")).alias(
            "recent"),
        F.sum((F.col("n_filipino_editions") > 0).cast("int")).alias(
            "filipino"),
    ).toPandas().set_index("us_code")
    stars = (ratings.filter(F.col("rating").isNotNull())
             .join(profile.select("work_key", "us_code"), "work_key")
             .groupBy("us_code").agg(
                 F.count(F.lit(1)).alias("ratings"),
                 F.avg("rating").alias("mean"),
                 F.stddev("rating").alias("sd"))
             .toPandas().set_index("us_code"))
    rows = []
    for code in US_LABELS:
        if code not in counts.index:
            continue
        c = counts.loc[code]
        row = {"US status": US_LABELS[code], "works": int(c["works"])}
        for column, label in [("shelved", "% shelved"),
                              ("recent", "% with recent edition"),
                              ("filipino", "% with Filipino edition")]:
            share = 100 * c[column] / c["works"]
            low, high = wilson_interval(c[column], c["works"])
            decimals = 3 if column == "filipino" else 2
            row[label] = (f"{share:.{decimals}f} "
                          f"[{low:.{decimals}f}, {high:.{decimals}f}]")
        if code in stars.index:
            r = stars.loc[code]
            half = 1.96 * r["sd"] / np.sqrt(r["ratings"])
            row["mean stars"] = (f"{r['mean']:.3f} "
                                 f"[{r['mean'] - half:.3f}, "
                                 f"{r['mean'] + half:.3f}]")
            row["ratings"] = int(r["ratings"])
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_to_choices(profile, author_deaths, min_want_to_read,
                           max_recent_editions=1):
    """Recount the headline groups under alternative data-handling choices.

    Two choices drive the rights estimates, so each is varied while
    everything else stays fixed:

    * first-publication year: the minimum of the work record and the
      earliest edition (used in the report), the work record whenever it
      exists, or the earliest edition only;
    * death year: Wikidata preferred (used in the report), Open Library
      only, or the later of the two (most cautious).

    For each variant the table reports works free and entering under the
    affected rule and the size of the 2027 watchlist. Flags are kept as
    computed for the main analysis, so the comparison isolates the choice.
    """
    from pyspark.sql import functions as F

    base = profile.select(
        "work_key", "author_keys", "work_year", "earliest_edition_year",
        "first_year", "us_code", "ph_code", "flagged", "want_to_read",
        "n_recent_editions").cache()
    wanted = ((F.col("want_to_read") >= min_want_to_read)
              & (F.col("n_recent_editions") <= max_recent_editions)
              & ~F.col("flagged"))

    def summarize(frame, us_col, ph_col, choice, variant):
        enters = (F.col(us_col) == "enters") | (F.col(ph_col) == "enters")
        r = frame.agg(
            F.sum((F.col(us_col) == "pd").cast("int")).alias("us_pd"),
            F.sum((F.col(us_col) == "enters").cast("int")).alias(
                "us_enters"),
            F.sum((F.col(ph_col) == "pd").cast("int")).alias("ph_pd"),
            F.sum((F.col(ph_col) == "enters").cast("int")).alias(
                "ph_enters"),
            F.sum((enters & wanted).cast("int")).alias("watchlist"),
        ).collect()[0]
        return {"choice": choice, "variant": variant, **r.asDict()}

    rows = []
    year_variants = {
        "min(work record, earliest edition) [used]": F.col("first_year"),
        "work record if present, else earliest edition":
            F.coalesce("work_year", "earliest_edition_year"),
        "earliest edition only": F.col("earliest_edition_year"),
    }
    for variant, expr in year_variants.items():
        frame = base.withColumn("us_alt", us_status_column(expr))
        rows.append(summarize(frame, "us_alt", "ph_code",
                              "first-publication year", variant))

    deaths = author_deaths.select("author_key", "wd_death_year",
                                  "ol_death_year")
    per_work = (base.select("work_key",
                            F.explode("author_keys").alias("author_key"))
                .join(deaths, "author_key", "left")
                .withColumn("wd_first", F.coalesce("wd_death_year",
                                                   "ol_death_year"))
                .withColumn("latest", F.greatest("wd_death_year",
                                                 "ol_death_year"))
                .groupBy("work_key").agg(
                    F.count(F.lit(1)).alias("n"),
                    F.count("wd_first").alias("k_wd"),
                    F.max("wd_first").alias("d_wd"),
                    F.count("ol_death_year").alias("k_ol"),
                    F.max("ol_death_year").alias("d_ol"),
                    F.count("latest").alias("k_late"),
                    F.max("latest").alias("d_late")))
    joined = base.join(per_work, "work_key", "left").fillna(
        0, subset=["n", "k_wd", "k_ol", "k_late"])
    death_variants = {
        "Wikidata preferred [used]": ("d_wd", "k_wd"),
        "Open Library only": ("d_ol", "k_ol"),
        "later of the two (cautious)": ("d_late", "k_late"),
    }
    for variant, (d, k) in death_variants.items():
        frame = joined.withColumn("ph_alt", life_plus_status_column(
            F.col(d), F.col("n"), F.col(k), 50))
        rows.append(summarize(frame, "us_code", "ph_alt", "death year",
                              variant))
    base.unpersist()
    return pd.DataFrame(rows)


def fiction_split(profile, min_want_to_read):
    """Compare fiction and non-fiction within the opportunity groups.

    A work counts as fiction when any of its subjects contains the word
    "fiction" (flag set during parsing). Translators and publishers
    budget differently for novels and reference works, so the split tells
    them what kind of books the lists actually contain.
    """
    from pyspark.sql import functions as F

    groups = {
        "all shelved works": F.col("want_to_read") > 0,
        "potentially PD, high demand":
            F.col("potentially_pd")
            & (F.col("want_to_read") >= min_want_to_read),
        "enters 2027, any demand":
            F.col("enters_2027") & (F.col("want_to_read") > 0),
        "enters 2027, high demand":
            F.col("enters_2027")
            & (F.col("want_to_read") >= min_want_to_read),
    }
    rows = []
    for name, condition in groups.items():
        r = profile.filter(condition).agg(
            F.count(F.lit(1)).alias("works"),
            F.round(100 * F.avg(F.col("is_fiction").cast("int")), 1).alias(
                "pct_fiction"),
            F.round(F.avg(F.when(F.col("is_fiction"), F.col("want_to_read"))),
                    1).alias("mean_wtr_fiction"),
            F.round(F.avg(F.when(~F.col("is_fiction"),
                                 F.col("want_to_read"))), 1).alias(
                "mean_wtr_nonfiction"),
            F.round(100 * F.avg(F.when(
                F.col("is_fiction"),
                (F.col("n_recent_editions") > 0).cast("int"))), 1).alias(
                "pct_recent_fiction"),
            F.round(100 * F.avg(F.when(
                ~F.col("is_fiction"),
                (F.col("n_recent_editions") > 0).cast("int"))), 1).alias(
                "pct_recent_nonfiction"),
        ).collect()[0]
        rows.append({"group": name, **r.asDict()})
    return pd.DataFrame(rows)


def translation_gaps(profile, min_want_to_read, min_languages=5,
                     top_n=None, authors=None):
    """Wanted free works already translated widely, but not into Filipino.

    A work with editions in many languages has shown that it travels
    across markets, so the absence of a Filipino edition is a concrete
    gap for a Philippine translator. Selection: potentially public domain
    or entering in 2027, at least `min_want_to_read` Want-to-Read
    entries, editions in at least `min_languages` languages, and no
    Tagalog or Filipino edition. Ranked by Want-to-Read entries.

    Returns
    -------
    dict
        count (works meeting every condition) and table (pandas top_n
        rows with author names, or None when top_n is None).
    """
    from pyspark.sql import functions as F

    selected = profile.filter(
        (F.col("potentially_pd") | F.col("enters_2027"))
        & (F.col("want_to_read") >= min_want_to_read)
        & (F.col("n_languages") >= min_languages)
        & (F.col("n_filipino_editions") == 0)
    )
    count = selected.count()
    table = None
    if top_n is not None:
        top = selected.orderBy(F.desc("want_to_read"), "work_key").limit(
            top_n)
        if authors is not None:
            top = add_author_names(top, authors)
        table = top.select(
            "title", *(["authors"] if authors is not None else []),
            "first_year", "last_death_year", "us_code", "ph_code",
            "want_to_read", "n_languages", "n_editions",
            "n_recent_editions").toPandas().sort_values(
            "want_to_read", ascending=False).reset_index(drop=True)
        for column in ["first_year", "last_death_year"]:
            table[column] = table[column].astype("Int64")
    return {"count": count, "table": table}


# ---------------------------------------------------------------------------
# PART 9: VALIDATION (INDEPENDENT RECOUNTS)
# ---------------------------------------------------------------------------


def validate_analysis(tables, status, profile, watchlist,
                      min_want_to_read, max_recent_editions=1):
    """Recount key numbers a second, independent way.

    Each check returns passed True/False and the two numbers compared.
    The recounts use different code paths (for example, raw edition rows
    instead of the per-work aggregate) so a bug in one path shows up as a
    mismatch rather than being repeated.
    """
    from pyspark.sql import functions as F

    checks = []

    def add(name, expected, actual):
        checks.append({"check": name, "expected": expected,
                       "actual": actual, "passed": expected == actual})

    n_works = tables["works"].count()
    add("profile has one row per work", n_works, profile.count())
    add("profile work keys are unique", n_works,
        profile.select("work_key").distinct().count())

    us_total = status.groupBy("us_code").count().agg(F.sum("count")) \
        .collect()[0][0]
    add("US status groups sum to all works", n_works, int(us_total))

    linked = tables["editions"].join(
        tables["works"].select("work_key"), "work_key", "left_semi"
    ).count()
    add("edition rows linked to known works = sum of n_editions", linked,
        int(profile.agg(F.sum("n_editions")).collect()[0][0]))

    wtr_linked = tables["reading_log"].filter(
        F.col("shelf") == "Want to Read"
    ).join(tables["works"].select("work_key"), "work_key",
           "left_semi").count()
    add("Want-to-Read rows on known works = sum of want_to_read",
        wtr_linked, int(profile.agg(F.sum("want_to_read")).collect()[0][0]))

    scans = tables["editions"].filter(F.col("ocaid").isNotNull()).join(
        tables["works"].select("work_key"), "work_key", "left_semi"
    ).count()
    add("digitized editions on known works = sum of n_digitized", scans,
        int(profile.agg(F.sum("n_digitized")).collect()[0][0]))

    recent = tables["editions"].filter(
        F.col("publish_year") >= RECENT_FROM_YEAR
    ).join(tables["works"].select("work_key"), "work_key",
           "left_semi").count()
    add("recent editions on known works = sum of n_recent_editions",
        recent, int(profile.agg(F.sum("n_recent_editions")).collect()[0][0]))

    wl = watchlist if isinstance(watchlist, pd.DataFrame) else \
        watchlist.toPandas()
    bad = wl[
        (wl["want_to_read"] < min_want_to_read)
        | (wl["n_recent_editions"] > max_recent_editions)
        | ~((wl["us_code"] == "enters") | (wl["ph_code"] == "enters"))
    ]
    add("every watchlist row meets all three criteria", 0, len(bad))

    # recompute the US rule in Python on a sample and compare with Spark
    sample = status.select("first_year", "us_code").sample(
        False, 0.001, SEED).toPandas()
    python_codes = sample["first_year"].map(
        lambda y: us_status(None if pd.isna(y) else int(y)))
    add("Spark US rule = Python US rule (0.1% sample)", len(sample),
        int((python_codes == sample["us_code"]).sum()))

    sample = status.select("last_death_year", "n_authors",
                           "n_authors_with_death", "ph_code").sample(
        False, 0.001, SEED).toPandas()
    python_codes = [
        life_plus_status(None if pd.isna(d) else int(d), int(n), int(k), 50)
        for d, n, k in zip(sample["last_death_year"], sample["n_authors"],
                           sample["n_authors_with_death"])
    ]
    add("Spark PH rule = Python PH rule (0.1% sample)", len(sample),
        int((pd.Series(python_codes) == sample["ph_code"].values).sum()))
    return pd.DataFrame(checks)


# Hand-checked facts for the 15 top-ranked rows of the 2027 watchlist (W)
# and of the potentially-public-domain list (P), as they appear in the
# report. Each value was checked by hand against publisher, library and
# encyclopaedia records; None means "not checked" (the listing does not
# depend on it, or no reliable source was found). These rows were chosen
# because they are the ones a reader would act on first, so they are a
# check of the lists' top, not a random sample of the catalogue.
AUDIT_SAMPLE = [
    # list, work_key, checked first year, checked last-author death, note
    ("W", "/works/OL267968W", 1931, None,
     "1931 text only; later revisions by co-authors remain protected"),
    ("W", "/works/OL2286161W", 1963, 1976, "Yale UP 1963; Albers d. 1976"),
    ("W", "/works/OL2269514W", 1939, 1976, "Trumbo d. 1976"),
    ("W", "/works/OL210644W", 1931, None,
     "London (Nisbet) 1931; demand may be confused with a 2018 title"),
    ("W", "/works/OL6563302W", 1931, 1962,
     "McGraw-Hill 1st ed. 1931; author death missing in the data"),
    ("W", "/works/OL7118113W", 1974, 1976, "Crowell 1974; Johnson d. 1976"),
    ("W", "/works/OL39394W", 1931, 2002,
     "Ghostwritten by M. Wirt Benson (d. 2002), not the linked author"),
    ("W", "/works/OL82931W", 1932, 1962,
     "Published October 1932, not 1931"),
    ("W", "/works/OL1127268W", 1931, 1937, "Adler d. 1937"),
    ("W", "/works/OL4785050W", 1940, 1976, "Sperry d. 1976"),
    ("W", "/works/OL2139804W", 1916, 1913,
     "First published 1916 (posthumous), so already free in the US"),
    ("W", "/works/OL326412W", 1958, 1976, "Heisenberg d. 1976"),
    ("W", "/works/OL707919W", 1931, 1936, "M. R. James d. 1936"),
    ("W", "/works/OL4775270W", 1939, 1976,
     "Columbia UP 1939; Boyer d. 1976"),
    ("W", "/works/OL7002113W", 1931, None, "Cape 1931"),
    ("P", "/works/OL527464W", 1937, 1970, "Hill d. 1970"),
    ("P", "/works/OL1063267W", 1936, 1955, "Carnegie d. 1955"),
    ("P", "/works/OL66554W", 1813, 1817, "Austen d. 1817"),
    ("P", "/works/OL244537W", -500, None,
     "Ancient text (c. 5th century BC); catalogue year 1900 is a "
     "placeholder"),
    ("P", "/works/OL3923952W", 1948, 1948, "Dazai d. 1948"),
    ("P", "/works/OL27482W", 1937, 1973, "Tolkien d. 1973"),
    ("P", "/works/OL10263W", 1943, 1944, "Saint-Exupery d. 1944"),
    ("P", "/works/OL1168007W", 1945, 1950, "Orwell d. 1950"),
    ("P", "/works/OL498556W", 1915, 1924, "Kafka d. 1924"),
    ("P", "/works/OL8165007W", 1926, 1957, "Clason d. 1957"),
    ("P", "/works/OL21177W", 1847, 1848,
     "Published 1847 (catalogue says 1846)"),
    ("P", "/works/OL166894W", 1866, 1881, "Dostoevsky d. 1881"),
    ("P", "/works/OL1230613W", 1942, 1960, "Camus d. 1960"),
    ("P", "/works/OL262421W", 1892, 1930, "Conan Doyle d. 1930"),
    ("P", "/works/OL64365W", 1932, 1963, "Huxley d. 1963"),
]


def compare_codes(estimated, checked):
    """Classify one estimated status code against the hand-checked one.

    Returns "not checked" when no checked value exists, "missing in data"
    when the estimate is an unknown_* code (the pipeline declined to
    guess), "correct" when the codes match and "wrong" otherwise.
    """
    if checked is None:
        return "not checked"
    if str(estimated).startswith("unknown"):
        return "missing in data"
    return "correct" if estimated == checked else "wrong"


def audit_estimates(profile, sample=AUDIT_SAMPLE):
    """Compare the pipeline's estimates with hand-checked facts.

    For every audited work the US and Philippine status codes are
    recomputed from the checked first year and death year with the same
    rule functions (us_status, life_plus_status), so a difference can
    only come from the catalogue data, not from the rules. The checked
    death year is treated as that of the last surviving author.

    Returns
    -------
    DataFrame
        One row per audited work with estimated and checked values, the
        outcome per regime and an overall verdict ("wrong" if either
        regime is wrong, otherwise "correct" if at least one regime could
        be checked and matched).
    """
    from pyspark.sql import functions as F

    keys = [key for _, key, _, _, _ in sample]
    estimates = {
        r["work_key"]: r.asDict() for r in profile.filter(
            F.col("work_key").isin(keys)).select(
            "work_key", "title", "first_year", "last_death_year",
            "us_code", "ph_code").collect()
    }
    rows = []
    for listed, key, year, death, note in sample:
        est = estimates.get(key)
        if est is None:
            continue
        checked_us = us_status(year) if year is not None else None
        checked_ph = (life_plus_status(death, 1, 1, 50)
                      if death is not None else None)
        us_outcome = compare_codes(est["us_code"], checked_us)
        ph_outcome = compare_codes(est["ph_code"], checked_ph)
        outcomes = {us_outcome, ph_outcome}
        if "wrong" in outcomes:
            verdict = "wrong"
        elif "correct" in outcomes:
            verdict = "correct"
        else:
            verdict = "not verifiable"
        rows.append({
            "list": {"W": "2027 watchlist", "P": "public domain"}[listed],
            "title": est["title"],
            "est_year": est["first_year"], "checked_year": year,
            "est_death": est["last_death_year"], "checked_death": death,
            "est_us": est["us_code"], "checked_us": checked_us,
            "us_outcome": us_outcome,
            "est_ph": est["ph_code"], "checked_ph": checked_ph,
            "ph_outcome": ph_outcome,
            "verdict": verdict, "note": note,
        })
    table = pd.DataFrame(rows)
    for column in ["est_year", "checked_year", "est_death",
                   "checked_death"]:
        table[column] = table[column].astype("Int64")
    return table


def audit_summary(audit):
    """Count audit outcomes per regime and list (pandas)."""
    parts = []
    for regime in ["us", "ph"]:
        counts = (audit.groupby(["list", f"{regime}_outcome"]).size()
                  .unstack(fill_value=0))
        counts.insert(0, "regime", {"us": "United States",
                                    "ph": "Philippines"}[regime])
        parts.append(counts)
    table = pd.concat(parts).fillna(0)
    columns = ["regime"] + [c for c in ["correct", "wrong",
                                        "missing in data", "not checked"]
                            if c in table.columns]
    for column in columns[1:]:
        table[column] = table[column].astype(int)
    return table[columns].reset_index()


# ---------------------------------------------------------------------------
# PART 10: CHARTS
# ---------------------------------------------------------------------------

# The report's colour system. Four hues, each with one fixed meaning that
# holds in every chart, in the pipeline diagram and in the notebook's own
# headings and tables:
#   INK    data, the pipeline's processing stages, US public domain
#   AMBER  the 2027 opportunity (works entering, the watchlist)
#   GREEN  reader demand
#   PLUM   uncertainty (the 1932-1963 renewal-unknown group, rights rules)
# Supply indicators use three shades of INK (INK_DARK, INK, INK_LIGHT).
# The four were checked with a colour-vision-deficiency validator on a
# light surface: every adjacent pair stays distinguishable for protan,
# deutan and tritan readers, and each passes 3:1 contrast. Grey marks
# "everything else" and the validation stage.
INK = "#1A69B0"
INK_DARK = "#0E3B66"
INK_LIGHT = "#86AED8"
AMBER = "#D9731C"
GREEN = "#178C68"
PLUM = "#8A4FB0"
GREY = "#8C939B"
TEXT = "#1E2A38"
MUTED = "#5B6675"
PAPER = "#FCFBF8"
RULE = "#E4E0D6"

PALETTE = {
    "US PD (<=1930)": INK,
    "Enters 2027": AMBER,
    "Renewal unknown (1932-63)": PLUM,
    "Other or unknown": GREY,
}


def use_report_style():
    """Apply the report's matplotlib style (fonts, colours, light grid).

    Call once at the top of the notebook; every chart function in this
    module then inherits the same warm paper background, dark-slate text
    and recessive axes, so figures look like one family.
    """
    from cycler import cycler

    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": RULE,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.prop_cycle": cycler(color=[INK, AMBER, GREEN, PLUM, GREY]),
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "text.color": TEXT,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })


def notebook_css():
    """CSS that gives the notebook's headings, captions and tables the
    same colour system as the charts. Display it with
    IPython.display.HTML; it applies in Jupyter and in the HTML export.
    """
    return f"""<style>
.jp-RenderedHTMLCommon h1, .rendered_html h1 {{
  color: {TEXT}; border-bottom: 3px solid {AMBER}; padding-bottom: 6px; }}
.jp-RenderedHTMLCommon h2, .rendered_html h2 {{
  color: {INK}; border-bottom: 1px solid {RULE}; padding-bottom: 4px;
  margin-top: 1.6em; }}
.jp-RenderedHTMLCommon h3, .rendered_html h3 {{ color: {TEXT}; }}
.jp-RenderedHTMLCommon strong, .rendered_html strong {{ color: {TEXT}; }}
.report-caption {{ color: {TEXT}; font-weight: 600; margin: 10px 0 4px 0;
  max-width: 1100px; }}
.report-caption span {{ color: {INK}; }}
.report-table table {{ border-collapse: collapse; font-size: 12px; }}
.report-table thead th {{ background: {INK}; color: #fff; padding: 5px 9px;
  text-align: left; font-weight: 600; }}
.report-table tbody td {{ padding: 4px 9px;
  border-bottom: 1px solid {RULE}; }}
.report-table tbody tr:nth-child(even) {{ background: #F5F2EA; }}
</style>"""


def _style(ax):
    """Shared light styling: no top/right spines, faint horizontal grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def plot_first_year_histogram(hist):
    """Works per first-publication year with the US cut-offs marked."""
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = [
        PALETTE["US PD (<=1930)"] if y <= 1930 else
        PALETTE["Enters 2027"] if y == 1931 else
        PALETTE["Renewal unknown (1932-63)"] if y <= 1963 else
        PALETTE["Other or unknown"]
        for y in hist["first_year"]
    ]
    ax.bar(hist["first_year"], hist["count"], width=1.0, color=colors)
    ax.set_yscale("log")
    for year, text in [(1930.5, "1931 enters 2027"), (1963.5, "1963"),
                       (1977.5, "1977")]:
        ax.axvline(year, color=MUTED, lw=0.8, ls="--")
        ax.text(year + 1, ax.get_ylim()[1] * 0.5, text, fontsize=8)
    ax.set_title("Works by estimated first-publication year (log scale)")
    ax.set_xlabel("First-publication year")
    ax.set_ylabel("Works")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_status_bars(us_table, ph_table):
    """Two horizontal bar charts: works by US status and by PH status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, table, column, title in [
        (axes[0], us_table, "US status", "United States (publication rule)"),
        (axes[1], ph_table, "PH status", "Philippines (life + 50 years)"),
    ]:
        data = table.sort_values("works")
        ax.barh(data[column], data["works"] / 1e6, color=INK)
        pairs = zip(data["works"], data["share_pct"])
        for y, (works, pct) in enumerate(pairs):
            ax.text(works / 1e6, y, f"  {pct:.1f}%", va="center",
                    fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Works (millions)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Estimated rights status on 1 January 2027")
    fig.tight_layout()
    return fig


def plot_monthly_wtr(monthly):
    """Monthly Want-to-Read entries by rights group, one panel each.

    Separate panels with their own y-axis keep small groups readable next
    to the much larger "Other" group.
    """
    groups = [g for g in PALETTE if g in monthly.columns]
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 2.2 * len(groups)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, groups):
        ax.plot(monthly.index, monthly[group], color=PALETTE[group], lw=1.4)
        ax.set_title(group, loc="left", fontsize=10)
        ax.set_ylabel("Entries")
        _style(ax)
    axes[-1].set_xlabel("Month logged")
    fig.suptitle("Want-to-Read entries per month, by rights group "
                 "(entries still on a shelf in Oct 2026)")
    fig.tight_layout()
    return fig


def plot_supply_comparison(groups):
    """Share of works with a recent edition, a scan, or a Filipino edition,
    for each US rights group."""
    data = groups.set_index("US status")[
        ["pct_with_recent_edition", "pct_digitized", "pct_filipino"]
    ].rename(columns={
        "pct_with_recent_edition":
            f"Has an edition from {RECENT_FROM_YEAR} on",
        "pct_digitized": "Has a digitized copy",
        "pct_filipino": "Has a Filipino-language edition",
    })
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # supply is one family (shades of INK), matching the Supply stage
    # of the pipeline diagram; green stays reserved for demand
    data.plot.barh(ax=ax, color=[INK_DARK, INK, INK_LIGHT],
                   width=0.75)
    ax.invert_yaxis()
    ax.set_xlabel("Share of works in the group (%)")
    ax.set_ylabel("")
    ax.set_title("Supply indicators by US rights group")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_watchlist(watchlist, top_n=20):
    """Demand (Want-to-Read) vs recent supply for the top watchlist works."""
    data = watchlist.head(top_n).iloc[::-1]
    labels = [
        (t if isinstance(t, str) and len(t) <= 45 else
         (str(t)[:42] + "...")) + f" ({int(y) if pd.notna(y) else '?'})"
        for t, y in zip(data["title"], data["first_year"])
    ]
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(data) + 1.5))
    ax.barh(labels, data["want_to_read"], color=PALETTE["Enters 2027"])
    for y, (wtr, recent) in enumerate(zip(data["want_to_read"],
                                          data["n_recent_editions"])):
        ax.text(wtr, y, f"  {int(wtr):,}  |  recent editions: "
                f"{int(recent)}", va="center", fontsize=8)
    ax.set_xlabel("Want-to-Read entries")
    ax.set_title("2027 watchlist: most-wanted works with few recent "
                 "editions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, data["want_to_read"].max() * 1.45)
    fig.tight_layout()
    return fig


def plot_pipeline_diagram():
    """Draw the analysis pipeline as a left-to-right flow diagram.

    Each card is one stage: a coloured title band (colours follow the
    report's colour system), the main operations, and the helper function
    that implements it, so the figure doubles as a map from the
    methodology to the code. Lanes underneath group the stages into the
    three phases described in Section 4. Returns a matplotlib Figure.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.75, 6.4)
    ax.axis("off")

    def card(x, y, w, h, number, title, lines, func, color):
        # card body with a thin border in the stage colour
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=1.3, edgecolor=color, facecolor="white", zorder=2))
        # coloured title band across the top of the card
        band = 0.46
        ax.add_patch(FancyBboxPatch(
            (x, y + h - band), w, band,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0, facecolor=color, zorder=3))
        ax.add_patch(FancyBboxPatch(
            (x, y + h - band), w, band / 2, boxstyle="square,pad=0",
            linewidth=0, facecolor=color, zorder=3))
        ax.text(x + 0.16, y + h - band / 2, f"{number}  {title}",
                ha="left", va="center", fontsize=10.5, fontweight="bold",
                color="white", zorder=4)
        ax.text(x + 0.16, y + h - band - 0.16, "\n".join(lines),
                ha="left", va="top", fontsize=8.6, color=TEXT,
                linespacing=1.45, zorder=4)
        ax.text(x + 0.16, y + 0.14, func, ha="left", va="bottom",
                fontsize=8, color=color, family="monospace",
                fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.3, color=color, zorder=1,
            connectionstyle="arc3,rad=0"))

    def lane(x1, x2, label, color):
        ax.plot([x1, x2], [-0.3, -0.3], color=color, lw=3,
                solid_capstyle="round")
        ax.text((x1 + x2) / 2, -0.62, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")

    card(0.1, 1.9, 2.15, 2.6, 1, "Raw dumps",
         ["6 monthly .txt.gz files", "Oct 2026, 18.5 GB",
          "works, editions, authors,", "wikidata, reading log,",
          "ratings"], "discover_dumps", GREY)
    card(2.6, 1.9, 2.15, 2.6, 2, "Parse",
         ["PySpark, explicit schemas", "free-text dates -> years",
          "publisher names cleaned", "6 concurrent jobs"],
         "build_parquet_tables", INK)
    card(5.1, 1.9, 2.0, 2.6, 3, "Parquet",
         ["6 clean tables, 3.2 GB", "field coverage profile",
          "duplicate-key check"], "profile_tables", INK)
    card(7.55, 4.35, 2.55, 1.95, "4a", "Rights status",
         ["death years: OL + Wikidata", "US, PH and life+70 rules",
          "data-quality flags"], "estimate_copyright_status", PLUM)
    card(7.55, 2.2, 2.55, 1.95, "4b", "Demand",
         ["Want-to-Read, shelves", "ratings, 12-month trends"],
         "compute_demand", GREEN)
    card(7.55, 0.05, 2.55, 1.95, "4c", "Supply",
         ["recent editions, publishers", "scans, Filipino editions"],
         "compute_supply", INK)
    card(10.55, 2.2, 2.85, 2.3, 5, "Profile & lists",
         ["one row per work (41.7M)", "2027 watchlist (53 works)",
          "public-domain list", "sensitivity checks"],
         "build_watchlist", AMBER)
    card(10.55, 0.05, 2.85, 1.85, 6, "Validation",
         ["10 independent recounts", "hand-checked audit, 30 rows"],
         "audit_estimates", GREY)

    arrow(2.25, 3.2, 2.6, 3.2)
    arrow(4.75, 3.2, 5.1, 3.2)
    for y, color in ((5.3, PLUM), (3.18, GREEN), (1.03, INK)):
        arrow(7.1, 3.2, 7.55, y, color)
        arrow(10.1, y, 10.55, 3.35, color)
    arrow(11.98, 2.2, 11.98, 1.9, AMBER)

    lane(0.1, 7.1, "INGEST ONCE  (Sections 5.2-5.4)", INK)
    lane(7.55, 10.1, "ANALYSE EVERY WORK  (6.1-6.4)", PLUM)
    lane(10.55, 13.4, "SELECT & VERIFY  (6.5-6.7)", AMBER)
    fig.tight_layout()
    return fig


def plot_demand_concentration(concentration):
    """Lorenz curve of Want-to-Read entries across shelved works."""
    lorenz = concentration["lorenz"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--",
            label="perfect equality")
    ax.plot(100 * lorenz["share_of_works"],
            100 * lorenz["share_of_entries"], color=GREEN, lw=2.2,
            label="Want-to-Read entries")
    ax.fill_between(100 * lorenz["share_of_works"],
                    100 * lorenz["share_of_entries"],
                    100 * lorenz["share_of_works"], color=GREEN,
                    alpha=0.12)
    ax.set_xlabel("Cumulative share of shelved works (%), least wanted "
                  "first")
    ax.set_ylabel("Cumulative share of entries (%)")
    ax.set_title(f"Demand is concentrated (Gini = "
                 f"{concentration['gini']:.2f})")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_demand_by_decade(table):
    """Three small panels sharing the decade axis: how often works are
    shelved, how wanted shelved works are, and how often they have a
    recent edition. Separate panels keep each measure on its own scale.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    panels = [
        ("pct_shelved", "Works with any Want-to-Read (%)", GREEN),
        ("wtr_per_shelved_work", "Entries per shelved work", GREEN),
        ("pct_recent_edition",
         f"Works with an edition since {RECENT_FROM_YEAR} (%)",
         INK),
    ]
    for ax, (column, title, color) in zip(axes, panels):
        ax.plot(table["decade"], table[column], color=color, lw=2,
                marker="o", ms=4)
        ax.axvspan(1925, 1935, color=AMBER, alpha=0.15, lw=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Decade of first publication")
        _style(ax)
    axes[0].annotate("1930s: the 2027\\nentry cohort", xy=(1930, 0),
                     xycoords=("data", "axes fraction"),
                     xytext=(1840, 0.85), textcoords=("data",
                                                      "axes fraction"),
                     fontsize=8, color=AMBER)
    fig.tight_layout()
    return fig


def plot_persistence(persistence):
    """Share of works that received a new Want-to-Read entry in the
    hold-out year, for each group defined before the cutoff."""
    table = persistence["table"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    colors = [GREEN, INK_LIGHT, GREY][:len(table)]
    bars = ax.barh(table["group"], table["pct_with_new_entry"],
                   color=colors)
    ax.invert_yaxis()
    for bar, value in zip(bars, table["pct_with_new_entry"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"Works with at least one new Want-to-Read entry in the "
                  f"12 months after {persistence['cutoff']} (%)")
    ax.set_title("Works wanted before the cutoff stay wanted after it")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
