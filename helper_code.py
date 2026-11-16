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
Part 2  Parse the dumps and save cleaned Parquet tables; merged keys
Part 3  Data-quality profile (coverage of the fields the analysis needs)
Part 4  Author death years (Open Library text; Wikidata when available)
Part 5  Estimate copyright status per work (US, Philippines, life+70)
Part 6  Demand: reading-log shelves, ratings and Want-to-Read trends
Part 7  Supply: editions, recent editions, publishers, scans, Filipino
Part 8  Duplicate titles, 2027 watchlist and public-domain list
Part 9  Validation: independent recounts of the published numbers
Part 10 Charts

Example
-------
>>> spark = create_spark()
>>> dumps = discover_dumps(resolve_data_dir())   # Jojie, 20221007 snapshot
>>> paths = build_parquet_tables(spark, dumps, resolve_parquet_dir())
>>> tables = load_tables(spark, paths)
>>> deaths = reconcile_author_deaths(tables["authors"],
...                                  tables.get("wikidata"))
>>> status = estimate_copyright_status(tables["works"], tables["editions"],
...                                    deaths["author_deaths"])

Rights classifications are estimates computed from catalogue metadata.
They are not legal advice and must be checked against primary records
(for example US renewal records) before any reprint decision.
"""

from datetime import date
from pathlib import Path
import getpass
import json
import os
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pyspark is imported lazily inside functions that need it, so that the
# pure-Python rule functions can be imported and read without Java.

SEED = 42

# The analysis date: rights are evaluated as they will stand on this day.
REFERENCE_DATE = date(2027, 1, 1)
REFERENCE_YEAR = REFERENCE_DATE.year

# Where the data lives on Jojie, the course's JupyterHub server. The shared
# folder holds one sub-folder per snapshot; only 20221007 contains the
# separate works, editions, authors, reading-log and ratings dumps (the
# other snapshots hold only the full revision history, ol_cdump), so it is
# the snapshot this analysis uses.
DATA_ROOT = "/mnt/data/public/openlibrary"
DATA_SNAPSHOT = "20221007"

# The snapshot is Open Library's dump of September 30, 2022 (its newest
# reading-log and rating entries are dated that day; the copy on Jojie was
# made on October 7, 2022). Later years in a record are impossible.
DUMP_DATE = date(2022, 9, 30)
MAX_VALID_YEAR = DUMP_DATE.year
MIN_VALID_YEAR = 1450  # printing with movable type in Europe

# "Recent" supply window: the ten calendar years ending with the dump year.
RECENT_FROM_YEAR = MAX_VALID_YEAR - 9

# Filipino-language codes used by Open Library (MARC language codes).
FILIPINO_LANGUAGE_KEYS = ("/languages/tgl", "/languages/fil")

# Wikidata items that matter here.
WIKIDATA_PHILIPPINES = "Q928"

# Dumps the analysis needs. Covers, lists and revision history are excluded.
DUMP_NAMES = ("works", "editions", "authors", "reading-log", "ratings")
# Used when present, skipped otherwise. The Jojie snapshot has no Wikidata
# dump, so death years then come from Open Library's own records only.
OPTIONAL_DUMP_NAMES = ("wikidata",)

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


def resolve_data_dir(custom_dir=None, root=DATA_ROOT,
                     snapshot=DATA_SNAPSHOT):
    """Return the folder that holds the Open Library dumps.

    By default this is the fixed Jojie location DATA_ROOT/DATA_SNAPSHOT
    (/mnt/data/public/openlibrary/20221007); no other folder is searched,
    so every run reads exactly the same data. `custom_dir` overrides it
    (the tests use this with hand-made miniature dumps). The folder must
    exist; it is not checked for individual files here, because
    discover_dumps does that and reports which dump is missing.

    Returns
    -------
    Path
    """
    path = Path(custom_dir) if custom_dir else Path(root) / snapshot
    if not path.is_dir():
        raise FileNotFoundError(
            f"Open Library data folder not found: {path}. On Jojie the "
            f"dumps are in {DATA_ROOT}/{DATA_SNAPSHOT}."
        )
    return path


def resolve_parquet_dir(custom_dir=None):
    """Return the folder for cleaned Parquet tables (created if needed).

    Uses an explicit argument, then OL_PARQUET_DIR, then ./parquet.
    """
    path = Path(custom_dir or os.environ.get("OL_PARQUET_DIR", "parquet"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_dumps(data_dir, names=DUMP_NAMES + OPTIONAL_DUMP_NAMES,
                   required=True, optional=OPTIONAL_DUMP_NAMES):
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
    optional : iterable of str
        Dumps that are never required (Wikidata): left out when absent.

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
        elif required and name not in optional:
            raise FileNotFoundError(f"No {name} dump in {data_dir}")
    return found


def _folder_size(path):
    """Total size in bytes of all files under a folder (0 if absent)."""
    path = Path(path)
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def dump_inventory(dumps, parquet_dir=None,
                   names=DUMP_NAMES + OPTIONAL_DUMP_NAMES):
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
    """Date label of one dump file (ISO text) or None.

    A dated file name (ol_dump_works_2022-09-30.txt.gz) gives its own
    date. A file named ..._latest sits in a snapshot folder such as
    20221007; if that folder is DATA_SNAPSHOT, DUMP_DATE is returned,
    otherwise the folder's own date.
    """
    path = Path(path)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    folder = path.parent.name
    if folder == DATA_SNAPSHOT:
        return DUMP_DATE.isoformat()
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", folder)
    return "-".join(match.groups()) if match else None


def check_dump_date(dumps, expected=DUMP_DATE.isoformat()):
    """Compare the dumps' dates with the date the report's text describes.

    Returns a short message: confirmation when every dump matches
    `expected`, otherwise a warning naming the dates found, since the
    numbers quoted in the narrative come from the `expected` dump.
    """
    dates = sorted({dump_date_of(path) or "unknown"
                    for path in dumps.values()})
    if dates == [expected]:
        return f"All dumps are dated {expected}, as described in the text."
    return (
        f"WARNING: dumps dated {', '.join(dates) or 'unknown'} were found, "
        f"but the report's text quotes the {expected} dump. The tables and "
        "charts will reflect the new data; update the numbers in the "
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
        """Schema of a list of Wikidata statements of one type."""
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
        """Year of one statement's time value (null if too imprecise)."""
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
        """True for a statement Wikidata does not mark deprecated."""
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


PARSERS = {
    "works": parse_works,
    "editions": parse_editions,
    "authors": parse_authors,
    "wikidata": parse_wikidata,
    "reading-log": parse_reading_log,
    "ratings": parse_ratings,
}


def build_parquet_tables(spark, dumps, parquet_dir, overwrite=False,
                         parallel=True, staging_dir=None, progress=True,
                         keep_staged=False):
    """Parse each dump once and save it as Parquet.

    A table whose folder already contains a _SUCCESS marker is reused, so a
    second "Run All" of the report takes minutes rather than hours. Pass
    overwrite=True after downloading a newer dump.

    Ingestion has two steps. stage_dumps first decompresses every .gz
    dump to plain text in `staging_dir` (default: the OL_STAGING_DIR
    environment variable, else a folder in the system temp directory;
    about 58 GB for the 2022 dumps), printing a progress bar. Spark then
    parses the plain text, which it can split across all cores, and the
    staged copies are deleted afterwards unless keep_staged=True. With
    parallel=True the dumps are parsed as concurrent Spark jobs. The raw
    line count of each dump is saved next to the tables for
    ingestion_check.

    Optional dumps (Wikidata) are built only when their raw file is in
    `dumps` or their table already exists; otherwise they are left out.

    Returns
    -------
    dict
        {name: Path to the Parquet folder}
    """
    parquet_dir = Path(parquet_dir)

    def folder(name):
        """Parquet folder of one dump (hyphens become underscores)."""
        return parquet_dir / name.replace("-", "_")

    check_manifest(parquet_dir, dumps)
    # An optional table is kept only if its dump is present, or if it
    # exists already and no rebuild was requested.
    names = list(DUMP_NAMES) + [
        name for name in OPTIONAL_DUMP_NAMES
        if name in dumps
        or (not overwrite and (folder(name) / "_SUCCESS").is_file())
    ]
    out = {name: folder(name) for name in names}
    todo = [
        name for name in names
        if overwrite or not (out[name] / "_SUCCESS").is_file()
    ]
    missing = [name for name in todo if name not in dumps]
    if missing:
        raise FileNotFoundError(
            f"No Parquet table and no raw dump for: {missing}"
        )

    if not todo:
        return out
    # Two runs writing the same folders delete each other's files (a
    # first attempt lost 30 million rows this way), so only one may run.
    lock = parquet_dir / ".ingestion.lock"
    acquire_lock(lock)
    try:
        return _build(spark, dumps, todo, out, parquet_dir, parallel,
                      staging_dir, progress, keep_staged)
    finally:
        lock.unlink(missing_ok=True)


MANIFEST = "manifest.json"


def check_manifest(parquet_dir, dumps):
    """Refuse to reuse Parquet tables built from other data or rules.

    Every cached table (including author_deaths, work_status and
    work_profile, which materialize reuses) depends on the snapshot and on
    the date constants. They are recorded in manifest.json the first time
    the folder is used; a later run with a different snapshot or
    constants raises an error instead of silently mixing results.
    """
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    folders = sorted({str(Path(p).parent) for p in dumps.values()})
    current = {"data_folders": folders, "dump_date": DUMP_DATE.isoformat(),
               "recent_from_year": RECENT_FROM_YEAR,
               "max_valid_year": MAX_VALID_YEAR,
               "reference_date": REFERENCE_DATE.isoformat()}
    path = parquet_dir / MANIFEST
    if path.is_file():
        saved = json.loads(path.read_text())
        if not folders:
            # reusing tables without raw dumps: only the rules can differ
            current["data_folders"] = saved.get("data_folders")
        if saved != current:
            raise RuntimeError(
                f"{parquet_dir} was built from {saved}, but this run uses "
                f"{current}. Delete the folder (or set OL_PARQUET_DIR to "
                "a new one) so every table is rebuilt from the same data."
            )
    else:
        path.write_text(json.dumps(current, indent=1))
    return current


def acquire_lock(lock):
    """Create a lock file holding this process's host and PID.

    A lock left behind by a process that no longer runs on this host
    (for example a killed kernel) is taken over; a live one raises.
    """
    import socket

    lock = Path(lock)
    me = f"{socket.gethostname()} {os.getpid()}"
    if lock.is_file():
        host, _, pid = lock.read_text().partition(" ")
        alive = False
        if host == socket.gethostname() and pid.strip().isdigit():
            try:
                os.kill(int(pid), 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        if alive:
            raise RuntimeError(
                f"Another run (PID {pid.strip()}) is writing to "
                f"{lock.parent}. Wait for it to finish.")
    lock.write_text(me)


def _build(spark, dumps, todo, out, parquet_dir, parallel, staging_dir,
           progress, keep_staged):
    """Body of build_parquet_tables, run while holding the lock."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    # one staging folder per Parquet folder, so separate runs never share
    tag = hashlib.md5(str(parquet_dir.resolve()).encode()).hexdigest()[:8]
    staging_dir = Path(staging_dir or os.environ.get(
        "OL_STAGING_DIR",
        Path(tempfile.gettempdir()) / f"ol_staging_{getpass.getuser()}"
        / tag))

    # Step 1: decompress every .gz dump to plain text, all at once.
    staged = stage_dumps({name: dumps[name] for name in todo}, staging_dir,
                         progress=progress)
    counts_file = parquet_dir / RAW_LINE_COUNTS
    counts = (json.loads(counts_file.read_text())
              if counts_file.is_file() else {})

    def write(name):
        """Parse one input and write it as Parquet."""
        text_path, n_lines = staged[name]
        table = PARSERS[name](spark, text_path)
        # Split the output into files of at most 2 million rows without a
        # shuffle; a shuffle would copy the whole table to disk again.
        (table.write.mode("overwrite")
         .option("maxRecordsPerFile", 2_000_000)
         .parquet(str(out[name])))
        counts[name] = n_lines
        return name

    # Step 2: parse the plain text with Spark. Plain text is splittable,
    # so each dump is read by many tasks in parallel.
    todo.sort(key=lambda n: -staged[n][0].stat().st_size)
    try:
        if parallel and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                list(pool.map(write, todo))
        else:
            for name in todo:
                write(name)
    finally:
        counts_file.parent.mkdir(parents=True, exist_ok=True)
        counts_file.write_text(json.dumps(counts, indent=1))
        if not keep_staged:
            for name in todo:
                text_path = staged[name][0]
                if text_path != Path(dumps[name]):
                    text_path.unlink(missing_ok=True)
    return out


# File in the Parquet folder that records how many lines each raw dump had.
RAW_LINE_COUNTS = "raw_line_counts.json"


def stage_dumps(dumps, staging_dir, progress=True, every=30):
    """Decompress .gz dumps to plain text in parallel, with progress.

    Why this step exists: a gzip file cannot be split, so Spark reads a
    .gz dump with ONE task however many cores the machine has; the 8 GB
    editions dump alone kept one core busy for over half an hour. The
    plain-text copy can be split into 128 MB blocks, so Spark parses it
    with every core. Counting lines here, with a decompressor independent
    of Spark's, also gives an exact reference for ingestion_check.

    Each decompression runs in its own thread (zlib releases the GIL).
    Every `every` seconds a progress line reports the share of compressed
    bytes read per dump and an estimated time to finish. Lines are
    counted while writing, so the Parquet tables can later be checked
    against the raw files (see ingestion_check).

    Returns
    -------
    dict
        {name: (Path of the plain-text file, number of lines)}.
        Uncompressed inputs are used in place.
    """
    import gzip
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    total = {n: Path(p).stat().st_size for n, p in dumps.items()}
    done_bytes = {n: 0 for n in dumps}
    finished = threading.Event()
    start = time.time()

    def stage(name):
        """Decompress one dump (or count lines of a plain file)."""
        source = Path(dumps[name])
        if source.suffix != ".gz":
            with open(source, "rb") as handle:
                lines = sum(chunk.count(b"\n") for chunk in
                            iter(lambda: handle.read(1 << 24), b""))
            done_bytes[name] = total[name]
            return name, (source, lines)
        target = staging_dir / source.name[:-3]
        lines = 0
        with open(source, "rb") as raw, \
                gzip.GzipFile(fileobj=raw) as unzipped, \
                open(target, "wb") as out:
            for chunk in iter(lambda: unzipped.read(1 << 24), b""):
                out.write(chunk)
                lines += chunk.count(b"\n")
                done_bytes[name] = raw.tell()
        done_bytes[name] = total[name]
        return name, (target, lines)

    def report():
        """Print a progress line every `every` seconds until done."""
        while not finished.wait(every):
            share = sum(done_bytes.values()) / max(sum(total.values()), 1)
            elapsed = time.time() - start
            eta = elapsed / share - elapsed if share > 0 else float("nan")
            parts = ", ".join(
                f"{n} {100 * done_bytes[n] / max(total[n], 1):.0f}%"
                for n in dumps)
            bar = "#" * int(30 * share) + "-" * (30 - int(30 * share))
            print(f"[{bar}] {100 * share:5.1f}%  {parts}  "
                  f"elapsed {elapsed / 60:.1f} min, about "
                  f"{eta / 60:.1f} min left", flush=True)

    reporter = threading.Thread(target=report, daemon=True)
    if progress:
        reporter.start()
    try:
        with ThreadPoolExecutor(max_workers=max(len(dumps), 1)) as pool:
            staged = dict(pool.map(stage, list(dumps)))
    except BaseException:
        # do not leave tens of GB of partial files in a shared /tmp
        for name, source in dumps.items():
            if Path(source).suffix == ".gz":
                (staging_dir / Path(source).name[:-3]).unlink(
                    missing_ok=True)
        raise
    finally:
        finished.set()
    if progress:
        print(f"Decompressed {len(dumps)} dumps in "
              f"{(time.time() - start) / 60:.1f} min", flush=True)
    return staged


def ingestion_check(tables, parquet_dir):
    """Compare each Parquet table's rows with its raw dump's line count.

    Works, editions and authors keep every line of their dump (one record
    per line, all of the dump's own type). Reading-log and ratings drop
    rows by design (custom shelves, missing work keys), so for them the
    kept share is reported, not tested.

    Returns
    -------
    DataFrame
        table, raw_lines, parquet_rows, kept_pct, complete.
    """
    counts_file = Path(parquet_dir) / RAW_LINE_COUNTS
    counts = (json.loads(counts_file.read_text())
              if counts_file.is_file() else {})
    rows = []
    for name, raw_lines in counts.items():
        key = name.replace("-", "_")
        if key not in tables:
            continue
        kept = tables[key].count()
        rows.append({
            "table": key, "raw_lines": raw_lines, "parquet_rows": kept,
            "kept_pct": round(100 * kept / raw_lines, 3) if raw_lines
            else np.nan,
            "complete": (kept == raw_lines) if key in
            ("works", "editions", "authors") else None,
        })
    return pd.DataFrame(rows)


# The other two files of the snapshot. They are read once each, straight
# from gzip (staging them would need about 200 GB of scratch disk), and
# reduced to compact Parquet tables.
HISTORY_FILES = {
    # latest version of EVERY record type (works, editions and authors,
    # but also redirects, deletions, lists, subjects, ...)
    "full_dump": "ol_dump_latest.txt.gz",
    # the same file again, keeping only the merged-key redirects
    "redirects": "ol_dump_latest.txt.gz",
    # every revision of every record since the catalogue began
    "revisions": "ol_cdump_latest.txt.gz",
}


def parse_full_dump(spark, path):
    """Record-type census of the all-types dump.

    Only the first two columns are read: the record type and the key's
    namespace (/works, /books, /authors, ...). A redirect or deletion
    keeps the namespace of the record it replaced, so the census shows
    how many works, editions and authors were merged or removed.

    Returns
    -------
    DataFrame
        record_type, key_prefix, records.
    """
    from pyspark.sql import functions as F

    parts = F.split(F.col("value"), "\t", 3)
    return (
        spark.read.text(str(path))
        .select(parts.getItem(0).alias("record_type"),
                F.regexp_extract(parts.getItem(1), r"^(/[a-z0-9_]+)/", 1)
                .alias("key_prefix"))
        .groupBy("record_type", "key_prefix")
        .agg(F.count(F.lit(1)).alias("records"))
    )


def parse_revisions(spark, path):
    """One row per revision in the complete (all-revisions) dump.

    Every revision keeps its record type, revision number and the date it
    was saved. Author revisions additionally keep the key and the
    free-text birth and death dates, parsed with the same rule as the
    authors table, so the history of each author's death year can be
    traced. Keys of other types are dropped to keep the table small.

    Returns
    -------
    DataFrame
        record_type, key (authors only), revision, modified (date),
        name, death_date, ol_death_year, ol_birth_year (authors only).
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    schema = T.StructType([
        T.StructField("name", T.StringType()),
        T.StructField("birth_date", T.StringType()),
        T.StructField("death_date", T.StringType()),
    ])
    parts = F.split(F.col("value"), "\t", 5)
    record_type = parts.getItem(0)
    is_author = record_type == "/type/author"
    df = spark.read.text(str(path)).select(
        record_type.alias("record_type"),
        F.when(is_author, parts.getItem(1)).alias("key"),
        parts.getItem(2).cast("int").alias("revision"),
        F.to_date(F.substring(parts.getItem(3), 1, 10)).alias("modified"),
        F.when(is_author, F.from_json(parts.getItem(4), schema)).alias("j"),
    )
    return df.select(
        "record_type", "key", "revision", "modified",
        F.col("j.name").alias("name"),
        F.col("j.death_date").alias("death_date"),
        life_year_from_text(F.col("j.death_date")).alias("ol_death_year"),
        life_year_from_text(F.col("j.birth_date")).alias("ol_birth_year"),
    )


def parse_redirects(spark, path):
    """Redirect records of the all-types dump: merged key -> surviving key.

    When Open Library editors merge two records, the losing key is kept as
    a /type/redirect whose JSON "location" names the record that replaced
    it. Reading-log and rating rows, and the author lists of works, can
    still carry a merged key, so without this table those rows would
    silently drop out of every join.

    Returns
    -------
    DataFrame
        old_key, new_key (one row per redirect, any namespace).
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    schema = T.StructType([T.StructField("location", T.StringType())])
    parts = F.split(F.col("value"), "\t", 5)
    return (
        spark.read.text(str(path))
        .filter(F.col("value").startswith("/type/redirect\t"))
        .select(parts.getItem(1).alias("old_key"),
                F.from_json(parts.getItem(4), schema)
                .getField("location").alias("new_key"))
        .filter(F.col("new_key").isNotNull())
    )


def resolve_redirect_chains(redirects, max_hops=5):
    """Follow redirect chains (A -> B -> C) to their final target.

    A record can be merged more than once, so a redirect may point to
    another redirect. Each pass replaces a target that is itself a
    redirect with that redirect's target; `max_hops` passes resolve chains
    of up to max_hops + 1 links. A key that still points to a redirect
    afterwards (a loop) is dropped, as is a redirect to itself.

    Returns
    -------
    DataFrame
        old_key, new_key, where new_key is never itself an old_key.
    """
    from pyspark.sql import functions as F

    resolved = redirects.select("old_key", "new_key")
    step = redirects.select(F.col("old_key").alias("new_key"),
                            F.col("new_key").alias("next_key"))
    for _ in range(max_hops):
        resolved = (
            resolved.join(step, "new_key", "left")
            .select("old_key",
                    F.coalesce("next_key", "new_key").alias("new_key"))
        )
    return (
        resolved.join(step.select(F.col("new_key").alias("old_again")),
                      F.col("new_key") == F.col("old_again"), "left_anti")
        .filter(F.col("old_key") != F.col("new_key"))
        .dropDuplicates(["old_key"])
    )


def apply_redirects(df, column, redirects):
    """Replace merged keys in `column` by the key that replaced them.

    `redirects` must already be resolved (resolve_redirect_chains). Rows
    whose key was never merged are returned unchanged; the table is small
    (under a million rows), so it is broadcast to every task instead of
    shuffling the large side of the join.
    """
    from pyspark.sql import functions as F

    lookup = F.broadcast(redirects.select(
        F.col("old_key").alias(column), F.col("new_key").alias("_new")))
    return (
        df.join(lookup, column, "left")
        .withColumn(column, F.coalesce("_new", column))
        .drop("_new")
    )


def resolve_merged_keys(tables, redirects):
    """Point every table at the surviving record of each merged key.

    Returns a copy of `tables` in which the work_key of editions,
    reading_log and ratings, and the author_keys of works, have been
    passed through the resolved redirects. Works whose author list holds
    no merged key are left untouched: only the affected works are
    exploded, so the 26-million-row works table is never shuffled.

    Parameters
    ----------
    tables : dict of Spark DataFrames
        Must hold works, editions, reading_log and ratings.
    redirects : Spark DataFrame
        old_key, new_key from resolve_redirect_chains.
    """
    from pyspark.sql import functions as F

    out = dict(tables)
    for name in ("editions", "reading_log", "ratings"):
        out[name] = apply_redirects(tables[name], "work_key", redirects)

    lookup = F.broadcast(redirects.select(
        F.col("old_key").alias("author_key"), "new_key"))
    # one small map per affected work: merged author key -> surviving key
    fixes = (
        tables["works"]
        .select("work_key", F.explode("author_keys").alias("author_key"))
        .join(lookup, "author_key")
        .groupBy("work_key")
        .agg(F.map_from_entries(F.collect_set(
            F.struct("author_key", "new_key"))).alias("_fix"))
        # small (one row per affected work) and needed by every later
        # use of the works table, so compute it once and keep it
        .cache()
    )
    out["works"] = (
        tables["works"].join(F.broadcast(fixes), "work_key", "left")
        .withColumn("author_keys", F.when(
            F.col("_fix").isNull(), F.col("author_keys")
        ).otherwise(F.array_distinct(F.transform(
            "author_keys",
            lambda k: F.coalesce(F.col("_fix")[k], k)))))
        .drop("_fix")
    )
    return out


def redirect_impact(raw_tables, resolved_tables):
    """Rows that find their record before and after resolving merges.

    For each link the analysis joins on, counts the rows whose key is a
    live record in the raw tables, and again after resolve_merged_keys.
    The difference is what a naive join would have dropped; the rest
    (still_unmatched) point to deleted records and cannot be recovered.

    Returns
    -------
    DataFrame
        link, rows, matched_raw, matched_resolved, recovered,
        recovered_pct, still_unmatched, still_unmatched_pct.
    """
    from pyspark.sql import functions as F

    work_keys = raw_tables["works"].select("work_key")
    author_keys = raw_tables["authors"].select("author_key")

    def matched(tables, name):
        """Rows of a fact table whose work_key is a live work."""
        rows = tables[name].filter(F.col("work_key").isNotNull())
        return rows.count(), rows.join(work_keys, "work_key",
                                       "left_semi").count()

    def author_links(tables):
        """Work-author links, and those whose author record is live."""
        links = tables["works"].select(
            F.explode("author_keys").alias("author_key"))
        return links.count(), links.join(author_keys, "author_key",
                                         "left_semi").count()

    rows = []
    labels = {"reading_log": "reading-log entry -> work",
              "ratings": "rating -> work", "editions": "edition -> work"}
    for name, label in labels.items():
        total, before = matched(raw_tables, name)
        _, after = matched(resolved_tables, name)
        rows.append((label, total, before, after, total))
    total, before = author_links(raw_tables)
    total_after, after = author_links(resolved_tables)
    rows.append(("work -> author", total, before, after, total_after))
    table = pd.DataFrame(rows, columns=["link", "rows", "matched_raw",
                                        "matched_resolved", "_rows_after"])
    table["recovered"] = table["matched_resolved"] - table["matched_raw"]
    table["recovered_pct"] = (100 * table["recovered"]
                              / table["rows"]).round(3)
    table["still_unmatched"] = (table["_rows_after"]
                                - table["matched_resolved"])
    table["still_unmatched_pct"] = (100 * table["still_unmatched"]
                                    / table["_rows_after"]).round(3)
    return table.drop(columns="_rows_after")


HISTORY_PARSERS = {"full_dump": parse_full_dump,
                   "redirects": parse_redirects,
                   "revisions": parse_revisions}


def build_history_tables(spark, data_dir, parquet_dir, overwrite=False):
    """Reduce the all-types dump and the revision dump to Parquet.

    Each file is one gzip stream, so Spark reads it with a single task;
    the two files are read as concurrent jobs. This is the slowest step
    of the pipeline (about an hour for the 27.7 GB revision dump), and
    like build_parquet_tables it runs only once: a table whose folder
    holds a _SUCCESS marker is reused.

    Returns
    -------
    dict
        {"full_dump": Path, "redirects": Path, "revisions": Path}
    """
    from concurrent.futures import ThreadPoolExecutor

    data_dir, parquet_dir = Path(data_dir), Path(parquet_dir)
    out = {name: parquet_dir / name for name in HISTORY_FILES}
    todo = [name for name in HISTORY_FILES
            if overwrite or not (out[name] / "_SUCCESS").is_file()]
    if not todo:
        return out
    for name in todo:
        if not (data_dir / HISTORY_FILES[name]).is_file():
            raise FileNotFoundError(
                f"{HISTORY_FILES[name]} not found in {data_dir}")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    lock = parquet_dir / ".history.lock"
    acquire_lock(lock)

    def write(name):
        """Parse one input and write it as Parquet."""
        table = HISTORY_PARSERS[name](spark, data_dir / HISTORY_FILES[name])
        (table.write.mode("overwrite")
         .option("maxRecordsPerFile", 20_000_000)
         .parquet(str(out[name])))

    try:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            list(pool.map(write, todo))
    finally:
        lock.unlink(missing_ok=True)
    return out


def snapshot_inventory(root=DATA_ROOT, needed=DUMP_NAMES):
    """List every Open Library snapshot folder on the server.

    For each folder: the files it holds, their total size, and which of
    the dumps this analysis needs are present. This is the evidence for
    the choice of snapshot: only a folder with all `needed` dumps can
    support the rights, demand and supply measures together.

    Returns
    -------
    DataFrame
        snapshot, files, GiB, and one True/False column per needed dump.
    """
    rows = []
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(p.name for p in folder.glob("ol_*"))
        row = {"snapshot": folder.name, "files": ", ".join(files),
               "GiB": round(sum((folder / f).stat().st_size
                                for f in files) / 2**30, 1)}
        for name in needed:
            row[name] = any(f.startswith(f"ol_dump_{name}_") for f in files)
        rows.append(row)
    return pd.DataFrame(rows)


def raw_input_size(data_dir):
    """Size of every input file, like `du -sch` on the snapshot folder.

    Returns
    -------
    DataFrame
        file, GiB and GB per file plus a total row. du reports binary
        gigabytes (GiB, 2**30 bytes), so both units are shown.
    """
    rows = [{"file": p.name, "bytes": p.stat().st_size}
            for p in sorted(Path(data_dir).glob("ol_*.txt*"))]
    table = pd.DataFrame(rows)
    table.loc[len(table)] = {"file": "total", "bytes": table["bytes"].sum()}
    table["GiB"] = (table["bytes"] / 2**30).round(2)
    table["GB"] = (table["bytes"] / 1e9).round(2)
    return table.drop(columns="bytes")


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


def describe_raw_data(tables):
    """Simple descriptive statistics for every raw dataset.

    One Spark pass per table computes a handful of statistics that a
    reader needs to picture the data: its size, its key distributions and
    the period it covers. The all-types dump (full_dump) and the revision
    history (revisions) are included when they are in `tables`.

    Returns
    -------
    DataFrame
        dataset, statistic, value (value is pre-formatted text, so counts,
        shares and dates can share one column).
    """
    from pyspark.sql import functions as F

    rows = []

    def add(dataset, statistic, value):
        """Append one result row."""
        rows.append({"dataset": dataset, "statistic": statistic,
                     "value": value})

    def pct(part, whole):
        """Format part / whole as a percentage with one decimal."""
        return f"{100 * (part or 0) / whole:.1f}%" if whole else "-"

    w = tables["works"].agg(
        F.count(F.lit(1)).alias("n"),
        F.sum((F.size("author_keys") > 0).cast("int")).alias("with_author"),
        F.avg("n_subjects").alias("subjects_mean"),
        F.sum(F.col("is_fiction").cast("int")).alias("fiction"),
        F.min("work_year").alias("min_year"),
        F.expr("percentile_approx(work_year, 0.5, 10000)").alias("median"),
    ).collect()[0]
    add("works", "records", f"{w['n']:,}")
    add("works", "with at least one author key",
        pct(w["with_author"], w["n"]))
    add("works", "subjects per work (mean)", f"{w['subjects_mean']:.2f}")
    add("works", "tagged as fiction", pct(w["fiction"], w["n"]))
    add("works", "median own first-publish year (where given)",
        f"{w['median']}")

    e = tables["editions"]
    per_work = e.filter(F.col("work_key").isNotNull()).groupBy(
        "work_key").count()
    epw = per_work.agg(
        F.avg("count").alias("mean"),
        F.expr("percentile_approx(count, 0.5, 10000)").alias("median"),
        F.max("count").alias("max"),
    ).collect()[0]
    es = e.agg(
        F.count(F.lit(1)).alias("n"),
        F.expr("percentile_approx(publish_year, 0.5, 10000)").alias(
            "median_year"),
        F.sum((F.col("publish_year") >= RECENT_FROM_YEAR).cast("int"))
        .alias("recent"),
    ).collect()[0]
    n_publishers = (e.select(F.explode("publisher_keys").alias("p"))
                    .agg(F.countDistinct("p")).collect()[0][0])
    n_languages = (e.select(F.explode("language_keys").alias("l"))
                   .agg(F.countDistinct("l")).collect()[0][0])
    add("editions", "records", f"{es['n']:,}")
    add("editions", "editions per linked work (mean / median / max)",
        f"{epw['mean']:.2f} / {epw['median']} / {epw['max']:,}")
    add("editions", "median publication year", f"{es['median_year']}")
    add("editions", f"published {RECENT_FROM_YEAR} or later",
        pct(es["recent"], es["n"]))
    add("editions", "distinct normalized publishers", f"{n_publishers:,}")
    add("editions", "distinct languages", f"{n_languages:,}")

    a = tables["authors"].agg(
        F.count(F.lit(1)).alias("n"),
        F.sum(F.col("ol_birth_year").isNotNull().cast("int")).alias("birth"),
        F.sum(F.col("ol_death_year").isNotNull().cast("int")).alias("death"),
        F.expr("percentile_approx(ol_death_year, 0.5, 10000)").alias(
            "median_death"),
    ).collect()[0]
    add("authors", "records", f"{a['n']:,}")
    add("authors", "with a usable birth year", pct(a["birth"], a["n"]))
    add("authors", "with a usable death year", pct(a["death"], a["n"]))
    add("authors", "median death year (where known)",
        f"{a['median_death']}")

    r = tables["reading_log"].agg(
        F.count(F.lit(1)).alias("n"),
        F.countDistinct("work_key").alias("works"),
        F.min("logged_date").alias("first"),
        F.max("logged_date").alias("last"),
        F.sum((F.col("shelf") == "Want to Read").cast("int")).alias("wtr"),
    ).collect()[0]
    add("reading-log", "shelf entries", f"{r['n']:,}")
    add("reading-log", "distinct works shelved", f"{r['works']:,}")
    add("reading-log", "Want to Read share", pct(r["wtr"], r["n"]))
    add("reading-log", "date range", f"{r['first']} to {r['last']}")

    g = tables["ratings"].agg(
        F.count(F.lit(1)).alias("n"),
        F.countDistinct("work_key").alias("works"),
        F.avg("rating").alias("mean"),
        F.min("rated_date").alias("first"),
        F.max("rated_date").alias("last"),
    ).collect()[0]
    add("ratings", "ratings", f"{g['n']:,}")
    add("ratings", "distinct works rated", f"{g['works']:,}")
    add("ratings", "mean stars (1-5)", f"{g['mean']:.2f}")
    add("ratings", "date range", f"{g['first']} to {g['last']}")

    if "full_dump" in tables:
        fd = tables["full_dump"].groupBy("record_type").agg(
            F.sum("records").alias("n")).toPandas()
        total = int(fd["n"].sum())
        add("all-types dump", "records (all types)", f"{total:,}")
        add("all-types dump", "distinct record types", f"{len(fd):,}")
        for kind in ["/type/redirect", "/type/delete"]:
            n = int(fd.loc[fd["record_type"] == kind, "n"].sum())
            add("all-types dump", f"{kind} records", pct(n, total))

    if "revisions" in tables:
        rv = tables["revisions"].agg(
            F.count(F.lit(1)).alias("n"),
            F.min("modified").alias("first"),
            F.max("modified").alias("last"),
            F.sum((F.col("revision") == 1).cast("int")).alias("created"),
            F.sum((F.col("record_type") == "/type/author").cast("int"))
            .alias("author_revs"),
        ).collect()[0]
        add("revision history", "revisions (all types)", f"{rv['n']:,}")
        add("revision history", "revisions that created a record",
            pct(rv["created"], rv["n"]))
        add("revision history", "author revisions", f"{rv['author_revs']:,}")
        add("revision history", "date range", f"{rv['first']} to "
            f"{rv['last']}")
    return pd.DataFrame(rows)


def investigate_missing(tables, author_history=None):
    """Look for the reasons behind the missing values the analysis meets.

    Three gaps matter for the rights estimates. Each is broken down by a
    variable that could explain it, so the report can say why the values
    are missing before deciding how to treat them:

    * authors without a death year, by birth period (a person born after
      1940 is probably still alive, so the gap is genuine, not an error)
      and, when the revision history is given, by whether the record was
      ever edited after its creation (an unedited record was left as the
      import created it);
    * works without their own first_publish_date, by whether an edition
      supplies a year instead;
    * editions without a work link, by whether they have a publish year.

    Parameters
    ----------
    tables : dict of Spark DataFrames
    author_history : Spark DataFrame, optional
        per_author from death_year_history (needs n_revisions).

    Returns
    -------
    dict of pandas DataFrames: authors, authors_edited, works, editions.
    """
    from pyspark.sql import functions as F

    born = F.col("ol_birth_year")
    period = (
        F.when(born.isNull(), "no birth year either")
        .when(born < 1850, "born before 1850")
        .when(born < 1900, "born 1850-1899")
        .when(born < 1940, "born 1900-1939")
        .otherwise("born 1940 or later (likely alive)")
    )
    authors = (
        tables["authors"].groupBy(period.alias("birth period")).agg(
            F.count(F.lit(1)).alias("authors"),
            F.sum(F.col("ol_death_year").isNull().cast("int")).alias(
                "without_death_year"),
        ).toPandas()
    )
    authors["missing_pct"] = (100 * authors["without_death_year"]
                              / authors["authors"]).round(1)
    authors["share_of_all_missing_pct"] = (
        100 * authors["without_death_year"]
        / authors["without_death_year"].sum()).round(1)
    authors = authors.sort_values("authors", ascending=False).reset_index(
        drop=True)

    out = {"authors": authors}
    if author_history is not None:
        edited = (
            author_history.groupBy(
                F.when(F.col("death_year").isNull(), "no death year")
                .otherwise("has death year").alias("death year"))
            .agg(F.count(F.lit(1)).alias("authors"),
                 F.sum((F.col("n_revisions") == 1).cast("int")).alias(
                     "never_edited"),
                 F.avg("n_revisions").alias("mean_revisions"))
            .toPandas()
        )
        edited["never_edited_pct"] = (100 * edited["never_edited"]
                                      / edited["authors"]).round(1)
        edited["mean_revisions"] = edited["mean_revisions"].round(2)
        out["authors_edited"] = edited

    years = (
        tables["editions"].filter(F.col("work_key").isNotNull())
        .groupBy("work_key")
        .agg(F.max(F.col("publish_year").isNotNull().cast("int")).alias(
            "edition_has_year"))
    )
    works = (
        tables["works"].select("work_key", "work_year")
        .join(years, "work_key", "left")
        .groupBy(
            F.when(F.col("work_year").isNotNull(), "own year given")
            .when(F.col("edition_has_year") == 1,
                  "no own year, an edition gives one")
            .when(F.col("edition_has_year") == 0,
                  "no own year, editions undated")
            .otherwise("no own year, no edition linked").alias("work year"))
        .count().toPandas().rename(columns={"count": "works"})
    )
    works["share_pct"] = (100 * works["works"] / works["works"].sum()).round(2)
    out["works"] = works.sort_values("works", ascending=False).reset_index(
        drop=True)

    editions = (
        tables["editions"].groupBy(
            F.when(F.col("work_key").isNotNull(), "linked to a work")
            .otherwise("no work link").alias("work link"))
        .agg(F.count(F.lit(1)).alias("editions"),
             F.round(100 * F.avg(F.col("publish_year").isNotNull()
                                 .cast("int")), 1).alias("dated_pct"),
             F.round(100 * F.avg(F.col("ocaid").isNotNull().cast("int")),
                     1).alias("digitized_pct"))
        .toPandas()
    )
    editions["share_pct"] = (100 * editions["editions"]
                             / editions["editions"].sum()).round(2)
    out["editions"] = editions
    return out


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


def reconcile_author_deaths(authors, wikidata=None):
    """Attach one death year (and birth year) to every Open Library author.

    Links an author to Wikidata in BOTH directions: the author record's
    remote_ids.wikidata Q-id, and Wikidata items whose P648 lists the
    author's OL ID. When several Wikidata items match, the latest death
    year is used (the conservative choice). Agreement between the two
    sources is measured on authors that have both.

    With wikidata=None (the Jojie snapshot has no Wikidata dump) every
    year comes from Open Library's own birth_date / death_date text, the
    Wikidata columns are null, wd_philippine_citizen is False, and the
    agreement table counts zero authors with both sources.

    Returns
    -------
    dict
        author_deaths (Spark DataFrame, one row per author) and
        source_summary / agreement (pandas tables for the report).
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    if wikidata is None:
        wikidata = authors.sparkSession.createDataFrame([], T.StructType([
            T.StructField("qid", T.StringType()),
            T.StructField("ol_author_ids", T.ArrayType(T.StringType())),
            T.StructField("wd_birth_year", T.IntegerType()),
            T.StructField("wd_death_year", T.IntegerType()),
            T.StructField("wd_philippine_citizen", T.BooleanType()),
        ]))

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
    # sums over zero rows are null; with no Wikidata they must read 0
    agreement = agreement.fillna(0).astype(int)
    n = agreement.loc[0, "authors_with_both"]
    for column in ["exact_match", "within_one_year", "differ_over_10",
                   "ph_status_would_flip"]:
        agreement[f"{column}_pct"] = (
            round(100 * agreement.loc[0, column] / n, 2) if n else np.nan
        )
    # the summaries are done; free the memory (the caller saves the
    # table as Parquet, which recomputes it once)
    deaths.unpersist()
    return {
        "author_deaths": deaths,
        "source_summary": source_summary.reset_index(drop=True),
        "agreement": agreement,
    }


def full_dump_census(full_dump, tables):
    """Live, redirected and deleted records per namespace, and a check.

    The all-types dump holds the latest version of every key, so a key
    whose record was merged into another is a /type/redirect and a
    removed one is a /type/delete. The live works, editions and authors
    in it must equal the rows of the separately parsed tables, which
    checks the two files against each other.

    Returns
    -------
    DataFrame
        namespace, live, redirect, delete, other, redirect_pct,
        parsed_table_rows, matches.
    """
    counts = full_dump.toPandas()
    live_type = {"/works": "/type/work", "/books": "/type/edition",
                 "/authors": "/type/author"}
    table_name = {"/works": "works", "/books": "editions",
                  "/authors": "authors"}
    rows = []
    for prefix, live in live_type.items():
        part = counts[counts["key_prefix"] == prefix]
        by_type = part.groupby("record_type")["records"].sum()
        n_live = int(by_type.get(live, 0))
        n_redirect = int(by_type.get("/type/redirect", 0))
        n_delete = int(by_type.get("/type/delete", 0))
        parsed = tables[table_name[prefix]].count()
        rows.append({
            "namespace": prefix, "live": n_live, "redirect": n_redirect,
            "delete": n_delete,
            "other": int(by_type.sum()) - n_live - n_redirect - n_delete,
            "redirect_pct": round(100 * n_redirect / max(by_type.sum(), 1),
                                  2),
            "parsed_table_rows": parsed, "matches": parsed == n_live,
        })
    return pd.DataFrame(rows)


def record_type_census(full_dump):
    """Records of every type in the all-types dump (pandas)."""
    table = (full_dump.toPandas().groupby("record_type")["records"].sum()
             .sort_values(ascending=False).reset_index())
    table["share_pct"] = (100 * table["records"]
                          / table["records"].sum()).round(3)
    return table


def revision_activity(revisions, start=2008, end=MAX_VALID_YEAR):
    """Records created and edits saved per year, for the three main types.

    Revision 1 of a key is its creation; every later revision is an
    edit. Returns a pandas table with one row per year and, for works,
    editions and authors, the records created and the edits made.
    """
    from pyspark.sql import functions as F

    main = {"/type/work": "works", "/type/edition": "editions",
            "/type/author": "authors"}
    table = (
        revisions.filter(F.col("record_type").isin(*main))
        .withColumn("year", F.year("modified"))
        .filter(F.col("year").between(start, end))
        .groupBy("year", "record_type")
        .agg(F.sum((F.col("revision") == 1).cast("int")).alias("created"),
             F.sum((F.col("revision") > 1).cast("int")).alias("edits"))
        .toPandas()
    )
    table["record_type"] = table["record_type"].map(main)
    wide = table.pivot(index="year", columns="record_type",
                       values=["created", "edits"]).fillna(0).astype(int)
    wide.columns = [f"{kind}_{name}" for kind, name in wide.columns]
    return wide.sort_index()


def death_year_history(revisions, author_deaths):
    """Trace every current author's death year through its revisions.

    For each author still in the catalogue: when the record was created,
    when a parsable death year first appeared, the first death year ever
    recorded, how many different death years its revisions have carried,
    and the Philippine (life + 50) status each of those years implies.

    Returns
    -------
    dict
        per_author (Spark DataFrame), coverage (pandas: per year, the
        cumulative authors created and with a death year, for authors
        alive in the current dump), stability (pandas, one row) and
        entrants (pandas: year in which the death year of each author
        who died in 1976 was first recorded).
    """
    from pyspark.sql import functions as F

    entering = REFERENCE_YEAR - 51
    revs = (
        revisions.filter((F.col("record_type") == "/type/author")
                         & F.col("key").isNotNull())
        .select(F.col("key").alias("author_key"), "revision", "modified",
                "ol_death_year")
        .join(author_deaths.select("author_key"), "author_key", "left_semi")
    )
    ph_code = (F.when(F.col("ol_death_year") < entering, "pd")
               .when(F.col("ol_death_year") == entering, "enters")
               .when(F.col("ol_death_year").isNotNull(), "protected"))
    first_death = F.min(F.when(
        F.col("ol_death_year").isNotNull(),
        F.struct("revision", "ol_death_year")))
    per_author = (
        revs.groupBy("author_key").agg(
            F.min("modified").alias("created"),
            F.min(F.when(F.col("ol_death_year").isNotNull(),
                         F.col("modified"))).alias("death_added"),
            first_death.getField("ol_death_year").alias("first_death_year"),
            F.countDistinct("ol_death_year").alias("n_death_values"),
            F.size(F.collect_set(ph_code)).alias("n_ph_codes"),
            F.count(F.lit(1)).alias("n_revisions"),
        )
        .join(author_deaths.select("author_key", "death_year"),
              "author_key", "left")
    ).cache()

    created = per_author.groupBy(F.year("created").alias("year")).count()
    added = (per_author.filter(F.col("death_year").isNotNull())
             .groupBy(F.year("death_added").alias("year")).count())
    coverage = (
        created.withColumnRenamed("count", "created")
        .join(added.withColumnRenamed("count", "death_added"), "year",
              "full_outer")
        .fillna(0).toPandas().dropna().sort_values("year")
    )
    coverage["year"] = coverage["year"].astype(int)
    coverage["authors_cum"] = coverage["created"].cumsum()
    coverage["with_death_cum"] = coverage["death_added"].cumsum()
    coverage["pct_with_death"] = (100 * coverage["with_death_cum"]
                                  / coverage["authors_cum"]).round(2)

    has_death = F.col("death_year").isNotNull()
    stability = per_author.agg(
        F.count(F.lit(1)).alias("authors"),
        F.sum(has_death.cast("int")).alias("with_death_now"),
        F.sum((has_death & (F.col("n_death_values") > 1)).cast("int"))
        .alias("death_year_changed"),
        F.sum((has_death & (F.col("n_ph_codes") > 1)).cast("int"))
        .alias("ph_status_changed"),
        F.sum((~has_death & F.col("death_added").isNotNull()).cast("int"))
        .alias("death_year_removed"),
        F.sum((has_death & (F.col("death_added") > F.col("created")))
              .cast("int")).alias("death_added_after_creation"),
    ).toPandas()
    entrants = (
        per_author.filter(F.col("death_year") == entering)
        .groupBy(F.year("death_added").alias("year_recorded")).count()
        .toPandas().sort_values("year_recorded").reset_index(drop=True)
    )
    entrants["cum_pct"] = (100 * entrants["count"].cumsum()
                           / entrants["count"].sum()).round(1)
    return {"per_author": per_author, "coverage": coverage,
            "stability": stability, "entrants": entrants}


# ---------------------------------------------------------------------------
# PART 5: COPYRIGHT STATUS ESTIMATES
# ---------------------------------------------------------------------------
#
# The rules below describe the position on January 1, 2027.
#
# United States (17 U.S.C. sections 302-304; publication-based for older
# works). A work published in year Y before 1978 is protected for 95 years,
# ending on December 31 of year Y + 95, so it is public domain from
# January 1 of Y + 96.
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
# the life of the author plus 50 years, counted from January 1 of the year
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
    "enters": "Enters public domain Jan 1, 2027 (published 1931)",
    "renewal": "Renewal unknown (published 1932-1963)",
    "protected_pre78": "Protected (published 1964-1977)",
    "protected_post78": "Protected (published 1978 or later)",
    "unknown": "Unknown (no valid publication year)",
}
PH_LABELS = {
    "pd": "Public domain (last author died 1975 or earlier)",
    "enters": "Enters public domain Jan 1, 2027 (last author died 1976)",
    "protected": "Protected (last author died 1977 or later)",
    "unknown_death": "Unknown (an author's death year is missing)",
    "unknown_author": "Unknown (no author linked)",
}
LIFE70_LABELS = {
    "pd": "Public domain (died 1955 or earlier)",
    "enters": "Enters public domain Jan 1, 2027 (died 1956)",
    "protected": "Protected (died 1957 or later)",
    "unknown": "Unknown",
}


def us_status(first_year, reference_year=REFERENCE_YEAR):
    """US status code from a first-publication year (pure Python).

    Returns one of the US_LABELS keys. The year that enters the public
    domain on January 1 of reference_year is reference_year - 96.
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

    Protection ends on December 31 of death_year + term, so the year that
    enters the public domain on January 1 of reference_year is
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
        """Works and share per label of one status column."""
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
        """Spark expression mapping a status code to a common group."""
        if regime == "us":
            return (
                F.when(F.col(code_col) == "pd", "Public domain already")
                .when(F.col(code_col) == "enters", "Enters Jan 1, 2027")
                .when(F.col(code_col) == "renewal",
                      "Renewal unknown (US only)")
                .when(F.col(code_col).startswith("protected"), "Protected")
                .otherwise("Unknown (missing data)")
            )
        return (
            F.when(F.col(code_col) == "pd", "Public domain already")
            .when(F.col(code_col) == "enters", "Enters Jan 1, 2027")
            .when(F.col(code_col) == "protected", "Protected")
            .otherwise("Unknown (missing data)")
        )

    row = status.select(
        group("us_code", "us").alias("United States (publication + 95)"),
        group("ph_code", "life").alias("Philippines (life + 50)"),
        group("life70_code", "life").alias("Life + 70 (comparison)"),
    )
    order = ["Public domain already", "Enters Jan 1, 2027",
             "Renewal unknown (US only)", "Protected",
             "Unknown (missing data)"]
    tables = []
    for column in row.columns:
        counts = row.groupBy(column).count().toPandas()
        tables.append(counts.set_index(column)["count"].rename(column))
    return (pd.concat(tables, axis=1).reindex(order).fillna(0)
            .astype(int).rename_axis("Status on Jan 1, 2027"))


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


def monthly_want_to_read(reading_log, status, start="2017-10-01",
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
        """Percentage of rows meeting a condition, rounded to 2 places."""
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
            F.round(
                F.sum(F.col("ratings_mean") * F.col("ratings_count"))
                / F.sum("ratings_count"), 2
            ).alias("mean_rating"),
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
        """Spark expression and labels for binning a column by edges."""
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
    "gap_after_death",
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
    """Shared tail of the two list builders: limit, add names, sort.

    n_records (how many duplicate work records a title merges) is shown
    when the input is the consolidated title table.
    """
    display = DISPLAY_COLUMNS + [c for c in ["n_records"]
                                 if c in selected.columns]
    if top_n is None:
        return selected.select(*display)
    top = selected.orderBy(*order).limit(top_n)
    if authors is not None:
        top = add_author_names(top, authors)
    # the join for names does not keep row order, so sort again in pandas
    table = top.toPandas().sort_values(
        ["want_to_read", "wtr_last_12m", "ratings_count", "work_key"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    # whole years read better as integers; Int64 keeps missing values
    for column in ["first_year", "last_death_year"]:
        table[column] = table[column].astype("Int64")
    table["ratings_mean"] = table["ratings_mean"].round(2)
    columns = display[:2] + (["authors"] if authors is not None
                             else []) + display[2:]
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
            ("enters the public domain on Jan 1, 2027", entering),
            (f"high demand (>= {min_want_to_read} Want-to-Read)", demand),
            (f"low supply (<= {max_recent_editions} edition since "
             f"{RECENT_FROM_YEAR})",
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


def normalize_title(title):
    """Pure-Python twin of title_key: a comparable form of a title.

    Lowercases, turns every run of characters that are neither letters
    nor digits into one space, and drops a leading English article, so
    "The Intelligent Investor" and "Intelligent investor." compare equal.
    Letters of any script are kept. Returns "" for an empty title.
    """
    if title is None:
        return ""
    text = re.sub(r"[\W_]+", " ", str(title).lower()).strip()
    return re.sub(r"^(the|a|an) ", "", text)


def title_key(column):
    """Spark expression mirroring normalize_title."""
    from pyspark.sql import functions as F

    text = F.trim(F.regexp_replace(F.lower(column), r"[^\p{L}\p{N}]+", " "))
    return F.coalesce(F.regexp_replace(text, r"^(the|a|an) ", ""), F.lit(""))


def consolidate_titles(profile, reference_year=REFERENCE_YEAR):
    """Merge duplicate work records into one row per title.

    Open Library often holds several work records for one book, each with
    part of its editions and shelf entries, so a single record understates
    both demand and supply. Records are merged when they share the
    normalized title AND exactly the same set of author keys; a record
    with no title or no author stays on its own. Requiring the same
    authors keeps the merge conservative (it cannot join different books
    by different people), at the price of leaving apart duplicates whose
    titles differ, for example by a subtitle.

    Per title: shelf, rating and edition counts are summed; first_year is
    the earliest of the records; the US status and the two list columns
    are recomputed from it; the Philippine status is shared by
    construction (same authors). n_publishers, n_recent_publishers and
    n_languages take the largest record's value, a lower bound, because
    distinct names cannot be recounted from per-record totals. work_key
    and title are those of the record with the most Want-to-Read entries.

    Returns
    -------
    Spark DataFrame
        The profile's list columns, one row per title, plus n_records.
    """
    from pyspark.sql import functions as F

    key = title_key(F.col("title"))
    title_id = F.when(
        (key != "") & (F.size("author_keys") > 0),
        F.concat_ws("|", key, F.array_join(F.array_sort("author_keys"),
                                           ",")),
    ).otherwise(F.col("work_key"))
    summed = ["want_to_read", "currently_reading", "already_read",
              "shelf_total", "wtr_last_12m", "wtr_prev_12m",
              "ratings_count", "n_editions", "n_recent_editions",
              "n_digitized", "n_filipino_editions"]
    largest = ["n_publishers", "n_recent_publishers", "n_languages",
               "latest_edition_year", "last_death_year", "last_birth_year"]
    shared = ["author_keys", "n_authors", "n_authors_with_death",
              "ph_code", "life70_code", "has_philippine_author",
              "author_dates_conflict"]
    rating_sum = F.sum(F.col("ratings_mean") * F.col("ratings_count"))
    titles = (
        profile.withColumn("title_id", title_id).groupBy("title_id").agg(
            F.count(F.lit(1)).alias("n_records"),
            # the most-shelved record represents the title
            F.max(F.struct("want_to_read", "n_editions", "work_key",
                           "title")).alias("_best"),
            F.min("first_year").alias("first_year"),
            F.max(F.col("is_fiction").cast("int")).cast("boolean").alias(
                "is_fiction"),
            *[F.sum(c).alias(c) for c in summed],
            *[F.max(c).alias(c) for c in largest],
            *[F.first(c).alias(c) for c in shared],
            rating_sum.alias("_rating_sum"),
        )
        .withColumn("work_key", F.col("_best.work_key"))
        .withColumn("title", F.col("_best.title"))
        .withColumn("ratings_mean", F.when(
            F.col("ratings_count") > 0,
            F.col("_rating_sum") / F.col("ratings_count")))
        .drop("_best", "_rating_sum")
    )
    # the flags and status of a title follow from its earliest year
    return (
        titles.withColumn("year_before_birth", F.coalesce(
            F.col("first_year") < F.col("last_birth_year") + 10,
            F.lit(False)))
        .withColumn("gap_after_death", F.coalesce(
            F.col("first_year") - F.col("last_death_year") > 70,
            F.lit(False)))
        .withColumn("us_code", us_status_column(F.col("first_year"),
                                                reference_year))
        .withColumn("flagged", F.col("year_before_birth")
                    | F.col("author_dates_conflict"))
        .withColumn("potentially_pd", ~F.col("flagged") & (
            (F.col("us_code") == "pd") | (F.col("ph_code") == "pd")))
        .withColumn("enters_2027", ~F.col("flagged") & (
            (F.col("us_code") == "enters") | (F.col("ph_code") == "enters")))
    )


def consolidation_summary(profile, titles):
    """What merging duplicate records changes (pandas, one row).

    Returns records, titles, the records absorbed into another row, the
    titles built from several records, and the Want-to-Read entries that
    sit on such multi-record titles.
    """
    from pyspark.sql import functions as F

    n_records = profile.count()
    row = titles.agg(
        F.count(F.lit(1)).alias("titles"),
        F.sum((F.col("n_records") > 1).cast("int")).alias(
            "titles_from_several_records"),
        F.max("n_records").alias("largest_title_records"),
        F.sum("want_to_read").alias("wtr"),
        F.sum(F.when(F.col("n_records") > 1, F.col("want_to_read"))
              .otherwise(0)).alias("wtr_on_merged"),
    ).collect()[0]
    return pd.DataFrame([{
        "work_records": n_records, "titles": row["titles"],
        "records_absorbed": n_records - row["titles"],
        "absorbed_pct": round(100 * (n_records - row["titles"])
                              / n_records, 2),
        "titles_from_several_records": row["titles_from_several_records"],
        "largest_title_records": row["largest_title_records"],
        "wtr_on_merged_titles_pct": round(
            100 * row["wtr_on_merged"] / row["wtr"], 1) if row["wtr"]
        else np.nan,
    }])


def first_year_sensitivity(editions, status, min_years=3,
                           reference_year=REFERENCE_YEAR):
    """US status under a more cautious first-year rule (pandas).

    The baseline takes the EARLIEST edition year, so one mis-dated
    edition makes a work look older. The alternative ignores a lone
    earliest date: for a work with at least `min_years` distinct edition
    years it uses the SECOND-earliest one (works with fewer keep the
    earliest; the work's own first_publish_date still counts). The table
    shows, per baseline US group, how many works keep their group and
    how many move, which bounds how much of each group rests on a single
    edition date.

    Returns
    -------
    DataFrame
        One row per baseline US status: works, tested (works with at
        least min_years distinct years), moved, moved_pct (of works).
    """
    from pyspark.sql import functions as F

    years = (
        editions.filter(F.col("work_key").isNotNull()
                        & F.col("publish_year").isNotNull())
        .groupBy("work_key")
        .agg(F.array_sort(F.collect_set("publish_year")).alias("_years"))
        .select("work_key",
                F.size("_years").alias("_n_years"),
                # get() gives null, not an error, for a single year
                F.get("_years", F.lit(1)).alias("_second_year"))
    )
    tested = F.coalesce(F.col("_n_years") >= min_years, F.lit(False))
    compared = (
        status.select("work_key", "work_year", "first_year", "us_code")
        .join(years, "work_key", "left")
        .withColumn("_alt_year", F.when(
            tested, F.least("work_year", "_second_year")
        ).otherwise(F.col("first_year")))
        .withColumn("_alt_code", us_status_column(F.col("_alt_year"),
                                                  reference_year))
        .groupBy("us_code")
        .agg(F.count(F.lit(1)).alias("works"),
             F.sum(tested.cast("int")).alias("tested"),
             F.sum((F.col("_alt_code") != F.col("us_code")).cast("int"))
             .alias("moved"))
        .toPandas()
    )
    compared["moved_pct"] = (100 * compared["moved"]
                             / compared["works"]).round(2)
    compared["US status"] = compared["us_code"].map(US_LABELS)
    compared["order"] = compared["us_code"].map(list(US_LABELS).index)
    return (compared.sort_values("order")
            [["US status", "works", "tested", "moved", "moved_pct"]]
            .reset_index(drop=True))


def born_long_ago_sensitivity(status, author_deaths, born_before=1850,
                              reference_year=REFERENCE_YEAR):
    """Works a "certainly dead" rule would move to Philippine PD (pandas).

    The baseline leaves a work "unknown" when any author lacks a death
    year. The alternative accepts one safe inference: an author born
    before `born_before` with no recorded death was certainly dead before
    the entering year (born 1849 at the latest, such an author would have
    been 127 in 1976; no verified human life exceeds 122 years). A work
    moves to "public domain" when every author either has a death year
    before the entering year or is covered by that inference.

    Returns
    -------
    DataFrame
        One row: works unknown under the baseline, works that would move,
        their share, and how many of the movers are not already public
        domain under the US rule.
    """
    from pyspark.sql import functions as F

    entering = reference_year - 51
    unknown = status.filter(F.col("ph_code") == "unknown_death")
    facts = (
        unknown.select("work_key", "us_code",
                       F.explode("author_keys").alias("author_key"))
        .join(author_deaths.select("author_key", "death_year",
                                   "birth_year"), "author_key", "left")
    )
    inferred = F.col("death_year").isNull() & F.coalesce(
        F.col("birth_year") < born_before, F.lit(False))
    dead_before = F.coalesce(F.col("death_year") < entering, F.lit(False))
    movers = (
        facts.groupBy("work_key", "us_code")
        .agg(F.min((inferred | dead_before).cast("int")).alias("_all"))
        .filter(F.col("_all") == 1)
    )
    row = movers.agg(
        F.count(F.lit(1)).alias("moved"),
        F.sum((F.col("us_code") != "pd").cast("int")).alias("not_us_pd"),
    ).collect()[0]
    n_unknown = unknown.count()
    moved = row["moved"] or 0
    return pd.DataFrame([{
        "works_unknown_death": n_unknown,
        "would_become_ph_public_domain": moved,
        "share_of_unknown_pct": round(100 * moved / n_unknown, 2)
        if n_unknown else np.nan,
        "of_which_not_us_public_domain": row["not_us_pd"] or 0,
    }])


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
        """Append one result row."""
        checks.append({"check": name, "expected": expected,
                       "actual": actual, "passed": expected == actual})

    n_works = tables["works"].count()
    add("profile has one row per work", n_works, profile.count())
    add("profile work keys are unique", n_works,
        profile.select("work_key").distinct().count())

    us_total = status.groupBy("us_code").count().agg(F.sum("count")) \
        .collect()[0][0]
    add("US status groups sum to all works", n_works, int(us_total))

    work_keys = tables["works"].select("work_key")
    # one pass over the raw edition rows of known works gives all three
    # edition recounts (the profile got them from the per-work aggregate)
    editions = tables["editions"].join(work_keys, "work_key", "left_semi")
    recount = editions.agg(
        F.count(F.lit(1)).alias("linked"),
        F.sum(F.col("ocaid").isNotNull().cast("int")).alias("scans"),
        F.sum(F.coalesce(F.col("publish_year") >= RECENT_FROM_YEAR,
                         F.lit(False)).cast("int")).alias("recent"),
    ).collect()[0]
    sums = profile.agg(
        F.sum("n_editions").alias("n_editions"),
        F.sum("want_to_read").alias("want_to_read"),
        F.sum("n_digitized").alias("n_digitized"),
        F.sum("n_recent_editions").alias("n_recent_editions"),
    ).collect()[0]
    add("edition rows linked to known works = sum of n_editions",
        recount["linked"], int(sums["n_editions"]))

    wtr_linked = tables["reading_log"].filter(
        F.col("shelf") == "Want to Read"
    ).join(work_keys, "work_key", "left_semi").count()
    add("Want-to-Read rows on known works = sum of want_to_read",
        wtr_linked, int(sums["want_to_read"]))
    add("digitized editions on known works = sum of n_digitized",
        recount["scans"], int(sums["n_digitized"]))
    add("recent editions on known works = sum of n_recent_editions",
        recount["recent"], int(sums["n_recent_editions"]))

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


# Hand-checked facts for 30 well-known, high-demand works: 15 candidates
# for the 2027 watchlist (W) and 15 for the potentially-public-domain list
# (P). Each value was checked by hand against publisher, library and
# encyclopaedia records; None means "not checked" (the listing does not
# depend on it, or no reliable source was found). The facts do not depend
# on the dump, so the audit applies to any snapshot; works absent from a
# snapshot are skipped. This is a check of high-demand works a reader
# would act on first, not a random sample of the catalogue.
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
     "Ancient text (c. 5th century BC); only modern editions are "
     "catalogued"),
    ("P", "/works/OL3923952W", 1948, 1948, "Dazai d. 1948"),
    ("P", "/works/OL27482W", 1937, 1973, "Tolkien d. 1973"),
    ("P", "/works/OL10263W", 1943, 1944, "Saint-Exupery d. 1944"),
    ("P", "/works/OL1168007W", 1945, 1950, "Orwell d. 1950"),
    ("P", "/works/OL498556W", 1915, 1924, "Kafka d. 1924"),
    ("P", "/works/OL8165007W", 1926, 1957, "Clason d. 1957"),
    ("P", "/works/OL21177W", 1847, 1848,
     "Published 1847; E. Bronte d. 1848"),
    ("P", "/works/OL166894W", 1866, 1881, "Dostoevsky d. 1881"),
    ("P", "/works/OL1230613W", 1942, 1960, "Camus d. 1960"),
    ("P", "/works/OL262421W", 1892, 1930, "Conan Doyle d. 1930"),
    ("P", "/works/OL64365W", 1932, 1963, "Huxley d. 1963"),
]


# Hand check of every title on the 2027 watchlist of the 20221007
# snapshot: first publication and the death year of every author, looked
# up in encyclopaedia and library records. Verdicts:
#   confirmed       the title enters the public domain as estimated
#   already free    it is free, but earlier than estimated
#   needs care      the record is not one single book
#   false positive  it does not enter the public domain in 2027
WATCHLIST_REVIEW = {
    "/works/OL273187W": (
        "confirmed", "Graham d. 1976; use the 1949 or 1973 text, not the "
        "2003 edition with new commentary"),
    "/works/OL267968W": (
        "confirmed", "1931 first edition only; every later revision "
        "stays protected"),
    "/works/OL6563302W": (
        "confirmed", "McGraw-Hill 1931 first edition only"),
    "/works/OL1127268W": ("confirmed", "Little, Brown 1931"),
    "/works/OL472833W": (
        "confirmed", "Christie d. 1976; collection of her plays"),
    "/works/OL471559W": (
        "confirmed", "Christie d. 1976; published 1977, and the term for "
        "a posthumous work is the same"),
    "/works/OL472688W": (
        "confirmed", "Christie d. 1976, writing as Mary Westmacott"),
    "/works/OL4775275W": (
        "confirmed", "Boyer d. 1976; first published 1939 as The "
        "Concepts of the Calculus"),
    "/works/OL2906425W": ("confirmed", "Dennis d. 1976; published 1955"),
    "/works/OL472813W": (
        "needs care", "Christie d. 1976, but the record mixes several "
        "omnibus volumes"),
    "/works/OL273188W": (
        "false positive", "co-author David Dodd (d. 1988) is missing "
        "from the record; protected in the Philippines until 2039"),
    "/works/OL3408913W": ("confirmed", "Getty d. 1976; published 1965"),
    "/works/OL5367088W": ("confirmed", "Little, Brown 1931"),
    "/works/OL15148479W": (
        "confirmed", "Kinross d. 1976; published 1977"),
    "/works/OL4134118W": ("confirmed", "Gallico d. 1976; published 1958"),
    "/works/OL2139804W": (
        "already free", "first published 1916 and Saussure d. 1913; the "
        "earliest catalogued edition is 1931"),
    "/works/OL1132800W": (
        "false positive", "Bieber d. 1978, not 1976 as catalogued; "
        "protected in the Philippines until 2029"),
}


def review_watchlist(watchlist, review=None):
    """Attach the hand-check verdict and note to each watchlist row.

    Rows without an entry in `review` (WATCHLIST_REVIEW by default) are
    marked "not checked", so a list built from another snapshot shows at
    once which of its titles still need a look.
    """
    review = WATCHLIST_REVIEW if review is None else review
    table = watchlist.copy()
    checks = table["work_key"].map(
        lambda key: review.get(key, ("not checked", "")))
    table["check"] = [verdict for verdict, _ in checks]
    table["note"] = [note for _, note in checks]
    return table


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
    fig.suptitle("Estimated rights status on January 1, 2027")
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
                 f"(entries still on a shelf on {DUMP_DATE:%b %-d, %Y})")
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


def plot_pipeline_diagram(facts=None):
    """Draw the analysis pipeline as a left-to-right flow diagram.

    `facts` fills in the run's own numbers: raw_gb, the size of the raw
    snapshot in GiB (shown as "?" when missing).

    Each card is one stage: a coloured title band (colours follow the
    report's colour system), the main operations, and the helper function
    that implements it, so the figure doubles as a map from the
    methodology to the code. Lanes underneath group the stages into the
    three phases described in Section 3. Returns a matplotlib Figure.
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
        """Draw one pipeline stage as a card with a coloured title band."""
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
        """Draw an arrow between two points."""
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.3, color=color, zorder=1,
            connectionstyle="arc3,rad=0"))

    def lane(x1, x2, label, color):
        """Draw a phase lane with its label under the cards."""
        ax.plot([x1, x2], [-0.3, -0.3], color=color, lw=3,
                solid_capstyle="round")
        ax.text((x1 + x2) / 2, -0.62, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")

    facts = {"raw_gb": "?", **(facts or {})}
    card(0.1, 1.9, 2.15, 2.6, 1, "Raw snapshot",
         [f"{DATA_SNAPSHOT} on Jojie,", f"{facts['raw_gb']} GiB, 7 .gz files",
          "5 record dumps +", "all-types dump +", "revision history"],
         "raw_input_size", GREY)
    card(2.6, 1.9, 2.15, 2.6, 2, "Parse",
         ["gunzip -> splittable text", "PySpark, explicit schemas",
          "free-text dates -> years", "history files: 3 jobs"],
         "build_parquet_tables", INK)
    card(5.1, 1.9, 2.0, 2.6, 3, "Parquet",
         ["8 clean tables", "rows = raw lines", "coverage, nulls",
          "merged keys resolved"],
         "ingestion_check", INK)
    card(7.55, 4.35, 2.55, 1.95, "4a", "Rights status",
         ["death years + their history", "US, PH and life+70 rules",
          "data-quality flags"], "estimate_copyright_status", PLUM)
    card(7.55, 2.2, 2.55, 1.95, "4b", "Demand",
         ["Want-to-Read, shelves", "ratings, 12-month trends"],
         "compute_demand", GREEN)
    card(7.55, 0.05, 2.55, 1.95, "4c", "Supply",
         ["recent editions, publishers", "scans, Filipino editions"],
         "compute_supply", INK)
    card(10.55, 2.2, 2.85, 2.3, 5, "Profile & lists",
         ["one row per work",
          "duplicate titles merged",
          "2027 watchlist", "public-domain list"],
         "build_watchlist", AMBER)
    card(10.55, 0.05, 2.85, 1.85, 6, "Validation",
         ["independent recounts", "sensitivity checks",
          "hand-checked audit"],
         "audit_estimates", GREY)

    arrow(2.25, 3.2, 2.6, 3.2)
    arrow(4.75, 3.2, 5.1, 3.2)
    for y, color in ((5.3, PLUM), (3.18, GREEN), (1.03, INK)):
        arrow(7.1, 3.2, 7.55, y, color)
        arrow(10.1, y, 10.55, 3.35, color)
    arrow(11.98, 2.2, 11.98, 1.9, AMBER)

    lane(0.1, 7.1, "INGEST & EXPLORE ONCE  (Sections 4.2-4.5)", INK)
    lane(7.55, 10.1, "ANALYSE EVERY WORK  (4.6-4.8)", PLUM)
    lane(10.55, 13.4, "SELECT & VERIFY  (4.9-4.10)", AMBER)
    fig.tight_layout()
    return fig


def plot_death_history(coverage, entrants, activity):
    """Three panels on how the catalogue's author data grew.

    Left: edits and new records per year (catalogue activity). Middle:
    share of authors with a death year over time. Right: cumulative
    share of the 1976 deaths (the Philippine 2027 entrants) whose death
    year had been recorded by each year.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for column, color, label in [
        ("edits_editions", INK, "edition edits"),
        ("created_editions", INK_LIGHT, "editions created"),
        ("edits_authors", PLUM, "author edits"),
        ("edits_works", GREY, "work edits"),
    ]:
        if column in activity:
            ax.plot(activity.index, activity[column] / 1e6, color=color,
                    lw=1.6, label=label)
    ax.set_title("Catalogue activity per year")
    ax.set_ylabel("Revisions (millions)")
    ax.legend(fontsize=8)
    _style(ax)
    ax = axes[1]
    data = coverage[coverage["year"] >= 2008]
    ax.plot(data["year"], data["pct_with_death"], color=PLUM, lw=2)
    ax.set_title("Authors with a death year")
    ax.set_ylabel("Share of authors on record (%)")
    _style(ax)
    ax = axes[2]
    ax.step(entrants["year_recorded"], entrants["cum_pct"], where="post",
            color=AMBER, lw=2)
    ax.set_title("When the 1976 deaths were recorded")
    ax.set_ylabel("Cumulative share of 1976 deaths (%)")
    ax.set_ylim(0, 102)
    _style(ax)
    for ax in axes:
        ax.set_xlabel("Year")
    fig.tight_layout()
    return fig
