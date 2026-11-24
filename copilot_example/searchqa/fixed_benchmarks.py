"""Fixed benchmark manifests for the Lite training, validation and test phases."""
import hashlib
import json
import os
from pathlib import Path


SPLITS = {
    "searchqa": {
        "train": (400, "cded018312f19f27e1da3e019931fb4d18fbbdbc4fd6d1651b5a52f29dedd268"),
        "val": (200, "d2913aff1ba10bd9a577e79c24121108218b30136c0d5a7002eed36133babafe"),
        "test": (1400, "7fca98256e96c837eee2612da9b9d29fb9d4e5cb908c751ca9bfdfd2d10ab6fb"),
    },
    "triviaqa": {
        "train": (1600, "f465cbaea5820ac4a01a9d0f5d6dc6b71668a78bf21cf783c687e3f181ac3445"),
        "val": (200, "c2b8957c0814fc54e381821f2efb6af200bae1e26605304ca3bac7c10747176c"),
        "test": (1000, "34c26854b2b9348c90fb740675bf45562937c1525f5a770824474e0b6df6397c"),
    },
    "hotpotqa": {
        "train": (1600, "925adec60c7d1488b925b5c5facaaeb9b0f2d60b155928db809fde778433a6d9"),
        "val": (200, "9022dabfa5ddd7a7df3a32eee1a8abbab8963ea807013f257dbefb1250d3d82b"),
        "test": (1000, "7b81664c3206ab4a9d4e3bd274a7bbe7b7d09976c74918970a02ecbb463b7439"),
    },
    "bird": {
        "train": (834, "bf49ab424f7627116e654bb8719fdee322031a35da199aa04b6c370ae0c281b3"),
        "val": (200, "2b3ab585858e168cb05ab1cf2ce8ceb4ecd24a4ae2d375d89ba9d94a362b19d1"),
        "test": (500, "706fcd80ba2577502ed013f49f22ceac5071af0dbfe58adb6fe22ee030b2487f"),
    },
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_ROOT = ROOT.parent / "bird-text-to-sql-260820/data_dev/dev_databases"


def load_splits(split_dir=None, benchmark="searchqa"):
    data = Path(split_dir or os.getenv(f"{benchmark.upper()}_SPLIT_DIR")
                or ROOT.parent / "SkillOpt" / "data" / f"{benchmark}_split")
    splits = {}
    for split, (size, fingerprint) in SPLITS[benchmark].items():
        path = data / split / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        ids = [r["id"] for r in rows]
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        if len(rows) != size or digest != fingerprint:
            raise ValueError(f"{path} must contain the {size} fixed {benchmark} IDs in manifest order")
        fields = ("question", "db_id", "prompt", "SQL") if benchmark == "bird" else ("question", "context")
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in fields):
                raise ValueError(f"Missing {benchmark} task fields for {split}/{row['id']}")
            if benchmark == "bird":
                db = row["db_id"]
                if Path(db).name != db or db in (".", ".."):
                    raise ValueError(f"Invalid database ID for {split}/{row['id']}")
            else:
                golds = row.get("answers")
                if not isinstance(golds, list) or not golds or not all(isinstance(g, str) for g in golds):
                    raise ValueError(f"Missing reference answers for {split}/{row['id']}")
        splits[split] = rows
    return splits


def check_databases(splits, root):
    for db in sorted({r["db_id"] for rows in splits.values() for r in rows}):
        path = root / db / (db + ".sqlite")
        if not path.is_file():
            raise FileNotFoundError(f"BIRD database missing: {path}")
