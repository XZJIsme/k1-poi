"""
Process the raw GETNext-format datasets shipped with k1-poi.

Run from the k1-poi project root:

    python data/data_process.py

Default inputs:
    data/raw/Gowalla-CA
    data/raw/NYC
    data/raw/TKY

Default output:
    data/processed
"""

import argparse
import csv
import os
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SECONDS_PER_DAY = 24 * 60 * 60
CAT_NAME_RE = re.compile(r"'name'\s*:\s*'([^']+)'")


def parse_utc_time(utc_str: str) -> datetime:
    value = str(utc_str).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def parse_gowalla_cat_name(raw: str) -> str:
    value = "" if raw is None else str(raw)
    match = CAT_NAME_RE.search(value)
    if match:
        return match.group(1).strip()
    return value


def load_foursquare_records(csv_path: Path, split: str):
    records = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            user = row.get("user_id")
            poi = row.get("POI_id")
            cat_id = row.get("POI_catid")
            cat_name = row.get("POI_catname")
            if user is None or poi is None or cat_id is None or cat_name is None:
                continue

            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                timezone_offset = int(row["timezone"])
                dt = parse_utc_time(row["UTC_time"])
            except (KeyError, TypeError, ValueError):
                continue

            records.append(
                (
                    split,
                    str(user),
                    str(poi),
                    str(cat_id),
                    str(cat_name),
                    lat,
                    lon,
                    timezone_offset,
                    dt,
                )
            )
    return records


def load_gowalla_records(csv_path: Path, split: str):
    records = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            user = row.get("user_id")
            poi = row.get("POI_id")
            if user is None or poi is None:
                continue

            cat_id = row.get("POI_catid_code") or "0"
            cat_name = parse_gowalla_cat_name(row.get("POI_catname"))

            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                dt = parse_utc_time(row["checkin_time"])
            except (KeyError, TypeError, ValueError):
                continue

            records.append(
                (
                    split,
                    str(user),
                    str(poi),
                    str(cat_id),
                    str(cat_name),
                    lat,
                    lon,
                    0,
                    dt,
                )
            )
    return records


def filter_users_with_single_checkin(records):
    user_counts = Counter(record[1] for record in records)
    return [record for record in records if user_counts[record[1]] >= 2]


def add_time_features(records):
    last_time_per_user = {}
    processed = []

    for split, user, poi, cat_id, cat_name, lat, lon, timezone_offset, dt in records:
        seconds_since_midnight = (
            dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        )
        tod = seconds_since_midnight / SECONDS_PER_DAY

        if user in last_time_per_user:
            delta_days = (dt - last_time_per_user[user]).total_seconds() / SECONDS_PER_DAY
        else:
            delta_days = -1.0
        last_time_per_user[user] = dt

        utc_time = int(dt.timestamp())
        processed.append(
            (
                split,
                user,
                poi,
                cat_id,
                cat_name,
                lat,
                lon,
                timezone_offset,
                tod,
                delta_days,
                float(utc_time),
                utc_time,
            )
        )

    return processed


def build_user_sequences(records):
    user_seqs = defaultdict(list)
    for (
        user,
        poi,
        cat_id,
        cat_name,
        lat,
        lon,
        timezone_offset,
        tod,
        delta_days,
        timestamp,
        utc_time,
    ) in records:
        user_seqs[user].append(
            {
                "poi_id": poi,
                "cat_id": cat_id,
                "cat_name": cat_name,
                "lat": lat,
                "lon": lon,
                "timezone_offset": timezone_offset,
                "tod": tod,
                "delta_days": delta_days,
                "timestamp": timestamp,
                "utc_time": utc_time,
            }
        )
    return dict(user_seqs)


def build_including_and_excluding(train, val, test):
    including = {
        "train": build_user_sequences(train),
        "val": build_user_sequences(val),
        "test": build_user_sequences(test),
    }

    train_users = {record[0] for record in train}
    train_pois = {record[1] for record in train}

    val_filtered = [record for record in val if record[0] in train_users and record[1] in train_pois]
    test_filtered = [record for record in test if record[0] in train_users and record[1] in train_pois]

    excluding = {
        "train": including["train"],
        "val": build_user_sequences(val_filtered),
        "test": build_user_sequences(test_filtered),
    }

    return including, excluding


def save_pickle(data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output_path}")


def process_dataset(dataset_tag: str, input_dir: Path, output_dir: Path, loader, file_prefix: str):
    split_paths = {
        "train": input_dir / f"{file_prefix}_train.csv",
        "val": input_dir / f"{file_prefix}_val.csv",
        "test": input_dir / f"{file_prefix}_test.csv",
    }
    for path in split_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    records = []
    for split, path in split_paths.items():
        records.extend(loader(path, split))

    records = filter_users_with_single_checkin(records)
    records.sort(key=lambda record: record[-1])
    records = add_time_features(records)

    train = [record[1:] for record in records if record[0] == "train"]
    val = [record[1:] for record in records if record[0] == "val"]
    test = [record[1:] for record in records if record[0] == "test"]

    including, excluding = build_including_and_excluding(train, val, test)

    save_pickle(including, output_dir / f"{dataset_tag}_including_cold.pkl")
    save_pickle(excluding, output_dir / f"{dataset_tag}_excluding_cold.pkl")


def process_all(raw_dir: Path, output_dir: Path):
    jobs = [
        ("CA", raw_dir / "Gowalla-CA", load_gowalla_records, "gowalla"),
        ("NYC", raw_dir / "NYC", load_foursquare_records, "NYC"),
        ("TKY", raw_dir / "TKY", load_foursquare_records, "TKY"),
    ]
    for dataset_tag, input_dir, loader, file_prefix in jobs:
        print(f"Processing {dataset_tag} from {input_dir}")
        process_dataset(dataset_tag, input_dir, output_dir, loader, file_prefix)


def main():
    parser = argparse.ArgumentParser(
        description="Process k1-poi raw datasets into train/val/test pickle files."
    )
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if Path.cwd().name != "k1-poi":
        print("Warning: this script is designed to run with k1-poi as the working directory.")

    process_all(raw_dir=args.raw_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
