"""
Process the optional REPLAY Foursquare / Gowalla datasets for k1-poi.

Run from the k1-poi project root:

    python data/data_process_4_replay_4sq_gwl.py

Default input:
    data/raw/replay-4sq-gwl/data

Default output:
    data/processed

The raw REPLAY files do not include POI category fields. To keep the processed
pickle schema compatible with the main k1-poi data pipeline, this script fills
cat_id="1" and cat_name="1".
"""

import argparse
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path


SECONDS_PER_DAY = 24 * 60 * 60
CAT_ID = "1"
CAT_NAME = "1"


def parse_utc_time(utc_str: str) -> datetime:
    value = str(utc_str).strip()
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_records(checkins_path: Path, offset_path: Path):
    records = []
    with checkins_path.open("r", encoding="utf-8", errors="replace") as f_ck, offset_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as f_offset:
        for line_no, (checkin_line, offset_line) in enumerate(
            zip_longest(f_ck, f_offset), start=1
        ):
            if checkin_line is None or offset_line is None:
                raise ValueError(
                    f"Line count mismatch between {checkins_path} and {offset_path} "
                    f"(first mismatch at line {line_no})"
                )

            checkin_line = checkin_line.strip()
            offset_line = offset_line.strip()
            if not checkin_line:
                continue

            parts = checkin_line.split("\t")
            if len(parts) < 5:
                continue

            try:
                user = str(parts[0])
                dt = parse_utc_time(parts[1])
                lat = float(parts[2])
                lon = float(parts[3])
                poi = str(parts[4])
                timezone_offset = int(offset_line)
            except Exception:
                continue

            seconds_since_midnight = (
                dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
            )
            tod = seconds_since_midnight / SECONDS_PER_DAY
            utc_time = int(dt.timestamp())

            records.append(
                (user, poi, CAT_ID, CAT_NAME, lat, lon, timezone_offset, tod, utc_time)
            )

    return records


def filter_users_with_single_checkin(records):
    user_counts = Counter(record[0] for record in records)
    return [record for record in records if user_counts[record[0]] >= 2]


def add_delta_days(records):
    last_time_per_user = {}
    processed = []
    for user, poi, cat_id, cat_name, lat, lon, timezone_offset, tod, utc_time in records:
        previous_utc_time = last_time_per_user.get(user)
        if previous_utc_time is None:
            delta_days = -1.0
        else:
            delta_days = (utc_time - previous_utc_time) / SECONDS_PER_DAY
        last_time_per_user[user] = utc_time

        processed.append(
            (
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


def split_train_val_test(records, train_ratio=0.8, val_ratio=0.1):
    n_records = len(records)
    n_train = int(n_records * train_ratio)
    n_val = int(n_records * (train_ratio + val_ratio))
    return records[:n_train], records[n_train:n_val], records[n_val:]


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


def build_including_and_excluding(records):
    train, val, test = split_train_val_test(records)

    including = {
        "train": build_user_sequences(train),
        "val": build_user_sequences(val),
        "test": build_user_sequences(test),
    }

    train_users = {record[0] for record in train}
    train_pois = {record[1] for record in train}

    val_filtered = [record for record in val if record[0] in train_users and record[1] in train_pois]
    test_filtered = [
        record for record in test if record[0] in train_users and record[1] in train_pois
    ]

    excluding = {
        "train": including["train"],
        "val": build_user_sequences(val_filtered),
        "test": build_user_sequences(test_filtered),
    }

    return including, excluding


def process_one_dataset(tag: str, input_dir: Path, output_dir: Path, use_sample: bool):
    if tag not in {"gowalla", "4sq"}:
        raise ValueError(f"Unknown tag: {tag} (expected 'gowalla' or '4sq')")

    checkins_name = f"checkins-{tag}.txt"
    offset_name = f"checkins_{tag}_time_offset.txt"
    if use_sample:
        checkins_name += ".sample"
        offset_name += ".sample"

    checkins_path = input_dir / checkins_name
    offset_path = input_dir / offset_name
    if not checkins_path.exists():
        raise FileNotFoundError(checkins_path)
    if not offset_path.exists():
        raise FileNotFoundError(offset_path)

    print(f"Processing {tag}:")
    print(f"  checkins: {checkins_path}")
    print(f"  offset:   {offset_path}")

    records = load_records(checkins_path=checkins_path, offset_path=offset_path)
    records = filter_users_with_single_checkin(records)
    records.sort(key=lambda record: record[-1])
    records = add_delta_days(records)

    including, excluding = build_including_and_excluding(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    including_path = output_dir / f"replay_{tag}_including_cold.pkl"
    excluding_path = output_dir / f"replay_{tag}_excluding_cold.pkl"

    with including_path.open("wb") as f:
        pickle.dump(including, f, protocol=pickle.HIGHEST_PROTOCOL)
    with excluding_path.open("wb") as f:
        pickle.dump(excluding, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved including-cold to {including_path}")
    print(f"Saved excluding-cold to {excluding_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process optional REPLAY Foursquare/Gowalla data for k1-poi."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw/replay-4sq-gwl/data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="*", default=["gowalla", "4sq"])
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Use *.sample files for quick verification.",
    )
    args = parser.parse_args()

    if Path.cwd().name != "k1-poi":
        print("Warning: this script is designed to run with k1-poi as the working directory.")

    for tag in args.datasets:
        process_one_dataset(
            tag=tag,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            use_sample=args.use_sample,
        )


if __name__ == "__main__":
    main()
