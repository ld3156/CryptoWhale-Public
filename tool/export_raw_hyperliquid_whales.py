from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import boto3
import pandas as pd

try:
    import awswrangler as wr
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: awswrangler. Install with "
        "`pip install awswrangler boto3 pandas pyarrow`."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"

DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "raw_whales"
DEFAULT_BUCKET = os.getenv("HYPERLIQUID_S3_BUCKET", "")
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

EXPORT_FORMAT_ALIASES = {
    "parquet": "parquet",
    "p": "parquet",
    ".parquet": "parquet",
    "1": "parquet",
    "csv": "csv",
    "c": "csv",
    ".csv": "csv",
    "2": "csv",
}


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing environment variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


load_env_file(ENV_FILE)


def configure_aws_session() -> None:
    """Configure boto3 from env vars if provided; otherwise use default AWS credential chain."""
    session_kwargs = {"region_name": DEFAULT_REGION}

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key

    if session_token:
        session_kwargs["aws_session_token"] = session_token

    boto3.setup_default_session(**session_kwargs)


def parse_time_like(series: pd.Series) -> pd.Series:
    raw = pd.Series(series)
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().any():
        median_value = numeric.dropna().median()
        if median_value > 1e12:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if median_value > 1e9:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")

    return pd.to_datetime(raw, utc=True, errors="coerce")


def parse_user_timestamp(text: str) -> pd.Timestamp:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty timestamp input.")

    timestamp = pd.Timestamp(cleaned)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def prompt_timestamp(label: str) -> pd.Timestamp:
    while True:
        raw = input(
            f"{label} (UTC, example 2026-03-01 00:00:00 or 2026-03-01T00:00:00Z): "
        )
        try:
            return parse_user_timestamp(raw)
        except Exception as exc:
            print(f"Invalid timestamp: {exc}")


def normalize_export_format(raw: str | None, default: str | None = None) -> str:
    if raw is None or not raw.strip():
        if default is not None:
            return default
        raise ValueError("Export format is required.")

    normalized = EXPORT_FORMAT_ALIASES.get(raw.strip().lower())
    if normalized is None:
        raise ValueError(
            "Invalid export format. Use `parquet` or `csv` "
            "(shorthands: `1`/`p` and `2`/`c`)."
        )
    return normalized


def prompt_export_format() -> str:
    while True:
        raw = input("Export format [1=parquet, 2=csv] (default parquet): ").strip()
        try:
            return normalize_export_format(raw, default="parquet")
        except ValueError as exc:
            print(exc)


def build_partition_filter(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Callable[[dict[str, str]], bool]:
    partition_hours = pd.date_range(
        start=start_ts.floor("h"),
        end=end_ts.floor("h"),
        freq="h",
        tz="UTC",
    )
    allowed = {
        (hour.strftime("%Y-%m-%d"), hour.strftime("%H"))
        for hour in partition_hours
    }

    def partition_filter(partition: dict[str, str]) -> bool:
        dt_value = partition.get("dt")
        hh_value = partition.get("hh")

        if dt_value is None:
            return True
        if hh_value is None:
            return any(str(dt_value) == allowed_dt for allowed_dt, _ in allowed)

        return (str(dt_value), str(hh_value).zfill(2)) in allowed

    return partition_filter


def prepare_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    for col in ("ingested_at", "create_time", "update_time"):
        if col in frame.columns:
            frame[col] = parse_time_like(frame[col])

    if "ingested_at" in frame.columns:
        ordered_cols = ["ingested_at"] + [
            col for col in frame.columns if col != "ingested_at"
        ]
        return frame[ordered_cols].sort_values("ingested_at").reset_index(drop=True)

    return frame.reset_index(drop=True)


def coalesce_timestamps(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    merged = pd.Series(index=frame.index, dtype="datetime64[ns, UTC]")
    for col in candidates:
        if col in frame.columns:
            merged = merged.fillna(frame[col])
    return merged


def load_windowed_dataset(
    bucket: str,
    dataset_name: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    if not bucket:
        raise ValueError(
            "S3 bucket is required. Pass `--bucket ...` or set HYPERLIQUID_S3_BUCKET."
        )

    s3_path = f"s3://{bucket}/{dataset_name}/"
    print(f"\nFetching {dataset_name} from {s3_path}")

    frame = wr.s3.read_parquet(
        path=s3_path,
        dataset=True,
        partition_filter=build_partition_filter(start_ts, end_ts),
    )
    frame = prepare_loaded_frame(frame)

    if dataset_name == "whale_alerts":
        filter_ts = coalesce_timestamps(frame, ["create_time", "ingested_at"])
    else:
        filter_ts = coalesce_timestamps(
            frame,
            ["update_time", "create_time", "ingested_at"],
        )

    mask = filter_ts.between(start_ts, end_ts, inclusive="both")
    filtered = frame.loc[mask].copy()
    filtered["_sort_ts"] = filter_ts.loc[mask]
    filtered = (
        filtered.sort_values("_sort_ts")
        .drop(columns="_sort_ts")
        .reset_index(drop=True)
    )

    print(f"Loaded {len(frame):,} rows from matching partitions.")
    print(f"Filtered to {len(filtered):,} rows inside {start_ts} -> {end_ts} UTC.")
    return filtered


def export_frame(frame: pd.DataFrame, output_path: Path, export_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "csv":
        frame.to_csv(output_path, index=False)
        return

    frame.to_parquet(output_path, index=False)


def build_export_dir(base_dir: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Path:
    label = (
        f"raw_whales_{start_ts.strftime('%Y%m%dT%H%M%SZ')}"
        f"_to_{end_ts.strftime('%Y%m%dT%H%M%SZ')}"
    )
    return base_dir / label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw Hyperliquid whale alerts and whale positions from S3."
    )
    parser.add_argument(
        "--from-ts",
        dest="from_ts",
        help="Start timestamp. Naive values are treated as UTC.",
    )
    parser.add_argument(
        "--to-ts",
        dest="to_ts",
        help="End timestamp. Naive values are treated as UTC.",
    )
    parser.add_argument(
        "--format",
        metavar="FORMAT",
        help="Export file format: parquet or csv. If omitted, the script prompts interactively.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="S3 bucket name. Default: value of HYPERLIQUID_S3_BUCKET.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_ts = parse_user_timestamp(args.from_ts) if args.from_ts else prompt_timestamp("From")
    end_ts = parse_user_timestamp(args.to_ts) if args.to_ts else prompt_timestamp("To")

    if start_ts > end_ts:
        raise SystemExit("`from` must be earlier than or equal to `to`.")

    try:
        export_format = (
            normalize_export_format(args.format)
            if args.format
            else prompt_export_format()
        )
        output_dir = build_export_dir(Path(args.outdir), start_ts, end_ts)

        configure_aws_session()

        alerts = load_windowed_dataset(args.bucket, "whale_alerts", start_ts, end_ts)
        positions = load_windowed_dataset(args.bucket, "whale_positions", start_ts, end_ts)

        alerts_path = output_dir / f"whale_alerts_raw.{export_format}"
        positions_path = output_dir / f"whale_positions_raw.{export_format}"

        export_frame(alerts, alerts_path, export_format)
        export_frame(positions, positions_path, export_format)

    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print("\nExport complete.")
    print(f"format -> {export_format}")
    print(f"whale_alerts  -> {alerts_path}")
    print(f"whale_positions -> {positions_path}")


if __name__ == "__main__":
    main()
