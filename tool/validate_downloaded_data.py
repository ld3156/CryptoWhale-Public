import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORT_JSON = DATA_DIR / "_validation_report.json"
REPORT_CSV = DATA_DIR / "_validation_report.csv"

INTERVAL_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


@dataclass
class FileCheckResult:
    file: str
    interval: str
    row_count: int
    has_time_ms_column: bool
    valid_time_ms_rows: int
    invalid_time_ms_rows: int
    duplicate_time_ms_count: int
    not_sorted_count: int
    missing_points_count: int
    max_gap_ms: int
    coverage_percent: float
    gap_locations: str
    status: str
    notes: str


def infer_interval_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    for interval in INTERVAL_TO_MS:
        if stem.endswith(f"_{interval}"):
            return interval
    return "unknown"


def ms_to_utc_text(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def validate_one_file(path: Path) -> FileCheckResult:
    interval = infer_interval_from_filename(path.name)
    expected_step = INTERVAL_TO_MS.get(interval)

    has_time_ms = False
    total_rows = 0
    valid_ts: List[int] = []
    invalid_time_rows = 0
    notes: List[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        has_time_ms = "time_ms" in headers

        if not has_time_ms:
            return FileCheckResult(
                file=path.name,
                interval=interval,
                row_count=0,
                has_time_ms_column=False,
                valid_time_ms_rows=0,
                invalid_time_ms_rows=0,
                duplicate_time_ms_count=0,
                not_sorted_count=0,
                missing_points_count=0,
                max_gap_ms=0,
                coverage_percent=0.0,
                gap_locations="",
                status="FAIL",
                notes="missing 'time_ms' column",
            )

        for row in reader:
            total_rows += 1
            t = (row.get("time_ms") or "").strip()
            if not t:
                invalid_time_rows += 1
                continue
            if not t.isdigit():
                invalid_time_rows += 1
                continue
            valid_ts.append(int(t))

    duplicate_count = len(valid_ts) - len(set(valid_ts))

    # Sorting / monotonic check
    not_sorted_count = 0
    for prev, cur in zip(valid_ts, valid_ts[1:]):
        if cur < prev:
            not_sorted_count += 1

    # Gap check (on sorted unique timestamps)
    missing_points_count = 0
    max_gap_ms = 0
    gap_segments: List[str] = []
    coverage_percent = 0.0
    if expected_step and len(valid_ts) >= 2:
        sorted_unique = sorted(set(valid_ts))
        expected_points = ((sorted_unique[-1] - sorted_unique[0]) // expected_step) + 1
        actual_points = len(sorted_unique)
        coverage_percent = 100.0 if expected_points <= 0 else (actual_points / expected_points) * 100.0

        for prev, cur in zip(sorted_unique, sorted_unique[1:]):
            gap = cur - prev
            if gap > max_gap_ms:
                max_gap_ms = gap
            if gap > expected_step:
                # Count how many expected points are missing in this gap
                missing_n = (gap // expected_step) - 1
                missing_points_count += missing_n
                gap_start = prev + expected_step
                gap_end = cur - expected_step
                gap_segments.append(
                    f"{gap_start}->{gap_end} UTC({ms_to_utc_text(gap_start)} ~ {ms_to_utc_text(gap_end)}), missing={missing_n}"
                )
    elif interval == "unknown":
        notes.append("interval not inferred from filename, skipped gap check")
    elif len(valid_ts) == 1:
        coverage_percent = 100.0

    gap_locations = ""
    if gap_segments:
        max_show = 20
        visible = gap_segments[:max_show]
        hidden = len(gap_segments) - len(visible)
        gap_locations = " | ".join(visible)
        if hidden > 0:
            gap_locations += f" | ... and {hidden} more gap segments"

    if total_rows == 0:
        status = "WARN"
        notes.append("empty file")
    elif invalid_time_rows > 0 or duplicate_count > 0 or not_sorted_count > 0:
        status = "WARN"
    else:
        status = "OK"

    if missing_points_count > 0:
        status = "WARN"
        notes.append("time gaps detected")

    return FileCheckResult(
        file=path.name,
        interval=interval,
        row_count=total_rows,
        has_time_ms_column=True,
        valid_time_ms_rows=len(valid_ts),
        invalid_time_ms_rows=invalid_time_rows,
        duplicate_time_ms_count=duplicate_count,
        not_sorted_count=not_sorted_count,
        missing_points_count=missing_points_count,
        max_gap_ms=max_gap_ms,
        coverage_percent=round(coverage_percent, 4),
        gap_locations=gap_locations,
        status=status,
        notes="; ".join(notes),
    )


def collect_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    out = []
    for p in sorted(data_dir.glob("*.csv")):
        # Skip generated report files
        if p.name.startswith("_validation_report"):
            continue
        out.append(p)
    return out


def write_reports(results: List[FileCheckResult]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # JSON report
    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # CSV report
    if not results:
        return
    headers = list(asdict(results[0]).keys())
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def print_summary(results: List[FileCheckResult]) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.status == "OK")
    warn = sum(1 for r in results if r.status == "WARN")
    fail = sum(1 for r in results if r.status == "FAIL")

    print("============================================================")
    print(f"Checked files: {total}")
    print(f"OK: {ok} | WARN: {warn} | FAIL: {fail}")
    print("------------------------------------------------------------")
    for r in results:
        print(
            f"{r.file}: status={r.status}, rows={r.row_count}, "
            f"invalid_time={r.invalid_time_ms_rows}, dup_time={r.duplicate_time_ms_count}, "
            f"unsorted_pairs={r.not_sorted_count}, missing_points={r.missing_points_count}, "
            f"coverage={r.coverage_percent:.2f}%"
        )
        if r.gap_locations:
            print(f"  gaps: {r.gap_locations}")
        if r.notes:
            print(f"  notes: {r.notes}")
    print("------------------------------------------------------------")
    print(f"JSON report: {REPORT_JSON}")
    print(f"CSV report:  {REPORT_CSV}")


def clean_invalid_time_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "time_ms" not in headers:
            return 0
        rows = list(reader)

    kept: List[Dict[str, str]] = []
    removed = 0
    for row in rows:
        t = (row.get("time_ms") or "").strip()
        if t and t.isdigit():
            kept.append(row)
        else:
            removed += 1

    if removed == 0:
        return 0

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
    return removed


def prompt_and_cleanup_invalid_rows(results: List[FileCheckResult]) -> None:
    targets = [r for r in results if r.invalid_time_ms_rows > 0]
    if not targets:
        print("\nNo invalid time_ms rows found. Cleanup is not needed.")
        return

    total_invalid = sum(r.invalid_time_ms_rows for r in targets)
    print("\nInvalid time_ms rows detected:")
    for r in targets:
        print(f"- {r.file}: invalid_time_ms_rows={r.invalid_time_ms_rows}")
    print(f"Total invalid rows: {total_invalid}")

    ans = input("Delete invalid time_ms rows now? (y/n): ").strip().lower()
    if ans != "y":
        print("Cleanup skipped by user.")
        return

    print("\nCleaning invalid rows...")
    total_removed = 0
    for r in targets:
        p = DATA_DIR / r.file
        removed = clean_invalid_time_rows(p)
        total_removed += removed
        print(f"- {r.file}: removed={removed}")
    print(f"Cleanup done. Total removed rows: {total_removed}")


def main() -> None:
    files = collect_csv_files(DATA_DIR)
    if not files:
        print(f"No CSV files found in '{DATA_DIR}'.")
        return

    results: List[FileCheckResult] = []
    for path in files:
        try:
            results.append(validate_one_file(path))
        except Exception as e:
            results.append(
                FileCheckResult(
                    file=path.name,
                    interval=infer_interval_from_filename(path.name),
                    row_count=0,
                    has_time_ms_column=False,
                    valid_time_ms_rows=0,
                    invalid_time_ms_rows=0,
                    duplicate_time_ms_count=0,
                    not_sorted_count=0,
                    missing_points_count=0,
                    max_gap_ms=0,
                    coverage_percent=0.0,
                    gap_locations="",
                    status="FAIL",
                    notes=f"exception: {e}",
                )
            )

    write_reports(results)
    print_summary(results)
    prompt_and_cleanup_invalid_rows(results)

    # Re-run validation after potential cleanup so reports stay up to date.
    refreshed_results: List[FileCheckResult] = []
    for path in files:
        try:
            refreshed_results.append(validate_one_file(path))
        except Exception as e:
            refreshed_results.append(
                FileCheckResult(
                    file=path.name,
                    interval=infer_interval_from_filename(path.name),
                    row_count=0,
                    has_time_ms_column=False,
                    valid_time_ms_rows=0,
                    invalid_time_ms_rows=0,
                    duplicate_time_ms_count=0,
                    not_sorted_count=0,
                    missing_points_count=0,
                    max_gap_ms=0,
                    coverage_percent=0.0,
                    gap_locations="",
                    status="FAIL",
                    notes=f"exception: {e}",
                )
            )
    write_reports(refreshed_results)
    print("\nRe-validation finished. Reports have been refreshed.")


if __name__ == "__main__":
    main()
