"""Auto-generated curated CTA signals (see tool/run_cta_signal_sweep.py)."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "cta_signal_catalog_top30.csv"

def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_CSV)
