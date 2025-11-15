#!/usr/bin/env python3
"""Fetch CIMIS station metadata (including GPS) for stations present in the dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests


DATASET_PATH = Path("data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz")
OUTPUT_DIR = Path("data/external")
OUTPUT_JSON = OUTPUT_DIR / "cimis_station_metadata.json"
OUTPUT_CSV = OUTPUT_DIR / "cimis_station_metadata.csv"
API_URL = "https://et.water.ca.gov/api/station"


def ensure_directory(path: Path) -> None:
    """Ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_decimal(value: str) -> float:
    """Extract decimal degrees from CIMIS HMS string."""
    if not value:
        return float("nan")
    if "/" in value:
        decimal_part = value.split("/")[-1].strip()
        try:
            return float(decimal_part)
        except ValueError:
            pass
    # fallback: remove non-numeric characters except minus and dot
    cleaned = "".join(ch for ch in value if ch.isdigit() or ch in ".-")
    return float(cleaned) if cleaned else float("nan")


def load_station_ids(path: Path) -> pd.DataFrame:
    """Load unique station identifiers from the main dataset."""
    df = pd.read_csv(path, usecols=["Stn Id", "Stn Name", "CIMIS Region"]).drop_duplicates()
    df = df.sort_values("Stn Id").reset_index(drop=True)
    return df


def fetch_station_metadata() -> Dict[str, dict]:
    """Fetch all CIMIS station metadata via REST API."""
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    stations = payload.get("Stations", [])
    return {station["StationNbr"]: station for station in stations}


def build_metadata_table(
    station_ids: pd.DataFrame, station_lookup: Dict[str, dict]
) -> pd.DataFrame:
    """Construct metadata table for required stations."""
    records = []
    for _, row in station_ids.iterrows():
        station_id = str(row["Stn Id"])
        station = station_lookup.get(station_id)
        if not station:
            continue
        record = {
            "Stn Id": int(station_id),
            "Stn Name": row["Stn Name"],
            "CIMIS Region": row.get("CIMIS Region"),
            "Name (CIMIS)": station.get("Name"),
            "City": station.get("City"),
            "County": station.get("County"),
            "Elevation (ft)": station.get("Elevation"),
            "IsActive": station.get("IsActive"),
            "IsEtoStation": station.get("IsEtoStation"),
            "Latitude": parse_decimal(station.get("HmsLatitude", "")),
            "Longitude": parse_decimal(station.get("HmsLongitude", "")),
            "ConnectDate": station.get("ConnectDate"),
            "DisconnectDate": station.get("DisconnectDate"),
            "GroundCover": station.get("GroundCover"),
        }
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    ensure_directory(OUTPUT_CSV)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    station_ids = load_station_ids(DATASET_PATH)
    station_lookup = fetch_station_metadata()

    metadata_df = build_metadata_table(station_ids, station_lookup)
    if metadata_df.empty:
        raise RuntimeError("No station metadata retrieved; verify station IDs and API availability.")

    metadata_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(metadata_df.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"Saved metadata for {len(metadata_df)} stations.")
    print(f"- CSV: {OUTPUT_CSV}")
    print(f"- JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

