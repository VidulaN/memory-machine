"""
build_dataset.py
Fetch historical climate data from Open-Meteo and engineer features + drought labels.
Default coordinates: Togo Kit 1001 reference node (6.14°N, 1.22°E).
Usage:
    python build_dataset.py [--lat LAT] [--lon LON] [--start YEAR] [--end YEAR]
"""

import argparse
import os
import requests
import pandas as pd

os.makedirs("data", exist_ok=True)


def fetch_and_build(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    print(f"Fetching Open-Meteo archive  {lat}°N {lon}°E  {start} → {end} ...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "precipitation_hours",
            "windspeed_10m_max",
        ],
        "start_date": start,
        "end_date":   end,
        "timezone":   "Africa/Abidjan",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    df = pd.DataFrame({
        "date":             pd.to_datetime(daily["time"]),
        "temp_max":         daily["temperature_2m_max"],
        "precipitation":    daily["precipitation_sum"],
        "evapotranspiration": daily["et0_fao_evapotranspiration"],
        "precip_hours":     daily["precipitation_hours"],
        "windspeed":        daily["windspeed_10m_max"],
    })

    # ── Rolling features (model inputs) ──────────────────────────────────────
    df["precip_7day"]  = df["precipitation"].rolling(7).mean()
    df["precip_30day"] = df["precipitation"].rolling(30).mean()
    df["temp_7day"]    = df["temp_max"].rolling(7).mean()
    df["evap_7day"]    = df["evapotranspiration"].rolling(7).mean()

    # ── Label: 14-day windows keep label independent of 30-day features ──────
    df["precip_14day"] = df["precipitation"].rolling(14).mean()
    df["evap_14day"]   = df["evapotranspiration"].rolling(14).mean()

    precip_low = df["precip_14day"].quantile(0.25)
    evap_high  = df["evap_14day"].quantile(0.75)

    df["drought_risk"] = (
        (df["precip_14day"] < precip_low) &
        (df["evap_14day"]   > evap_high)
    ).astype(int)

    df = df.dropna()

    drought_days = df["drought_risk"].sum()
    total_days   = len(df)
    print(f"Drought days : {drought_days} / {total_days}  "
          f"({drought_days / total_days * 100:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build drought training dataset")
    parser.add_argument("--lat",   type=float, default=6.14,         help="Latitude")
    parser.add_argument("--lon",   type=float, default=1.22,         help="Longitude")
    parser.add_argument("--start", type=str,   default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str,   default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--out",   type=str,   default="data/dataset.csv")
    args = parser.parse_args()

    df = fetch_and_build(args.lat, args.lon, args.start, args.end)
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")
    print(df[["date", "temp_max", "precipitation", "precip_30day", "drought_risk"]].tail(10).to_string())


if __name__ == "__main__":
    main()
