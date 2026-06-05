"""
predict_live.py
Fetch today's climate snapshot for a given location, run the model,
and print a narrative report. Designed to run as a daily cron job.

Usage:
    python predict_live.py [--lat LAT] [--lon LON] [--name NAME]
"""

import argparse
import pickle
import os
from datetime import date, timedelta

import pandas as pd
import requests

from narrative.generate import generate_narrative

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "drought_model.pkl")

FEATURES = [
    "temp_max", "precipitation", "precip_7day",
    "precip_30day", "temp_7day", "evapotranspiration",
]


def fetch_recent(lat: float, lon: float) -> pd.Series:
    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=60)

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
        "start_date": str(start),
        "end_date":   str(end),
        "timezone":   "auto",
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

    df["precip_7day"]  = df["precipitation"].rolling(7).mean()
    df["precip_30day"] = df["precipitation"].rolling(30).mean()
    df["temp_7day"]    = df["temp_max"].rolling(7).mean()

    return df.dropna().iloc[-1]


def predict(row: pd.Series) -> tuple[int, list[float]]:
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    X        = pd.DataFrame([row[FEATURES]])
    pred     = mdl.predict(X)[0]
    probs    = mdl.predict_proba(X)[0].tolist()
    return int(pred), probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat",  type=float, default=6.14)
    parser.add_argument("--lon",  type=float, default=1.22)
    parser.add_argument("--name", type=str,   default="Togo")
    args = parser.parse_args()

    print(f"Fetching latest climate data for {args.name} ...")
    row = fetch_recent(args.lat, args.lon)

    print(f"  Latest date  : {row['date'].date()}")
    print(f"  Temp max     : {row['temp_max']:.1f}°C")
    print(f"  Precip 30d   : {row['precip_30day']:.2f}mm")

    pred, probs = predict(row)

    features = {k: row[k] for k in FEATURES}
    narrative = generate_narrative(features, pred, probs)

    separator = "─" * 55
    print(f"\n{separator}")
    print("  SOIL MEMORY MACHINE — Daily Report")
    print(separator)
    print(f"  Date      : {row['date'].date()}")
    print(f"  Location  : {args.name}  ({args.lat}°N, {args.lon}°E)")
    print(f"  Drought   : {'YES' if pred == 1 else 'NO'}  "
          f"(confidence {max(probs) * 100:.1f}%)")
    print(separator)
    print(f"\n{narrative}\n")
    print(separator)


if __name__ == "__main__":
    main()
