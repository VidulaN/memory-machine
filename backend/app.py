"""
Teleagriculture – Soil Memory Machine
Flask API backend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import os
import pandas as pd
import requests as req
from datetime import date, timedelta

from narrative.generate import generate_narrative

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "drought_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "temp_max", "precipitation", "precip_7day",
    "precip_30day", "temp_7day", "evapotranspiration",
]


# ── Climate data fetching ─────────────────────────────────────────────────────

def fetch_climate(lat: float, lon: float, start_year: int = 1990) -> pd.DataFrame:
    """Fetch daily climate data from Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
        ],
        "start_date": f"{start_year}-01-01",
        "end_date": "2024-12-31",
        "timezone": "auto",
    }

    resp = req.get(url, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json()["daily"]

    df = pd.DataFrame({
        "date":             pd.to_datetime(daily["time"]),
        "temp_max":         daily["temperature_2m_max"],
        "precipitation":    daily["precipitation_sum"],
        "evapotranspiration": daily["et0_fao_evapotranspiration"],
    })

    # Rolling feature engineering
    df["precip_7day"]  = df["precipitation"].rolling(7).mean()
    df["precip_30day"] = df["precipitation"].rolling(30).mean()
    df["temp_7day"]    = df["temp_max"].rolling(7).mean()
    df["evap_7day"]    = df["evapotranspiration"].rolling(7).mean()
    df["precip_14day"] = df["precipitation"].rolling(14).mean()
    df["evap_14day"]   = df["evapotranspiration"].rolling(14).mean()

    return df.dropna()


def fetch_recent(lat: float, lon: float) -> dict:
    """Fetch the most recent 60-day window for live Kit comparison."""
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

    resp = req.get(url, params=params, timeout=30)
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

    row = df.dropna().iloc[-1]
    return row.to_dict()


# ── Aggregation & prediction ──────────────────────────────────────────────────

def aggregate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").agg(
        temp_max         = ("temp_max",          "mean"),
        precipitation    = ("precipitation",      "sum"),
        evapotranspiration = ("evapotranspiration", "mean"),
        precip_7day      = ("precip_7day",        "mean"),
        precip_30day     = ("precip_30day",       "mean"),
        temp_7day        = ("temp_7day",          "mean"),
    ).reset_index()
    return yearly


def predict_years(yearly_df: pd.DataFrame) -> pd.DataFrame:
    X     = yearly_df[FEATURES]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    out = yearly_df.copy()
    out["drought_predicted"]   = preds
    out["drought_probability"] = (probs * 100).round(1)
    return out


def make_forecast(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Extrapolate 2025-2027 using a 5-year rolling mean as the base row."""
    base = yearly_df.tail(5)[FEATURES].mean()

    rows = []
    for yr in [2025, 2026, 2027]:
        row        = base.copy()
        row["year"] = yr
        rows.append(row)

    fdf   = pd.DataFrame(rows)
    preds = model.predict(fdf[FEATURES])
    probs = model.predict_proba(fdf[FEATURES])[:, 1]
    fdf["drought_predicted"]   = preds
    fdf["drought_probability"] = (probs * 100).round(1)
    return fdf


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET"])
def predict():
    lat  = request.args.get("lat",  type=float)
    lon  = request.args.get("lon",  type=float)
    name = request.args.get("name", type=str, default="this location")

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required query parameters"}), 400

    try:
        df      = fetch_climate(lat, lon)
        yearly  = aggregate_by_year(df)
        results = predict_years(yearly)
        fcast   = make_forecast(results)

        current    = results[results["year"] == 2024].iloc[0]
        recent5    = results.tail(5)
        early5     = results.head(5)
        avg_temp   = float(recent5["temp_max"].mean())
        avg_precip = float(recent5["precipitation"].mean())
        trend      = float(
            recent5["drought_probability"].mean()
            - early5["drought_probability"].mean()
        )
        current_prob = float(current["drought_probability"])

        # Build year-by-year payload
        history = []
        for _, row in results.iterrows():
            history.append({
                "year":        int(row["year"]),
                "precip":      round(float(row["precipitation"])),
                "temp":        round(float(row["temp_max"]), 1),
                "drought_prob": round(float(row["drought_probability"]), 1),
                "drought":     int(row["drought_predicted"]),
                "forecast":    False,
            })
        for _, row in fcast.iterrows():
            history.append({
                "year":        int(row["year"]),
                "precip":      round(float(row["precipitation"])),
                "temp":        round(float(row["temp_max"]), 1),
                "drought_prob": round(float(row["drought_probability"]), 1),
                "drought":     int(row["drought_predicted"]),
                "forecast":    True,
            })

        return jsonify({
            "country":      name,
            "lat":          lat,
            "lon":          lon,
            "current_prob": current_prob,
            "avg_temp":     round(avg_temp, 1),
            "avg_precip":   round(avg_precip),
            "trend":        round(trend, 1),
            "narrative":    generate_narrative(
                name, current_prob, avg_temp, avg_precip, trend, current_prob
            ),
            "history":      history,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/recent", methods=["GET"])
def recent():
    """Return the latest 60-day climate snapshot (for Kit vs Open-Data panel)."""
    lat  = request.args.get("lat",  type=float)
    lon  = request.args.get("lon",  type=float)

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon are required"}), 400

    try:
        row = fetch_recent(lat, lon)
        return jsonify({
            "date":             str(row["date"].date()),
            "temp_max":         round(row["temp_max"], 1),
            "precipitation":    round(row["precipitation"], 1),
            "precip_7day":      round(row["precip_7day"], 2),
            "precip_30day":     round(row["precip_30day"], 2),
            "temp_7day":        round(row["temp_7day"], 1),
            "evapotranspiration": round(row["evapotranspiration"], 2),
            "windspeed":        round(row["windspeed"], 1),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  "RandomForest drought classifier",
        "kit_node": "Kit 1001 – Togo (6.14°N, 1.22°E)",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
