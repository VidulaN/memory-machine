# Teleagriculture — Soil Memory Machine

A precision drought intelligence platform combining satellite-era climate archives,
machine learning, and real-time IoT kit data for agricultural decision-making.

---

## Project structure

```
teleagriculture/
├── backend/
│   ├── app.py                  # Flask REST API
│   ├── build_dataset.py        # Fetch + engineer training data
│   ├── train_model.py          # Train & evaluate RandomForest
│   ├── predict_live.py         # CLI daily report (cron-friendly)
│   ├── requirements.txt
│   ├── data/                   # dataset.csv goes here
│   ├── model/                  # drought_model.pkl goes here
│   └── narrative/
│       ├── __init__.py
│       └── generate.py         # Narrative text generation
└── frontend/
    └── index.html              # Self-contained dashboard (no build step)
```

---

## Quick start

### 1 — Install Python dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Build the training dataset

Fetches daily Open-Meteo data for the Togo Kit 1001 reference node by default.
You can pass `--lat`, `--lon`, `--start`, `--end` to use a different location.

```bash
python build_dataset.py
# Saved to data/dataset.csv
```

To build a multi-region dataset, run the script several times with different
coordinates and concatenate the CSVs:

```bash
python build_dataset.py --lat 7.95 --lon -1.02 --out data/ghana.csv
cat data/dataset.csv data/ghana.csv > data/combined.csv
```

### 3 — Train the model

```bash
python train_model.py --data data/dataset.csv --out model/drought_model.pkl
```

You will see a classification report, confusion matrix, ROC-AUC score,
5-fold cross-validation results, and feature importances printed to stdout.

### 4 — Start the API server

```bash
python app.py
# Running on http://localhost:5000
```

Or with Gunicorn for production:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 5 — Open the dashboard

Open `frontend/index.html` directly in your browser (no build step needed).

If the Flask API is not reachable the dashboard automatically falls back to
offline demo data so the UI is always functional.

---

## API reference

### `GET /predict`

Fetch historical climate data, run the ML model year-by-year, and return
2025–2027 forecasts.

| Parameter | Type   | Required | Description               |
|-----------|--------|----------|---------------------------|
| `lat`     | float  | ✓        | Latitude                  |
| `lon`     | float  | ✓        | Longitude                 |
| `name`    | string |          | Display name (default: "this location") |

**Example:**
```
GET /predict?lat=6.14&lon=1.22&name=Togo
```

**Response:**
```json
{
  "country":      "Togo",
  "lat":          6.14,
  "lon":          1.22,
  "current_prob": 58.3,
  "avg_temp":     31.2,
  "avg_precip":   412,
  "trend":        4.1,
  "narrative":    "...",
  "history": [
    {"year":1990, "precip":390, "temp":30.1, "drought_prob":42.0, "drought":0, "forecast":false},
    ...
    {"year":2027, "precip":380, "temp":32.1, "drought_prob":68.0, "drought":1, "forecast":true}
  ]
}
```

### `GET /recent`

Returns the last 60-day climate snapshot for Kit vs Open-Data comparison.

| Parameter | Type  | Required |
|-----------|-------|----------|
| `lat`     | float | ✓        |
| `lon`     | float | ✓        |

### `GET /health`

Returns `{"status":"ok", "model":"RandomForest drought classifier", "kit_node":"..."}`.

---

## Daily live report (cron)

Run `predict_live.py` as a scheduled job to get a daily printed narrative:

```bash
python predict_live.py --lat 6.14 --lon 1.22 --name Togo
```

Add to crontab (runs every morning at 07:00):
```cron
0 7 * * * /path/to/.venv/bin/python /path/to/backend/predict_live.py >> /var/log/soil_memory.log 2>&1
```

---

## Dashboard features

| Feature | Description |
|---------|-------------|
| **Choropleth map** | 50+ countries colour-coded green → red by 2024 drought probability |
| **Click-to-analyse** | Select any country to load full historical + forecast data |
| **Kit 1001 panel** | Side-by-side comparison of Kit sensor readings vs Open-Meteo archive |
| **Filters** | All years / Last 10 years / Forecast only — applied to all charts |
| **Region zoom** | Snap to Africa, Asia, Europe, Americas, Oceania |
| **Compare tab** | Add up to 5 countries, view shared time-series + summary table |
| **CSV export** | Download the selected country's full dataset |
| **Dark mode** | Toggle via the moon button in the top bar |
| **Offline fallback** | Works without the Flask API using deterministic demo data |

---

## Connecting a real IoT kit

The `/recent` endpoint is designed as the data bridge. In your kit firmware,
POST sensor readings to a thin ingestion service that writes them into a small
SQLite or PostgreSQL table. Then modify `fetch_recent()` in `app.py` to read
from that table instead of (or alongside) the Open-Meteo call.

The frontend `Kit 1001` panel will automatically show live values once the
`/recent` endpoint returns real sensor data.

---

## Environment variables

| Variable     | Default                 | Description                        |
|--------------|-------------------------|------------------------------------|
| `FLASK_ENV`  | `production`            | Set to `development` for hot-reload |
| `PORT`       | `5000`                  | API port                           |
| `MODEL_PATH` | `model/drought_model.pkl` | Path to the trained model          |

---

## Dependencies

| Package       | Version | Purpose                      |
|---------------|---------|------------------------------|
| flask         | 3.0.3   | REST API framework           |
| flask-cors    | 4.0.1   | Cross-origin requests        |
| pandas        | 2.2.2   | Data wrangling               |
| scikit-learn  | 1.5.0   | RandomForest classifier      |
| requests      | 2.32.3  | Open-Meteo HTTP client       |
| numpy         | 1.26.4  | Numerical operations         |
| gunicorn      | 22.0.0  | Production WSGI server       |

Frontend uses CDN-delivered D3 v7, TopoJSON v3, and Chart.js v4 — no npm needed.
