# Teleagriculture — Soil Memory Machine

Pulls satellite-era climate archives and runs them through a RandomForest to flag drought risk, year by year. Pair it with a cheap IoT soil kit and you get a side-by-side view of what the sensors are saying vs what the historical record would predict.

Built this mostly for smallholder farming contexts in West Africa but the API takes any lat/lon so it works anywhere.

---

## What's in here

```
teleagriculture/
├── backend/
│   ├── app.py                  # Flask API
│   ├── build_dataset.py        # pulls + engineers training data
│   ├── train_model.py          # trains the RandomForest, prints eval
│   ├── predict_live.py         # daily CLI report, good for cron
│   ├── requirements.txt
│   ├── data/                   # dataset.csv lives here
│   ├── model/                  # drought_model.pkl lives here
│   └── narrative/
│       ├── __init__.py
│       └── generate.py
└── frontend/
    └── index.html              # single file, no build step needed
```

---

## Getting started

### 1. Python deps

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build the training data

Defaults to the Togo Kit 1001 reference node. Pass `--lat`, `--lon`, `--start`, `--end` to change location.

```bash
python build_dataset.py
# saves to data/dataset.csv
```

For multiple regions, run it a few times and concat:

```bash
python build_dataset.py --lat 7.95 --lon -1.02 --out data/ghana.csv
cat data/dataset.csv data/ghana.csv > data/combined.csv
```

### 3. Train

```bash
python train_model.py --data data/dataset.csv --out model/drought_model.pkl
```

Prints a classification report, confusion matrix, ROC-AUC, 5-fold CV results, and feature importances.

### 4. Run the API

```bash
python app.py
# http://localhost:5000
```

Production:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 5. Open the dashboard

Just open `frontend/index.html` in a browser — no build step, no npm. If the Flask API isn't running it falls back to offline demo data automatically.

---

## API

### `GET /predict`

Fetches historical climate data, runs the model year-by-year, and returns forecasts up to 2027.

| param | type | required | notes |
|-------|------|----------|-------|
| `lat` | float | yes | |
| `lon` | float | yes | |
| `name` | string | no | defaults to "this location" |

```
GET /predict?lat=6.14&lon=1.22&name=Togo
```

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

Last 60 days of climate data — used for the kit vs open-data comparison panel.

Params: `lat`, `lon` (both required).

### `GET /health`

```json
{"status":"ok", "model":"RandomForest drought classifier", "kit_node":"..."}
```

---

## Running as a daily cron job

```bash
python predict_live.py --lat 6.14 --lon 1.22 --name Togo
```

Crontab example (runs at 07:00):
```
0 7 * * * /path/to/.venv/bin/python /path/to/backend/predict_live.py >> /var/log/soil_memory.log 2>&1
```

---

## Dashboard

| thing | what it does |
|-------|-------------|
| choropleth map | 50+ countries, green → red by 2024 drought probability |
| click to analyse | click any country to load its full history + forecast |
| Kit 1001 panel | kit sensor readings next to Open-Meteo archive data |
| filters | all years / last 10 / forecast only |
| region zoom | Africa, Asia, Europe, Americas, Oceania |
| compare tab | up to 5 countries, shared time-series + summary table |
| CSV export | download the full dataset for the selected country |
| dark mode | moon button in the top bar |
| offline fallback | works without the API using deterministic demo data |

---

## Connecting a real IoT kit

The `/recent` endpoint is the bridge. Have your kit firmware POST sensor readings to a small ingestion service that writes to SQLite or Postgres, then update `fetch_recent()` in `app.py` to read from that table. The Kit 1001 panel will pick it up automatically.

---

## Env vars

| var | default | |
|-----|---------|--|
| `FLASK_ENV` | `production` | set to `development` for hot-reload |
| `PORT` | `5000` | |
| `MODEL_PATH` | `model/drought_model.pkl` | |

---

## Dependencies

```
flask==3.0.3
flask-cors==4.0.1
pandas==2.2.2
scikit-learn==1.5.0
requests==2.32.3
numpy==1.26.4
gunicorn==22.0.0
```

Frontend uses D3 v7, TopoJSON v3, and Chart.js v4 via CDN — no npm needed.