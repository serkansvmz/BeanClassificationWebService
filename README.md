# Dry Bean Classification WebApp

This project serves a PyTorch Dry Bean classifier with FastAPI and provides a simple HTML/CSS/JS web interface.

## Setup

1. Activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put your trained model file here:

`models/bean_classifier.pth`

4. Start the app:

```bash
uvicorn app.main:app --reload
```

5. Open in browser:

`http://127.0.0.1:8000`

## API

- `GET /health` -> checks model and preprocessing status
- `POST /predict` -> predicts class from 16 features

Example request:

```json
{
  "features": [1, 2, 3, 4, 5, 6, 7, 8, 0.7, 0.9, 0.8, 0.6, 0.01, 0.02, 0.03, 0.04]
}
```

## Important for Correct Predictions

If your training used `LabelEncoder`, save class order into `models/class_names.json`:

```json
["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
```

If your training used `StandardScaler`, save scaler stats into `models/scaler.json`:

```json
{
  "mean": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "std": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

If these files are missing, the app uses default class names and no scaling.
