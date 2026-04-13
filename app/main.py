from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import (
    CLASS_NAMES,
    MODEL_PATH,
    debug_predict,
    load_class_names,
    load_model,
    load_scaler,
    predict,
)
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Dry Bean Classifier", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

model = None
class_names = CLASS_NAMES
scaler = None


@app.on_event("startup")
def startup_event() -> None:
    global model, class_names, scaler
    if not MODEL_PATH.exists():
        model = None
        return
    model = load_model(MODEL_PATH)
    class_names = load_class_names()
    scaler = load_scaler()


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"class_names": class_names},
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "class_names": class_names,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_bean(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model is not loaded. Put your .pth file at models/bean_classifier.pth "
                "and restart the app."
            ),
        )

    predicted_class, probabilities = predict(
        model=model,
        features=payload.features,
        class_names=class_names,
        scaler=scaler,
    )
    return PredictionResponse(
        predicted_class=predicted_class, probabilities=probabilities
    )


@app.post("/debug/predict")
def predict_bean_debug(payload: PredictionRequest) -> dict:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model is not loaded. Put your .pth file at models/bean_classifier.pth "
                "and restart the app."
            ),
        )

    return debug_predict(
        model=model,
        features=payload.features,
        class_names=class_names,
        scaler=scaler,
    )
