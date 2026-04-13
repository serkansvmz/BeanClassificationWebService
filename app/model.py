from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

import torch
import torch.nn as nn


class BeanClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


CLASS_NAMES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

MODEL_PATH = Path("models/bean_classifier.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")
SCALER_PATH = Path("models/scaler.json")


def load_model(model_path: Path = MODEL_PATH) -> BeanClassifier:
    model = BeanClassifier()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_class_names(path: Path = CLASS_NAMES_PATH) -> List[str]:
    if not path.exists():
        return CLASS_NAMES

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) != 7:
        return CLASS_NAMES

    return [str(item) for item in data]


def load_scaler(path: Path = SCALER_PATH) -> Tuple[List[float], List[float]] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    mean = data.get("mean")
    std = data.get("std")
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 16
        or len(std) != 16
    ):
        return None

    safe_std = [float(v) if float(v) != 0.0 else 1.0 for v in std]
    return [float(v) for v in mean], safe_std


def predict(
    model: BeanClassifier,
    features: List[float],
    class_names: List[str],
    scaler: Tuple[List[float], List[float]] | None = None,
) -> Tuple[str, List[float]]:
    processed = features
    if scaler is not None:
        mean, std = scaler
        processed = [(x - m) / s for x, m, s in zip(features, mean, std)]

    input_tensor = torch.tensor([processed], dtype=torch.float32)
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())

    predicted_class = class_names[pred_idx]
    return predicted_class, probs


def debug_predict(
    model: BeanClassifier,
    features: List[float],
    class_names: List[str],
    scaler: Tuple[List[float], List[float]] | None = None,
) -> Dict[str, Any]:
    processed = features
    if scaler is not None:
        mean, std = scaler
        processed = [(x - m) / s for x, m, s in zip(features, mean, std)]

    input_tensor = torch.tensor([processed], dtype=torch.float32)
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())

    ranked = sorted(
        [
            {"class_name": class_names[idx], "probability": float(prob)}
            for idx, prob in enumerate(probs)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "predicted_class": class_names[pred_idx],
        "raw_features": [float(x) for x in features],
        "processed_features": [float(x) for x in processed],
        "probabilities": [float(x) for x in probs],
        "top3": ranked[:3],
        "class_names": class_names,
        "scaler_loaded": scaler is not None,
    }
