from typing import List

from pydantic import BaseModel, Field, conlist


class PredictionRequest(BaseModel):
    features: conlist(float, min_length=16, max_length=16)


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: List[float] = Field(
        ..., description="Class probabilities in model class order."
    )
