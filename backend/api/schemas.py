from typing import Optional, Dict, Any
from pydantic import BaseModel

class PredictRequest(BaseModel):
    model_id: str
    input: Dict[str, Any]

class PredictResponse(BaseModel):
    status: str
    model_id: str
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, Any]] = None
