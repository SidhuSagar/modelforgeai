from fastapi import APIRouter, HTTPException
from api.schemas import PredictRequest
from api.utils import success_response, error_response
from services.model_service import ModelService, list_available_models

router = APIRouter()

@router.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "ModelForge AI Phase-7 API"}

@router.get("/models", tags=["models"])
def models():
    models = list_available_models()
    return success_response({"models": models}, "Available models listed")

@router.post("/predict", tags=["predict"])
def predict(req: PredictRequest):
    try:
        svc = ModelService()
        pred, probs, meta = svc.predict(req.model_id, req.input)
        result = {
            "status": "success",
            "model_id": req.model_id,
            "prediction": pred,
            "probabilities": probs,
            "meta": meta
        }
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
