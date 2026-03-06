from fastapi.responses import JSONResponse
from typing import Dict, Any

def success_response(data: Dict[str, Any], message: str = "Success"):
    payload = {"status": "success", "message": message, "data": data}
    return JSONResponse(content=payload)

def error_response(message: str, status_code: int = 400):
    payload = {"status": "error", "message": message}
    return JSONResponse(status_code=status_code, content=payload)
