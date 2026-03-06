from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import routers as api_routers

app = FastAPI(title="ModelForge AI Phase-7 API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_routers.router, prefix="/api")

@app.get("/", tags=["root"])
def root():
    return {"message": "ModelForge AI Phase-7 API running"}
