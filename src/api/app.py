from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router.pd_model_route import router as pd_router
from src.api.router.lgd_model_route import router as lgd_router

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
# TODO: remplacer par config centralisée (.env + settings.py)

app = FastAPI()

# -----------------
# DEFAULT and LGD
# -----------------

app.include_router(pd_router)
app.include_router(lgd_router)



# -------------------
# Health endpoint test
# -------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------
# CORS config (MVP simple)
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # plus tard: variable d'env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Routers (à activer quand existants)
# -------------------
# from backend.api.routes.predict import router as predict_router
# app.include_router(predict_router, prefix="/api")