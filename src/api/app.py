from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# TODO: remplacer par config centralisée (.env + settings.py)

app = FastAPI()

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