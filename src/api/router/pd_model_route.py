# routes/pd_model_route.py
from fastapi import APIRouter, HTTPException
from src.api.service.pd_modelService import  PDModelService

router = APIRouter(prefix="/pd", tags=["PD Model"])

service = PDModelService()


@router.get("/predict/{loan_id}")
def predict(loan_id: str) -> dict:
    try:
        proba = service.predict(loan_id)
        return {"loan_id": loan_id, "pd_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''
    # POur tester et capturer les erreur avec traceback
    # tout les erreut interne sont capturé. 
@router.get("/predict/{loan_id}")
def predict(loan_id: str):
    try:
        proba = service.predict(loan_id)

        return {
            "loan_id": loan_id,
            "pd_probability": proba
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
'''