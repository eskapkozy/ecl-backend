import traceback

from fastapi import APIRouter, HTTPException

from src.api.service.lgd_modelService import LgdModelService

router = APIRouter(prefix="/lgd", tags=["LGD Model"])

service = LgdModelService()


@router.get("/predict/{loan_id}")
def predict(loan_id: str) -> dict:
    try:
        proportion = service.predict(loan_id)
        return {"loan_id": loan_id, "ldg_proportion": proportion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

