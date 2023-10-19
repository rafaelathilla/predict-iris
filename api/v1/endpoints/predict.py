from fastapi import APIRouter

from api.v1.schemas.request_predict import RequestPredict
from api.services.predict_service import PredictService

router = APIRouter()

@router.post("/predict",
            description="Predict species Iris",
            summary="Run predict species Iris",
            response_model = RequestPredict)
def predict_iris(data_predict: RequestPredict):
    PredictService().process(data_predict)

    
    return data_predict