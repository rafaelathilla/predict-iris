from fastapi import APIRouter

from api.v1.schemas.request_predict import RequestPredict
from api.services.predict_service import PredictService
from api.v1.schemas.response_predict import ResponsePredict

router = APIRouter()


@router.post(
    "/predict",
    description="Predict species Iris",
    summary="Run predict species Iris",
    response_model=ResponsePredict,
)
def predict_iris(data_predict: RequestPredict):
    return PredictService().process(data_predict)
