from typing import Optional
from pydantic import BaseModel
from api.v1.schemas.request_predict import RequestPredict


class ResponsePredict(BaseModel):
    data_input: RequestPredict
    data_proc: Optional[object]
    class_predict: str
