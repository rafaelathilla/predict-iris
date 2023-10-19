import pandas as pd
from io import StringIO
from api.v1.schemas.response_predict import ResponsePredict
from api.v1.schemas.request_predict import RequestPredict
from feature_engineering.pipeline_engineering import PipelineEngineering
from api.services.predictor import Predictor
from utils.utils import Util


class PredictService:
    features = RequestPredict.model_fields.keys()

    def __init__(self) -> None:
        self.util = Util()

    def process(self, data_predict: RequestPredict):
        df_predict = pd.read_json(
            StringIO(data_predict.model_dump_json()), orient="index"
        ).T
        data_process = self.feature_engineering(df_predict.values)
        predictions = self.predict(data_process)
        return ResponsePredict(
            data_input=data_predict,
            data_proc=data_process.values.tolist()[0],
            class_predict=self.to_class(predictions),
        )

    def feature_engineering(self, data):
        pipeline = PipelineEngineering(self.features).get_pipeline()
        return pipeline.transform(data)

    def predict(self, data_process):
        return Predictor().predict(data_process)

    def to_class(self, prediction):
        encoder = self.util.load_model_file("models/lab_enc.pkl")
        label_pred = encoder.inverse_transform(prediction)[0]

        return label_pred
