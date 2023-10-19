import pandas as pd
from io import StringIO
from api.v1.schemas.request_predict import RequestPredict
from feature_engineering.pipeline_engineering import PipelineEngineering
from api.services.predictor import Predictor

class PredictService:
    
    features = RequestPredict.model_fields.keys()
    
    def __init__(self) -> None:
        ...
    
    def process(self, data_predict: RequestPredict):
        df_predict = pd.read_json(StringIO(data_predict.model_dump_json()), orient ='index').T
        data_process = self.feature_engineering(df_predict.values)
        predictions = self.predict(data_process)
        print(predictions)
        
        
    def feature_engineering(self, data):
        pipeline = PipelineEngineering(self.features).get_pipeline()
        return  pipeline.transform(data)
        
        
    def predict(self, data_process):
        return Predictor().predict(data_process)

        
    
        
     