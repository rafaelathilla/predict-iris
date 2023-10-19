
from sklearn.pipeline import Pipeline
from feature_engineering.standard_scaler_pipe import StandardScalerPipe

class PipelineEngineering:
    
    def __init__(self, features) -> None:
        self.features = features
    
    def get_pipeline(self):
        return Pipeline(
            [
                ('scaler', StandardScalerPipe(self.features))
            ]
        )
        