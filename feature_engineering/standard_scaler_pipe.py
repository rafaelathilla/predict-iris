import pandas as pd
from utils.utils import Util
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class StandardScalerPipe(BaseEstimator, TransformerMixin):
    def __init__(self, features) -> None:
        self.features = features
        self.path_file = "models/std_scl.pkl"
        self.util = Util()

    def fit(self, df):
        scaler = StandardScaler()
        scaler.fit(df)
        self.util.save_model_file(scaler, self.path_file)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = self.util.load_model_file(self.path_file)
        std_features = scaler.transform(df)

        return pd.DataFrame(std_features, columns=self.features)
