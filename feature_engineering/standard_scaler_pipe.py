
import pandas as pd
from utils.utils import save_model_file, load_model_file
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class StandardScalerPipe(BaseEstimator, TransformerMixin):
    
    def __init__(self, features) -> None:
        self.features = features
        self.path_file = 'models/std_scl.pkl'
        
        
    def fit(self, df):
        scaler = StandardScaler()
        scaler.fit(df)    
        save_model_file(scaler, self.path_file)
        
        return self
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = load_model_file(self.path_file)
        std_features = scaler.transform(df)
    
        return pd.DataFrame(std_features, columns=self.features)

        
        
        
        
        

    