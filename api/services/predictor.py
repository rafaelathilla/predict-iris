
from utils.utils import Singleton, load_model_file
from sklearn.base import BaseEstimator, ClassifierMixin

class Predictor(BaseEstimator, ClassifierMixin, metaclass=Singleton):
    
    def __init__(self):
        self.model_path = 'models/knn_model.pkl'
        self.model = load_model_file(self.model_path)

    def fit(self, x, y):
        return self

    def predict(self, data):
        return self.model.predict(data)