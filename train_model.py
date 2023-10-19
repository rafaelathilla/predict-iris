import os
import sys
import logging
import pickle
import pandas as pd
from utils.utils import save_model_file
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from feature_engineering.pipeline_engineering import PipelineEngineering
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO) 


def main():
    dataset_iris = pd.read_csv('data/iris-data.csv', skiprows=0, delimiter=',')
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    label_enc = LabelEncoder()
    label_enc.fit(dataset_iris['class'])
    dataset_iris['class'] = label_enc.transform(dataset_iris['class'])
    
    
    X = dataset_iris.drop(['class'], axis=1).values
    Y = dataset_iris['class'].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    
    pipeline = PipelineEngineering(features).get_pipeline()
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    
    X_train.to_csv('data/train_after_feature_engineering.csv', index=False)
    X_test.to_csv('data/test_after_feature_engineering.csv', index=False)
    
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, Y_train)
    
    logger.info('###### Avaliação do Modelo ######')
    Y_pred = knn_model.predict(X_test)
    
    logger.info('..... Matriz de Confusão .......')
    cm = confusion_matrix(Y_test, Y_pred)
    df_cm = pd.DataFrame(cm, index=label_enc.classes_, columns=label_enc.classes_)
    logger.info(df_cm)
    
    
    logger.info('..... Relatório de Classificação .......')
    logger.info(classification_report(Y_test, Y_pred, target_names=label_enc.classes_))
    
    with open('models/knn_model.pkl', 'wb') as pickle_file:
        pickle.dump(knn_model, pickle_file)
        
        
    

if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(parent_dir)
    main()

