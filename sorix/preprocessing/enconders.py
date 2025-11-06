import numpy as np
import pandas as pd

class OneHotEncoder:
    def __init__(self):
        self.n_features = 0
        self.features = {

        }

    def fit(self, X:pd.DataFrame|np.ndarray):
        
        self.n_features = X.shape[1] if X.ndim == 2 else 1
        categorical_features = X.columns if isinstance(X, pd.DataFrame) else [f"F{i}" for i in range(self.n_features)]

        for feature in categorical_features:
            unique_values = np.unique(X[feature])
            self.features[feature] = unique_values
        return self
    
    def transform(self, X:pd.DataFrame|np.ndarray):


        X_one_hot = np.zeros((len(X),1))

        for feature in self.features.keys():
            X_one_hot = np.hstack([X_one_hot, np.eye(len(self.features[feature]))[np.searchsorted(self.features[feature], X[feature])]])

        self.n_features = X_one_hot.shape[1]-1
        return X_one_hot[:,1:]
    
    def fit_transform(self, X:np.ndarray|pd.DataFrame):
    
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X):
        return np.argmax(X, axis=1)
    

    def get_features_names(self):

        features = []

        for feature,cats in self.features.items():
            for cat in cats:
                features.append(f"{feature}_{cat}")

        return features
            
            
    

    def __str__(self) -> str:
        return f"OneHotEncoder({self.features})"
    
    def __repr__(self) -> str:
        return self.__str__()