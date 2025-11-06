import numpy as np

class ColumnTransformer:
    def __init__(self, transformers: list[tuple]):
        self.transformers = transformers
        self.n_features = 0
        self.features_names = []


    def fit_transform(self, X):
        
        X_final = np.zeros((len(X), 1))

        for transformer in self.transformers:
            tf = transformer[1]
            features = transformer[2]

            Xt = tf.fit_transform(X[features])
            self.n_features += tf.n_features
            X_final = np.hstack([X_final,Xt])



        return X_final[:,1:]
    
    def transform(self, X):

        X_final = np.zeros((len(X), 1))

        for transformer in self.transformers:
            tf = transformer[1]
            features = transformer[2]

            Xt = tf.transform(X[features])
            X_final = np.hstack([X_final,Xt])



        return X_final[:,1:]
    

    def get_features_names(self):
        
        features = []

        for transformer in self.transformers:
            cat = transformer[0]
            tf = transformer[1]

            features.extend([f'{cat}_{feature}' for feature in tf.get_features_names()])

        return features
