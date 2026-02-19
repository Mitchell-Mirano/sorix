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

    def state_dict(self):
        """Devuelve un diccionario con el estado del transformador de columnas."""
        state = {
            'n_features': self.n_features,
            'features_names': self.features_names,
            'transformers_states': []
        }
        for name, tf, cols in self.transformers:
            if hasattr(tf, 'state_dict') and callable(tf.state_dict):
                state['transformers_states'].append((name, tf.state_dict(), cols))
            else:
                # Fallback if the transformer doesn't have state_dict (unlikely in sorix)
                state['transformers_states'].append((name, tf, cols))
        return state

    def load_state_dict(self, state_dict):
        """Carga el estado del transformador de columnas."""
        self.n_features = state_dict['n_features']
        self.features_names = state_dict['features_names']
        
        # Mapping for easier lookup
        tf_states = {name: (state, cols) for name, state, cols in state_dict['transformers_states']}
        
        for name, tf, cols in self.transformers:
            if name in tf_states:
                state, _ = tf_states[name]
                if hasattr(tf, 'load_state_dict') and callable(tf.load_state_dict):
                    tf.load_state_dict(state)
                else:
                    # If it was saved as the object itself
                    # (this might happen if tf didn't have state_dict when saved)
                    # But in our new system they will have it.
                    pass
        return self

