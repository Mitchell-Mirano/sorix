# Preprocessing

The preprocessing module in **sorix** offers a comprehensive set of tools for data cleaning and feature engineering. These utilities transform raw data into a numerical format suitable for deep learning models, ensuring numerical stability and faster convergence.

### Feature Scaling

*   **[Scalers](01-Scalers.ipynb)**: Normalize or standardize numeric features. Includes `MinMaxScaler`, `StandardScaler`, and `RobustScaler`.

### Categorical Encoding

*   **[Encoders](02-Encoders.ipynb)**: Transform categorical variables into numerical representations. Currently supports `OneHotEncoder`.

### Pipeline Integration

*   **[ColumnTransformer](03-ColumnTransformer.ipynb)**: Apply different transformations to different columns in a single, reusable object.
