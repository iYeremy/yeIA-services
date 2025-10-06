import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class CreditDataPreprocessor:
    """
    Preprocesador para German Credit Risk dataset
    """
    def __init__(self):
        self.numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
        self.categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        self.target_feature = 'Risk'

    def fit_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Construye un pipeline para el preprocesamiento de los datos
        """

        #transformar features
        numeric_tf = Pipeline(steps=[("scaler", StandardScaler())]) # estandarizar numeros a el mismo formato
        categorical_if = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]) # convertir formatos a numeros, es decir male=1, female=0 en Sex

        #transformar columnas
        preprocessor = ColumnTransformer(
            transformers=[
                ('nun', numeric_tf, self.numerical_features),
                ('cat', categorical_if, self.categorical_features)
            ],
            remainder="passthrough"
        )

        # ajustar (eliminar columna Target [Risk])
        x_train = df.drop(self.target_feature, axis=1)
        preprocessor.fit(x_train)
        return preprocessor
    
    def process_data(self, df: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[pd.DataFrame, pd.Series]:
        """
        aplicar procesamiento y separar features
        df(pd.Dataframe): proceso del dataframe , preprocessor(ColumnTransformer): fit preprocessing
        """
        df_copy = df.copy()
        df_copy[self.target_feature] = df_copy[self.target_feature].map({'bad': 0, 'good': 1}) #transformar malos pagadores 0, buenos pagadores 1

        y = df_copy[self.target_feature]
        x = df_copy.drop(self.target_feature, axis =1)

        x_processed = preprocessor.transform(x)
        return x_processed, y