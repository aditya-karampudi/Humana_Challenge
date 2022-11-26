from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder


def transformer(df):
    
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant'),
        OneHotEncoder(handle_unknown='ignore'))
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
        ("cat", categorical_transformer, make_column_selector(dtype_include="category"))])
    
    preprocessor.fit(df)
    return preprocessor