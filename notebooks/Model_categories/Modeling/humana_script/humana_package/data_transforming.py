from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder


def transformer(df):
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

    categorical_transformer =  Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=999)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
        ("cat", categorical_transformer, make_column_selector(dtype_include="category"))])
    
    preprocessor.fit(df)
    return preprocessor