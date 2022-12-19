import constants as params
import data_transforming as transforming
import numpy as np
from io import StringIO
import pandas as pd
import joblib
import argparse
import os

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


if __name__ == "__main__":    
        
        parser = argparse.ArgumentParser()
        try:
            os.environ["SM_OUTPUT_DATA_DIR"]
        except KeyError:
            os.environ["SM_OUTPUT_DATA_DIR"] = "data/df_fe_train.csv"
        try:
            os.environ["SM_MODEL_DIR"]
        except KeyError:
            os.environ["SM_MODEL_DIR"] = "data/df_fe_train.csv"        
        try:
            os.environ["SM_CHANNEL_TRAIN"]
        except KeyError:
            os.environ["SM_CHANNEL_TRAIN"] = "data/df_fe_train.csv"
            
        print(os.environ["SM_OUTPUT_DATA_DIR"], "model", os.environ["SM_MODEL_DIR"], "train", os.environ["SM_CHANNEL_TRAIN"])
        # Sagemaker specific arguments. Defaults are set in the environment variables.
        parser.add_argument('--debugger', type=bool, default=False)
        parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
        parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
        
        args = parser.parse_args()

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
        input_files = [x for x in input_files if '.csv' in x]
        
        print(input_files)
        if len(input_files) == 0:
            raise ValueError(
                (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
                ).format(args.train, "train")
            )
#         feature_names = list(params.feature_columns_dtype.keys()) + list(params.label_columns_dtype.keys())

        raw_data = [
            pd.read_csv(
                file,
                header=None,
#                 skiprows = 1,
#                 names=params.fe_column_names,
                dtype={**params.feature_columns_dtype, **params.label_columns_dtype},
            )
            for file in input_files
        ]
        df_fe = pd.concat(raw_data)
        
        if len(df_fe.columns) == len(params.fe_column_names) + 1:
            # This is a labelled example, includes the ring label
            df_fe.columns = params.fe_column_names + [params.label_column]
        elif len(df_fe.columns) == len(params.fe_column_names):
            # This is an unlabelled example.
            df_fe.columns = params.fe_column_names
        print(df_fe.shape)
        print(df_fe.columns)

        # Labels should not be preprocessed. predict_fn will reinsert the labels after featurizing.
        df_fe.drop(params.label_column, axis=1, inplace=True)

        for col, type_ in params.feature_columns_dtype.items():
            df_fe[col] = df_fe[col].astype(type_)
            
        print(df_fe.dtypes.value_counts())
        sc = transforming.transformer(df = df_fe)
        print("Train data shape after preprocessing: {}".format(df_fe.shape))
        joblib.dump(sc, os.path.join(args.model_dir, "model.joblib"))
        

def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), 
                         header=None)
        
        if len(df.columns) == len(params.fe_column_names) + 1:
            dependent = True
            # This is a labelled example, includes the ring label
            df.columns = params.fe_column_names + [params.label_column]
        elif len(df.columns) == len(params.fe_column_names):
            # This is an unlabelled example.
            df.columns = params.fe_column_names
            dependent =False
            
        for col, type_ in params.feature_columns_dtype.items():
            df[col] = df[col].astype(type_)
            
        if dependent:
            df[params.label_column] = df[params.label_column].astype(params.label_columns_dtype[params.label_column])

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if params.label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[params.label_column], axis=1)
    else:
        # Return only the set of features
        return features
    
def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
        
        
        
        
        
        