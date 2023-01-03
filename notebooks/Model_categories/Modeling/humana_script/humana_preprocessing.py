from __future__ import print_function

import time
import sys
from io import StringIO
import os

import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
from joblib import dump, load
import joblib



import warnings
warnings.filterwarnings('ignore')

import humana_package.constants as params
import humana_package.data_preprocessing as preprocessing
import logging

logger = logging.getLogger()
    

    
def preprocess_data(args):
    
    # Take the set of files and read them all into a single pandas dataframe
    if args.debugger==True:
        input_data_path = args.input_data_path
    else:
        input_data_path = "/opt/ml/processing/input"
        
    if args.debugger==True:
        import sagemaker
        import boto3
        input_files = []
        s3_client = boto3.client("s3")
        bucket_name = "humana-data"
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=args.input_data_path.replace("s3://{}/".format(bucket_name),''))
        files = response.get("Contents")

        for file in files:
            input_files.append(file['Key'].split('/')[-1])
        
        input_files = [args.input_data_path + '/' + file for file in input_files if '.csv' in file]
        print(input_files)
    else:    
        input_files = [ os.path.join(input_data_path, file) for file in os.listdir(input_data_path) ]
        input_files = [x for x in input_files if '.csv' in x]
        
    print("The files used for processing the {} data {}".format(args.train_or_valid_or_test, input_files))
    logger.info("THE FILES ARE IMPORTED")
    raw_data = [pd.read_csv(file) for file in input_files]
    df = pd.concat(raw_data, axis=1)

    print("Successfully imported data from S3. Shape of the {} data {}".format(args.train_or_valid_or_test, df.shape))
    
#     raw_colnames = []
#     for type_, col_names in params.raw_columns.items():
#         raw_colnames.extend(col_names)
        
#     if len(df.columns) == len(raw_colnames) + 1:
#             # This is a labelled example, includes the ring label
#         df.columns = raw_colnames + [params.label_column]
#     elif len(df.columns) == len(raw_colnames):
#         df.columns = raw_colnames

    print("Successfully imported data from S3. Shape of the {} data {}".format(args.train_or_valid_or_test, df.shape))
    df = df.head(3000)
    df_fe = preprocessing.preprocessor(df=df)
    print("Preprocessed {} data".format(args.train_or_valid_or_test))
    
    if params.label_column in df.columns:
        df_fe = df[[params.label_column]].merge(df_fe, how ='left', left_index=True, right_index=True)
        
    return df, df_fe
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environent variables.
    parser.add_argument('--train_or_valid_or_test', type=str, default="test")
    
    # If using notebookdf_fe_encoded.csv for debugging or execution
    parser.add_argument('--debugger', type=bool, default=False)
    parser.add_argument('--input_data_path', type=str)

    args = parser.parse_args()
                      
    df, df_fe = preprocess_data(args=args)
    
    print(df_fe.shape)
    
    if args.debugger == True:
        filename = "data/df_fe_{}.csv".format(args.train_or_valid_or_test)
        df_fe.to_csv(filename, header=False, index=False)
        print("Saved the files")
    else:
        path_dir = "/opt/ml/processing/{}".format(args.train_or_valid_or_test)
        filename = "df_fe_{}.csv".format(args.train_or_valid_or_test)
        train_features_output_path = os.path.join(path_dir, filename)
        df_fe.to_csv(train_features_output_path, header=False, index=False)
            
            
#     if args.train_or_test == "train":                
#         sc = transforming.transformer(df = df_fe)
#         print("Train data shape after preprocessing: {}".format(df_fe.shape))
        
#         df_fe_encoded = sc.transform(df_fe)
#         print("Transformed Train data")
    
#         df_fe_encoded = df[[params.label_column]].merge(pd.DataFrame(df_fe_encoded), how = 'left', left_index=True, right_index=True)
        
#         for col, type_ in params.label_columns_dtype.items():
#             df_fe_encoded[col] = df_fe_encoded[col].astype(type_)
#         print(df_fe_encoded.head())
#         print(df_fe_encoded.shape)                
#         print(df_fe_encoded.columns)
#         if args.debugger == True:
#             pd.DataFrame(df_fe_encoded).to_csv("data/df_fe_encoded_train.csv", header=False, index=False)
#             joblib.dump(sc, 'data/std_scaler.joblib',)
# #             joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))
#             print("Saved the files")
#         else:
#             train_features_output_path = os.path.join("/opt/ml/processing/train", "df_fe_encoded_train.csv")
#             pd.DataFrame(df_fe_encoded).to_csv(train_features_output_path, header=None, index=False)
            
#             sc_features_output_path = os.path.join("/opt/ml/model", "std_scaler.joblib")
#             dump(sc, sc_features_output_path)
            
#     if args.train_or_test == "test":
        
#         if args.debugger == True:
#             sc = load("data/std_scaler.bin")
#         else:
#             sc = load("/opt/ml/objects/std_scaler_2.bin")
            
#         df_fe_encoded = sc.transform(df_fe)
#         print("Transformed Test data")
        
#         if args.debugger == True:
#             pd.DataFrame(df_fe_encoded).to_csv("data/df_fe_encoded_test.csv", header=False, index=False)
#             print("Saved the Files")
        
        
#         preprocessor = transforming.transformer(df = df_fe)
#         df_fe_encoded = preprocessor.transform(df_fe)
#         print("Transformed Train data")
        
#         df_xy = df[[params.label_column]].merge(df_fe_encoded, how = 'left', left_index=True, right_index=True)
                        
#         for col, type_ in params.label_columns_dtype.items():
#             df_xy[col] = df_xy[col].astype(type_)
                        
#         # Get the dependent column to first row
#         rearrange_cols = [params.label_column]  + [col for col in df_xy.columns if col != params.label_column]
#         df_xy = df_xy[cols]
        
        
                        
                        
   
    
    
# def input_fn(input_data, content_type):
#     """Parse input data payload
#     We currently only take csv input. Since we need to process both labelled
#     and unlabelled data we first determine whether the label column is present
#     by looking at how many columns were provided.
#     """
#     if content_type == "text/csv":
        
#         raw_colnames = []
#         for type_, col_names in params.raw_columns.items():
#             raw_colnames.extend(col_names)
        
#         input_files = [ os.path.join(valid_input, file) for file in os.listdir(params.valid_input) ]
#         input_files = [x for x in input_files if '.csv' in x]
        
#         if len(input_files) == 0:
#             raise ValueError(('There are no files in {}.\n' +
#                               'This usually indicates that the channel ({}) was incorrectly specified,\n' +
#                               'the data specification in S3 was incorrectly specified or the role specified\n' +
#                               'does not have permission to access the data.').format(params.valid_input, "valid"))
            
#         # Read the raw input data as CSV.
#         raw_data = [pd.read_csv(file, header=None, skiprows = 1) for file in input_files]
#         df = pd.concat(raw_data, axis=1)
        
#         print(type(df))
#         print(df.shape)
#         print('----------------------------------------')
# #         df = pd.read_csv(StringIO(input_data), header=None, skiprows = 1)

#         if len(df.columns) == len(raw_colnames) + 1:
#             # This is a labelled example, includes the ring label
#             df.columns = raw_colnames + [params.label_column]
#         elif len(df.columns) == len(raw_colnames):
#             df.columns = raw_colnames
            
#         print(df.shape)
#         print(df.columns)
#         print(df.head())
#         df_fe = preprocessing.preprocessor(df=df)
        
#         return df_fe
#     else:
#         raise ValueError("{} not supported by script!".format(content_type))


# def output_fn(prediction, accept):
#     """Format prediction output
#     The default accept/content-type between containers for serial inference is JSON.
#     We also want to set the ContentType or mimetype as the same value as accept so the next
#     container can read the response payload correctly.
#     """
#     if accept == "application/json":
#         instances = []
#         for row in prediction.tolist():
#             instances.append({"features": row})

#         json_output = {"instances": instances}

#         return worker.Response(json.dumps(json_output), mimetype=accept)
#     elif accept == "text/csv":
#         return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
#     else:
#         raise RuntimeException("{} accept type is not supported by this script.".format(accept))


# def predict_fn(input_data, model):
#     """Preprocess input data
#     We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
#     so we want to use .transform().
#     The output is returned in the following order:
#         rest of features either one hot encoded or standardized
#     """
#     features = model.transform(input_data)

#     if params.label_column in input_data:
#         # Return the label (as the first column) and the set of features.
#         return np.insert(features, 0, input_data[params.label_column], axis=1)
#     else:
#         # Return only the set of features
#         return features


# def model_fn(model_dir):
#     """Deserialize fitted model"""
#     preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
#     return preprocessor
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


