from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

import constants as params

def sum_feature_generation(df, categories):
    for category in categories:
        new_ind_name = category + "_ind_sum"
        subset_cols_criteria = "_" + category + "_"
        subset_cols = [x for x in df.columns if subset_cols_criteria in x and '_ind' in x]
        df[new_ind_name] = df[subset_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        df[subset_cols] = df[subset_cols].astype("category")
        

        new_pmpm_name = category + "_pmpm_sum"
        subset_cols_criteria = "_" + category + "_"
        subset_cols = [x for x in df.columns if subset_cols_criteria in x and '_pmpm' in x]

        df[new_pmpm_name] = df[subset_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    return df


def total_sum_features(df):
    df['total_ind'] = df[df.columns[df.columns.str.contains('_ind_sum')]].sum(axis=1)
    df['total_pmpm'] = df[df.columns[df.columns.str.contains('_pmpm_sum')]].sum(axis=1)
    df['service_bool'] = np.where(df['total_ind'] == 0, 0, 1)
    return df


def credit_attributes(df):
    # data Transformation
    df['credit_num_new_accounts'] = df[['credit_num_autobank_new', 
                                                      'credit_num_autofinance_new',
                                                      'credit_num_consumerfinance_new',
                                                      'credit_num_mtgcredit_new']].astype(float).sum(axis=1)
    
    df['credit_num_collections'] = (df[['credit_num_mtg_collections',
                                                      'credit_num_totalallcredit_collections']].astype(float).sum(axis=1))/2
    
    
    
    credit_df = pd.concat([df[params.feature_column_cols['credit']['numerical_cols']],
                           df[params.feature_curated_cols['credit']['numerical_cols']]], axis=1)
    
    return credit_df


def credit_fillna(credit_df):
    for col in credit_df.columns:
        na_value = params.credit_na_fill[col]
        credit_df[col] = credit_df[col].fillna(na_value)
    return credit_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    input_files = [x for x in input_files if '.csv' in x]
    
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
        
    raw_data = [pd.read_csv(file) for file in input_files]
    
    concat_data = pd.concat(raw_data, axis=1)

    for col, type_ in params.feature_columns_dtype['raw_cols'].items():
        concat_data[col] = concat_data[col].astype(type_)

    concat_data.columns = concat_data.columns.str.lower()


    condition_df = concat_data[params.feature_column_cols['condition']['numerical_cols'] + params.feature_column_cols['condition']['categorical_cols']]
    condition_df[params.feature_column_cols['condition']['categorical_cols']] = np.where(condition_df[params.feature_column_cols['condition']['categorical_cols']] != 0, 1,0)
    condition_df = sum_feature_generation(df = condition_df, categories = params.condition_categories)
    condition_fe_df = total_sum_features(df = condition_df)
    
    
    
    credit_df = credit_attributes(df=concat_data)
    credit_fe_df = credit_fillna(credit_df=credit_df)

    df_fe = credit_fe_df.merge(condition_fe_df, how='left', left_index=True, right_index=True)
    
    
    feature_dtypes_all = {}
    for key in params.feature_columns_dtype.keys():
        feature_columns_all =  {**feature_dtypes_all, **params.feature_columns_dtype[key]}
    
    for col, type_ in feature_dtypes_all.items():
        df_fe[col] = df_fe[col].astype(type_)
        
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())
    
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant'),
        OneHotEncoder(handle_unknown='ignore'))
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
        ("cat", categorical_transformer, make_column_selector(dtype_include="category"))])
    
    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
def input_fn(input_data, content_type):
    """Parse input data payload
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [params.label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names

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
    elif accept == "text/csv":
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
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


