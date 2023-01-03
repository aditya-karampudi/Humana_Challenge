from . import constants as params
import pandas as pd
import numpy as np


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

def condition_processing(df):
    condition_df = df[params.feature_column_cols['condition']['numerical_cols'] + params.feature_column_cols['condition']['categorical_cols']]
    condition_df[params.feature_column_cols['condition']['categorical_cols']] = np.where(condition_df[params.feature_column_cols['condition']['categorical_cols']] != 0, 1, 0)
    
    condition_df = sum_feature_generation(df = condition_df, categories = params.condition_categories)
    condition_fe_df = total_sum_features(df = condition_df)
    return condition_fe_df