
import pandas as pd
import numpy as np
import constants as params
import condition as condition
import credit as credit

from sklearn.utils import resample

# def oversampling_fn(df):

#     df_train_minority = X_train[X_train[params.label_column] == 1]
#     df_train_minority_upsampled_issue = resample(df_train_minority,
#                                            replace=True,     # sample with replacement
#                                            n_samples=39722,    # to match majority class
#                                            random_state=123) # repr)
#     df_train_majority = X_train[X_train.transportation_issues == 0]
#     df_train_balanced = df_train_majority.append(df_train_minority_upsampled_issue)
    
#     y_train = df_train_balanced['transportation_issues']
#     X_train = df_train_balanced.drop(columns=['transportation_issues'])
#     X_train.reset_index(inplace=True,drop=True)
#     y_train.reset_index(inplace=True,drop=True)
    
#     df_valid_minority = X_valid[X_valid.transportation_issues == 1]
#     df_valid_minority_upsampled_issue = resample(df_valid_minority,
#                                            replace=True,     # sample with replacement
#                                            n_samples=8512,    # to match majority class
#                                            random_state=123) # repr)
#     df_valid_majority = X_valid[X_valid.transportation_issues == 0]
#     df_valid_balanced = df_valid_majority.append(df_valid_minority_upsampled_issue)
    

#     y_valid = df_valid_balanced['transportation_issues'] 
#     X_valid = df_valid_balanced.drop(columns=['transportation_issues'])
#     X_valid.reset_index(inplace=True,drop=True)
#     y_valid.reset_index(inplace=True,drop=True)

#     return X_train,X_valid,y_train,y_valid


def preprocessor(df):
    for col, type_ in params.feature_columns_dtype['raw_cols'].items():
        df[col] = df[col].astype(type_)

    df.columns = df.columns.str.lower()
    condition_fe_df = condition.condition_processing(df=df)
    credit_fe_df = credit.credit_processing(df=df)

    df_fe = credit_fe_df.merge(condition_fe_df, how='left', left_index=True, right_index=True)

    for col, type_ in params.feature_dtypes_all.items():
        df_fe[col] = df_fe[col].astype(type_)
        
    return df_fe