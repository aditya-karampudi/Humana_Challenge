import constants as params
import pandas as pd
import numpy as np

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

def credit_processing(df):
    credit_df = credit_attributes(df=df)
    credit_fe_df = credit_fillna(credit_df=credit_df)
    return credit_fe_df