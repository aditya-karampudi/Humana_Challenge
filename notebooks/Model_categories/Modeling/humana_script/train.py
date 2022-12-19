import os
from xgboost.sklearn import XGBClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

if __name__ == "__main__":
    debugger = False
    if debugger:
        training_data_directory = "data/"
    else:
        training_data_directory = "/opt/ml/input/data/train"
     
    train_features_data = os.path.join(training_data_directory, "df_fe_encoded_train.csv")
    print("Reading input data")
    train_data = pd.read_csv(train_features_data, header=None)
    X_train = train_data.drop(columns = [0])
    y_train = train_data[[0]]

    # Initialize XGBoost model, use growth tree algorithm similar to lightgbm
    bst = XGBClassifier(n_jobs=-1, grow_policy='lossguide', tree_method ='hist', n_estimators=100)
    print("Training LR model")
    bst.fit(X_train, y_train)
    if debugger:
        model_output_directory = os.path.join('data/', "model.joblib")
    else:
        model_output_directory = os.path.join("/opt/ml/model", "model.joblib")
    print("Saving model to {}".format(model_output_directory))
    joblib.dump(bst, model_output_directory)
