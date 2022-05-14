import sys

import pandas as pd
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost
import category_encoders as ce
from lightgbm import LGBMClassifier, LGBMRegressor
from typing import Union
from joblib import dump, load


def metric_evaluation(model: Union[BaseEstimator, CatBoost], x, y):
    preds = model.predict(x)
    probas = model.predict_proba(x)[:, 1]
    accuracy = accuracy_score(preds, y)
    auc = roc_auc_score(y, probas)
    print(f"auc:\t{auc}\naccuracy:\t{accuracy}")
    return auc, accuracy


if __name__ == '__main__':
    df = pd.read_csv("../datasets/adult/adult_kaggle.csv")
    # setting up target column
    df["income"] = df.income.map(dict(zip(df.income.unique(), range(df.income.nunique()))))

    train_df, test_df = train_test_split(df, train_size=0.8, stratify=df["income"])
    x_train, y_train, x_test, y_test = train_df.drop("income", axis="columns"), train_df["income"], test_df.drop(
        "income", axis="columns"), test_df["income"]

    non_numeric_cols = list(df.select_dtypes(exclude=['int64']).columns)
    encoder = ce.CatBoostEncoder(cols=non_numeric_cols)
    x_train = encoder.fit_transform(x_train, y_train)
    x_test = encoder.transform(x_test, y_test)

    rf_txt = open("rf_scores.txt", "w")
    stdout_backup = sys.stdout
    sys.stdout = rf_txt
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)
    dump(rf_model, "rf_model.joblib")
    print(f"Random Forest model:\n{'=' * 50}\ntrain:")
    metric_evaluation(rf_model, x_train, y_train)
    print(f"{'-' * 50}\nvalidation:")
    metric_evaluation(rf_model, x_test, y_test)
    rf_txt.close()
    sys.stdout = stdout_backup

    xgb_txt = open("xgb_scores.txt", "w")
    stdout_backup = sys.stdout
    sys.stdout = xgb_txt
    xgb_model = XGBClassifier(use_label_encoder=False)
    xgb_model.fit(x_train, y_train)
    dump(xgb_model, "xgb_model.joblib")
    print(f"XGBoost model:\n{'=' * 50}\ntrain:")
    metric_evaluation(xgb_model, x_train, y_train)
    print(f"{'-' * 50}\nvalidation:")
    metric_evaluation(xgb_model, x_test, y_test)
    xgb_txt.close()
    sys.stdout = stdout_backup

    cat_txt = open("cat_scores.txt", "w")
    stdout_backup = sys.stdout
    sys.stdout = cat_txt
    cat_model = CatBoostClassifier(verbose=False)
    cat_model.fit(x_train, y_train)
    dump(cat_model, "cat_model.joblib")
    print(f"CatBoost model:\n{'=' * 50}\ntrain:")
    metric_evaluation(cat_model, x_train, y_train)
    print(f"{'-' * 50}\nvalidation:")
    metric_evaluation(cat_model, x_test, y_test)
    cat_txt.close()
    sys.stdout = stdout_backup

    lgbm_txt = open("lgbm_scores.txt", "w")
    stdout_backup = sys.stdout
    sys.stdout = lgbm_txt
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(x_train, y_train)
    dump(lgbm_model, "lgbm_model.joblib")
    print(f"LightGBM model:\n{'=' * 50}\ntrain:")
    metric_evaluation(lgbm_model, x_train, y_train)
    print(f"{'-' * 50}\nvalidation:")
    metric_evaluation(lgbm_model, x_test, y_test)
    lgbm_txt.close()
    sys.stdout = stdout_backup

    print("Done.")
