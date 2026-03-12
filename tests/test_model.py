import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score
import lightgbm as lgb
import catboost as cb
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def prepare_data():
    df = pd.read_csv("data/raw/appendicitis.csv")
    df = df.dropna(subset=["Diagnosis"])
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Diagnosis"].astype(str))
    X = df.drop(columns=["Diagnosis"])
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = X[col].fillna("unknown")
        X[col] = le.fit_transform(X[col].astype(str))
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TEST 1 — Random Forest avec GridSearch
def test_random_forest_gridsearch():
    X_train, X_test, y_train, y_test = prepare_data()
    rf = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    predictions = best_rf.predict(X_test)
    recall = recall_score(y_test, predictions)
    assert recall >= 0.80
    assert grid_rf.best_params_["n_estimators"] in [100, 200, 300]
    print(f" test_random_forest_gridsearch passé !")
    print(f"   Meilleurs params : {grid_rf.best_params_}")
    print(f"   Recall : {recall:.2%}")

# TEST 2 — LightGBM avec GridSearch
def test_lightgbm_gridsearch():
    X_train, X_test, y_train, y_test = prepare_data()
    lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1)
    param_grid_lgb = {
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.05, 0.1, 0.2],
    }
    grid_lgb = GridSearchCV(lgbm, param_grid_lgb, cv=5, scoring="f1", n_jobs=-1)
    grid_lgb.fit(X_train, y_train)
    best_lgb = grid_lgb.best_estimator_
    predictions = best_lgb.predict(X_test)
    recall = recall_score(y_test, predictions)
    assert recall >= 0.80
    assert grid_lgb.best_params_["n_estimators"] in [100, 200, 300]
    print(f" test_lightgbm_gridsearch passé !")
    print(f"   Meilleurs params : {grid_lgb.best_params_}")
    print(f"   Recall : {recall:.2%}")

# TEST 3 — CatBoost avec GridSearch
def test_catboost_gridsearch():
    X_train, X_test, y_train, y_test = prepare_data()
    cat = cb.CatBoostClassifier(verbose=0, random_state=42)
    param_grid_cat = {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
    }
    grid_cat = GridSearchCV(cat, param_grid_cat, cv=5, scoring="f1", n_jobs=-1)
    grid_cat.fit(X_train, y_train)
    best_cat = grid_cat.best_estimator_
    predictions = best_cat.predict(X_test)
    f1 = f1_score(y_test, predictions)
    assert f1 >= 0.80
    assert grid_cat.best_params_["iterations"] in [100, 200, 300]
    print(f" test_catboost_gridsearch passé !")
    print(f"   Meilleurs params : {grid_cat.best_params_}")
    print(f"   F1 : {f1:.2%}")

# TEST 4 — Données chargées correctement
def test_data_loaded():
    X_train, X_test, y_train, y_test = prepare_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train.isnull().sum().sum() == 0
    print(" test_data_loaded passé !")