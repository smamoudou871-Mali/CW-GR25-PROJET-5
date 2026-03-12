import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import load_data, preprocess_data, get_data_summary

# TEST 1 — Vérifier que les données se chargent bien
def test_load_data():
    df = load_data()
    assert df is not None
    assert len(df) > 0
    assert "Diagnosis" in df.columns
    print(" test_load_data passé !")

# TEST 2 — Vérifier gestion des valeurs manquantes
def test_no_missing_values():
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    print(" test_no_missing_values passé !")

# TEST 3 — Vérifier le split 80/20
def test_split_ratio():
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    total = len(X_train) + len(X_test)
    ratio = len(X_test) / total
    assert 0.18 <= ratio <= 0.22
    print(" test_split_ratio passé !")

# TEST 4 — Vérifier get_data_summary
def test_data_summary():
    df = load_data()
    summary = get_data_summary(df)
    assert "n_rows" in summary
    assert "n_columns" in summary
    assert "missing_values" in summary
    assert summary["n_rows"] > 0
    print(" test_data_summary passé !")

# TEST 5 — Vérifier le chargement du modèle
def test_load_model():
    import joblib
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'LightGBM.pkl'
    )
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None
        print(" test_load_model passé !")
    else:
        pytest.skip("Modèle pas encore entraîné")