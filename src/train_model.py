from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import joblib

print("Script lancé...")

# Charger les données
df = pd.read_csv("data/raw/appendicitis.csv")

# Supprimer les lignes où la cible est manquante
df = df.dropna(subset=["Diagnosis"])

# Vérifie le nom de la colonne cible
target_col = "Diagnosis"

# Encoder la cible
le_target = LabelEncoder()
y = le_target.fit_transform(df[target_col].astype(str))

X = df.drop(columns=[target_col])

# Encoder les variables catégorielles
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = X[col].fillna("unknown")
    X[col] = le.fit_transform(X[col].astype(str))

# Imputer les valeurs manquantes avec la médiane
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"Valeurs manquantes après imputation: {X.isnull().sum().sum()}")
print(f"Taille du dataset: {X.shape}")
print(f"Classes: {le_target.classes_}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 1. Random Forest avec GridSearchCV
print("\nEntraînement Random Forest...")
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
y_pred_rf = best_rf.predict(X_test)

# -------------------------------
# 2. LightGBM avec GridSearchCV
print("Entraînement LightGBM...")
lgbm = lgb.LGBMClassifier(random_state=42)
param_grid_lgb = {
    "n_estimators": [100, 200, 300],
    "max_depth": [-1, 5, 10],
    "learning_rate": [0.05, 0.1, 0.2],
}
grid_lgb = GridSearchCV(lgbm, param_grid_lgb, cv=5, scoring="f1", n_jobs=-1)
grid_lgb.fit(X_train, y_train)
best_lgb = grid_lgb.best_estimator_
y_pred_lgb = best_lgb.predict(X_test)

# -------------------------------
# 3. CatBoost avec GridSearchCV
print("Entraînement CatBoost...")
cat = cb.CatBoostClassifier(verbose=0, random_state=42)
param_grid_cat = {
    "iterations": [100, 200, 300],
    "depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}
grid_cat = GridSearchCV(cat, param_grid_cat, cv=5, scoring="f1", n_jobs=-1)
grid_cat.fit(X_train, y_train)
best_cat = grid_cat.best_estimator_
y_pred_cat = best_cat.predict(X_test)


# -------------------------------
# Fonction d'évaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall:", round(recall_score(y_true, y_pred), 3))
    print("F1:", round(f1_score(y_true, y_pred), 3))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_pred), 3))


# Évaluer les trois modèles
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("LightGBM", y_test, y_pred_lgb)
evaluate_model("CatBoost", y_test, y_pred_cat)

# Afficher les meilleurs hyperparamètres
print("\nMeilleurs paramètres RF:", grid_rf.best_params_)
print("Meilleurs paramètres LGBM:", grid_lgb.best_params_)
print("Meilleurs paramètres CatBoost:", grid_cat.best_params_)
print("Fin du script")


# Choisir le meilleur modèle (ex: best_cat)
best_model = best_cat  # ou best_rf, best_lgb selon vos scores
joblib.dump(best_model, "appendicite_pediatric/models/model_appendicite.pkl")
print("Modèle sauvegardé")