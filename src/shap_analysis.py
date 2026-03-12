"""
SHAP analysis using only the best model (CatBoost).
Based on train_model.py results, CatBoost performed best:
- Accuracy: 0.987
- F1: 0.984
"""

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import shap
import matplotlib.pyplot as plt
import os

# Créer le dossier pour les figures
os.makedirs("figures/shap", exist_ok=True)

print("Script lancé...")
print("=" * 50)
print("Analyse SHAP avec le meilleur modèle: CatBoost")
print("=" * 50)

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
# Entraîner CatBoost avec GridSearchCV (meilleur modèle)
# -------------------------------
print("\nEntraînement CatBoost (meilleur modèle)...")
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

# Évaluer le modèle
print(f"\n--- CatBoost (Best Model) ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_cat), 3))
print("Precision:", round(precision_score(y_test, y_pred_cat), 3))
print("Recall:", round(recall_score(y_test, y_pred_cat), 3))
print("F1:", round(f1_score(y_test, y_pred_cat), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_cat), 3))
print("\nMeilleurs paramètres:", grid_cat.best_params_)

# ==============================
# Analyse SHAP avec CatBoost
# ==============================
print("\n" + "=" * 50)
print("Génération des graphiques SHAP...")
print("=" * 50)

# Features
feature_names = X_train.columns.tolist()

# Créer l'explainer SHAP pour CatBoost
print("\n--- Analyse SHAP: CatBoost ---")
explainer = shap.TreeExplainer(best_cat)

# Calculer les valeurs SHAP
shap_values = explainer.shap_values(X_train)

# Pour les modèles avec plusieurs classes, prendre la classe positive (appendicitis)
if isinstance(shap_values, list):
    shap_values_plot = shap_values[1]  # Classe positive
else:
    shap_values_plot = shap_values

# 1. Graphique résumé SHAP (beeswarm)
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values_plot,
    X_train,
    feature_names=feature_names,
    show=False,
    max_display=20,
)
plt.title(
    "Graphique SHAP Résumé - CatBoost\n(Impact sur la prédiction d'appendicite)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("figures/shap/catboost_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> Résumé sauvegardé: figures/shap/catboost_summary.png")

# 2. Graphique-barres SHAP (importance globale)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_plot,
    X_train,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20,
)
plt.title(
    "Importance des caractéristiques - CatBoost",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "figures/shap/catboost_importance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("  -> Importance sauvegardé: figures/shap/catboost_importance.png")

# 3. Top 5 caractéristiques les plus importantes
print("\n  -> Top 5 symptômes/tests influenceants:")
try:
    sv = shap_values_plot
    # Gérer les différents formats de sortie SHAP
    if hasattr(sv, "shape"):
        if len(sv.shape) == 3:  # Format (n_samples, n_features, n_classes)
            sv = sv[:, :, 1]  # Prendre la classe positive
        mean_abs_shap = np.abs(sv).mean(axis=0)
    else:
        mean_abs_shap = np.abs(sv[1] if isinstance(sv, list) else sv).mean(axis=0)

    # S'assurer que c'est un array 1D
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()

    top_5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    top_5_features = [feature_names[int(i)] for i in top_5_idx]
    for i, feat in enumerate(top_5_features, 1):
        print(f"      {i}. {feat}")
except Exception as e:
    print(f"  -> Impossible d'extraire le top 5: {e}")

# ==============================
# Résumé des symptômes clés
# ==============================
print("\n" + "=" * 50)
print("RÉSUMÉ: Symptômes et tests clés pour CatBoost")
print("=" * 50)

# Calculer l'importance moyenne
mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
top_10_idx = np.argsort(mean_abs_shap)[-10:][::-1]

print("\nLes 10 caractéristiques les plus importantes pour le diagnostic:\n")
for i, idx in enumerate(top_10_idx, 1):
    print(f"  {i:2d}. {feature_names[idx]:40s} (Score: {mean_abs_shap[idx]:.4f})")

print("\n" + "=" * 50)
print("Analyse SHAP terminée !")
print("=" * 50)
print("\nFichiers générés:")
print("  - figures/shap/catboost_summary.png")
print("  - figures/shap/catboost_importance.png")
print("\nFin du script")
