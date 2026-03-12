from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import shap
import matplotlib.pyplot as plt
import os

# Créer le dossier pour les figures
os.makedirs("figures/shap", exist_ok=True)

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
# Fonction d’évaluation
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

# ==============================
# Analyse SHAP - Explicabilité
# ==============================
print("\n" + "=" * 50)
print("Génération des graphiques SHAP...")
print("=" * 50)


# Fonction pour générer les graphiques SHAP
def generate_shap_analysis(model, X_train, model_name, feature_names):
    """Génère les graphiques SHAP pour un modèle"""
    print(f"\n--- Analyse SHAP: {model_name} ---")

    # Créer l'explainer SHAP
    if model_name == "LightGBM":
        explainer = shap.TreeExplainer(model)
    elif model_name == "CatBoost":
        explainer = shap.TreeExplainer(model)
    else:  # Random Forest
        explainer = shap.TreeExplainer(model)

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
        f"Graphique SHAP Résumé - {model_name}\n(Impact sur la prédiction d'appendicite)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"figures/shap/{model_name.lower()}_summary.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  -> Résumé sauvegardé: figures/shap/{model_name.lower()}_summary.png")

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
        f"Importance des caractéristiques - {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"figures/shap/{model_name.lower()}_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"  -> Importance sauvegardé: figures/shap/{model_name.lower()}_importance.png"
    )

    # 3. Top 5 caractéristiques les plus importantes
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
        print(f"  -> Top 5 symptômes/tests influenceants:")
        for i, feat in enumerate(top_5_features, 1):
            print(f"      {i}. {feat}")
    except Exception as e:
        print(f"  -> Impossible d'extraire le top 5: {e}")

    return shap_values, explainer


# Générer l'analyse SHAP pour chaque modèle
print("\n" + "=" * 50)
print("Analyse SHAP - Modèle par modèle")
print("=" * 50)

# Features
feature_names = X_train.columns.tolist()

# Random Forest
shap_rf, _ = generate_shap_analysis(best_rf, X_train, "RandomForest", feature_names)

# LightGBM
shap_lgb, _ = generate_shap_analysis(best_lgb, X_train, "LightGBM", feature_names)

# CatBoost
shap_cat, _ = generate_shap_analysis(best_cat, X_train, "CatBoost", feature_names)

# ==============================
# Graphique comparatif des modèles
# ==============================
print("\n--- Génération du graphique comparatif ---")

# Calculer l'importance moyenne pour chaque modèle
importance_rf = np.abs(shap_rf[1] if isinstance(shap_rf, list) else shap_rf).mean(
    axis=0
)
importance_lgb = np.abs(shap_lgb[1] if isinstance(shap_lgb, list) else shap_lgb).mean(
    axis=0
)
importance_cat = np.abs(shap_cat[1] if isinstance(shap_cat, list) else shap_cat).mean(
    axis=0
)

# Top 10 features communes
top_features_idx = np.argsort(importance_rf + importance_lgb + importance_cat)[-10:]
top_features_names = [feature_names[i] for i in top_features_idx]

plt.figure(figsize=(14, 8))
x = np.arange(len(top_features_names))
width = 0.25

plt.barh(
    x - width,
    importance_rf[top_features_idx],
    width,
    label="Random Forest",
    color="#2ecc71",
)
plt.barh(x, importance_lgb[top_features_idx], width, label="LightGBM", color="#3498db")
plt.barh(
    x + width,
    importance_cat[top_features_idx],
    width,
    label="CatBoost",
    color="#e74c3c",
)

plt.yticks(x, top_features_names)
plt.xlabel("Importance SHAP moyenne", fontsize=12)
plt.title(
    "Comparaison des caractéristiques importantes\n(Top 10 - Tous modèles)",
    fontsize=14,
    fontweight="bold",
)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/shap/comparative_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> Comparaison sauvegardée: figures/shap/comparative_importance.png")

# ==============================
# Résumé des symptômes clés
# ==============================
print("\n" + "=" * 50)
print("RÉSUMÉ: Symptômes et tests clés")
print("=" * 50)

# Fusionner les importances
all_importances = importance_rf + importance_lgb + importance_cat
top_10_idx = np.argsort(all_importances)[-10:][::-1]

print("\nLes 10 caractéristiques les plus importantes pour le diagnostic:\n")
for i, idx in enumerate(top_10_idx, 1):
    print(
        f"  {i:2d}. {feature_names[idx]:40s} (Score combiné: {all_importances[idx]:.4f})"
    )

print("\n" + "=" * 50)
print("Analyse SHAP terminée !")
print("=" * 50)
print("\nFichiers générés:")
print("  - figures/shap/randomforest_summary.png")
print("  - figures/shap/lightgbm_summary.png")
print("  - figures/shap/catboost_summary.png")
print("  - figures/shap/comparative_importance.png")
print("\nFin du script")
