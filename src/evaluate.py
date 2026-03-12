"""
Evaluation module for model performance assessment.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Évalue un modèle et retourne les métriques.

    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        y_pred_proba: Probabilités prédites (optionnel)
        model_name: Nom du modèle

    Returns:
        dict: Métriques du modèle
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    # ROC-AUC si probabilités disponibles
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def print_evaluation(metrics):
    """Affiche les métriques d'évaluation"""
    print(f"\n--- {metrics['model']} ---")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")


def get_confusion_matrix(y_true, y_pred):
    """
    Calcule et retourne la matrice de confusion.

    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions

    Returns:
        array: Matrice de confusion
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    """
    Génère le rapport de classification.

    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions

    Returns:
        str: Rapport de classification
    """
    return classification_report(y_true, y_pred)


def compare_models(results):
    """
    Compare les résultats de plusieurs modèles.

    Args:
        results: Liste de dictionnaires contenant les métriques

    Returns:
        DataFrame: Tableau comparatif des modèles
    """
    df = pd.DataFrame(results)
    df = df.set_index("model")
    return df


def print_comparison(df):
    """Affiche le tableau comparatif des modèles"""
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)

    # Best model par métrique
    print("\nMeilleur modèle par métrique:")
    for col in df.columns:
        best_idx = df[col].idxmax()
        best_val = df[col].max()
        print(f"  {col}: {best_idx} ({best_val:.3f})")
