import joblib
import shap
import os

MODEL_PATH = "appendicite_pediatric/models/model_appendicite.pkl"
EXPLAINER_PATH = "appendicite_pediatric/explications/explainer_shap.pkl"

def main():
    print("Chargement du modèle...")
    model = joblib.load(MODEL_PATH)

    # Si le modèle est un pipeline, extraire le classifieur
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    else:
        classifier = model

    # Créer l'explainer (TreeExplainer fonctionne pour CatBoost, LightGBM, RF)
    explainer = shap.TreeExplainer(classifier)
    print("TreeExplainer créé.")

    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"Explainer sauvegardé dans {EXPLAINER_PATH}")

if __name__ == "__main__":
    main()