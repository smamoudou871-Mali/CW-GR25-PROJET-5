"""
Download data from UCI Machine Learning Repository.
Dataset: Appendicitis
ID: 938
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import os


def download_appendicitis_data():
    """
    Download and save the appendicitis dataset.

    Returns:
        DataFrame: The downloaded dataset
    """
    os.makedirs("data/raw", exist_ok=True)

    print("Téléchargement du dataset...")
    dataset = fetch_ucirepo(id=938)

    # Sauvegarde
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    df.to_csv("data/raw/appendicitis.csv", index=False)
    print("Données sauvegardées dans data/raw/appendicitis.csv")

    return df


if __name__ == "__main__":
    download_appendicitis_data()
