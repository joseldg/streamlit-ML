from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data(sample_size: int = 20000):
    print("Descargando MNIST... esto puede tardar unos minutos si no está en cache.")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.float32) / 255.0
    y = y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=200,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            n_jobs=-1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    }


def save_model(name: str, model):
    destination = MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, destination)
    print(f"Guardado: {destination}")


def train():
    X_train, X_test, y_train, y_test = load_data()
    models = build_models()

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        save_model(name, model)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} accuracy: {accuracy:.4f}\n")

    print("Entrenamiento completado. Modelos guardados en la carpeta 'models'.")


if __name__ == "__main__":
    train()
