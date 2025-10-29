from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

def train_wine_model():
    # Load wine dataset
    data = load_wine()
    X, y = data.data, data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_dir = Path(__file__).resolve().parents[1] / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "wine_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved at {model_path}")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_wine_model()
