from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np
from pathlib import Path

app = FastAPI(title="Wine Classification API")
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "wine_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Pick the names you want shown in the UI
LABELS = {
    0: "Barolo",       # was 'class_0'
    1: "Grignolino",   # was 'class_1'
    2: "Barbera"       # was 'class_2'
}

class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

@app.get("/")
def root():
    return {"message": "Wine Classification API is running"}

@app.post("/predict")
def predict(data: WineInput):
    X = np.array([[
        data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
        data.magnesium, data.total_phenols, data.flavanoids, data.nonflavanoid_phenols,
        data.proanthocyanins, data.color_intensity, data.hue, data.od280_od315, data.proline
    ]])
    cls = int(model.predict(X)[0])
    label = LABELS.get(cls, f"class_{cls}")  # safe fallback
    return {"predicted_class": cls, "predicted_label": label}
