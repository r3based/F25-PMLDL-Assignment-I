"""
FastAPI приложение для предсказания качества вина (XGBoost GPU/CPU)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import xgboost as xgb
import pandas as pd

# Создаем FastAPI приложение
app = FastAPI(title="Wine Quality Prediction API", version="2.0.0")

# Пути к модели и энкодеру
MODEL_PATH = "/app/models/wine_quality_xgb_optuna_gpu.json"
LE_PATH = "/app/models/wine_quality_xgb_label_encoder.pkl"

# Совместимо с Python 3.9 (без оператора | для аннотаций)
booster = None  # type: xgb.Booster
label_encoder = None

# Колонки, как при обучении (оригинальные с точками) + engineered
TRAIN_COLUMNS = [
    'fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides',
    'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    'log_residual_sugar', 'log_chlorides'
]

# Pydantic модель для входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

@app.on_event("startup")
async def load_model():
    """Загружает XGBoost Booster и LabelEncoder при старте"""
    global booster, label_encoder
    try:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        label_encoder = joblib.load(LE_PATH)
        print("XGBoost модель и LabelEncoder загружены")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "Wine Quality Prediction API (XGBoost)", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": booster is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_wine_quality(features: WineFeatures):
    if booster is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Сопоставляем входные поля API -> обучающие имена с точками
        base = {
            'fixed.acidity': features.fixed_acidity,
            'volatile.acidity': features.volatile_acidity,
            'citric.acid': features.citric_acid,
            'residual.sugar': features.residual_sugar,
            'chlorides': features.chlorides,
            'free.sulfur.dioxide': features.free_sulfur_dioxide,
            'total.sulfur.dioxide': features.total_sulfur_dioxide,
            'density': features.density,
            'pH': features.ph,
            'sulphates': features.sulphates,
            'alcohol': features.alcohol,
        }
        # Инженерные признаки как при обучении
        base['log_residual_sugar'] = float(np.log1p(base['residual.sugar']))
        base['log_chlorides'] = float(np.log1p(base['chlorides']))

        # Формируем DataFrame с правильным порядком колонок
        df = pd.DataFrame([[base[col] for col in TRAIN_COLUMNS]], columns=TRAIN_COLUMNS)
        dmatrix = xgb.DMatrix(df)

        probs = booster.predict(dmatrix)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = int(label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(np.max(probs))

        return PredictionResponse(prediction=pred_label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при предсказании: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    if booster is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    return {
        "model_type": "XGBoost (GPU-capable)",
        "version": "optuna_tuned",
        "features": TRAIN_COLUMNS,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
