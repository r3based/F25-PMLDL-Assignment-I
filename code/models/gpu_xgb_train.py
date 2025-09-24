import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb


def load_data():
    df = pd.read_csv('../../data/raw/wine_quality.csv')
    return df


def preprocess(df):
    # Базовые фичи + легкий feature engineering (без тяжелых шагов)
    X = df.drop('quality', axis=1).copy()
    y = df['quality'].copy()

    # Преобразование: добавить логарифм двух скошенных признаков
    if 'residual.sugar' in X.columns:
        X['log_residual_sugar'] = np.log1p(X['residual.sugar'])
    if 'chlorides' in X.columns:
        X['log_chlorides'] = np.log1p(X['chlorides'])

    # Кодируем метки для XGBoost
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le


def train_xgb_gpu(X_train, X_val, y_train, y_val):
    # Определяем, доступна ли GPU
    use_gpu = False
    tree_method = 'hist'
    try:
        import cupy as cp  # noqa: F401
        try:
            ndev = cp.cuda.runtime.getDeviceCount()
            if ndev > 0:
                use_gpu = True
                tree_method = 'gpu_hist'
        except Exception:
            use_gpu = False
            tree_method = 'hist'
    except Exception:
        use_gpu = False
        tree_method = 'hist'

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.1,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'tree_method': tree_method,
        'random_state': 42,
    }

    if use_gpu:
        params['predictor'] = 'gpu_predictor'

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'valid')]

    print(f"Using GPU: {use_gpu} | tree_method={tree_method}")
    print("Start training...")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    return model


def main():
    print("[GPU-XGB] Training start")
    df = load_data()
    X, y_enc, le = preprocess(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    t0 = time.time()
    model = train_xgb_gpu(X_train, X_val, y_train, y_val)
    t_train = time.time() - t0

    dval = xgb.DMatrix(X_val)
    y_prob = model.predict(dval)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1-weighted: {f1:.4f}")

    os.makedirs('../../models', exist_ok=True)
    model_path = '../../models/wine_quality_xgb_gpu.json'
    label_path = '../../models/wine_quality_xgb_label_encoder.pkl'
    model.save_model(model_path)

    import joblib
    joblib.dump(le, label_path)

    print(f"Saved model to {model_path}")
    print(f"Saved label encoder to {label_path}")
    print(f"Training time: {t_train:.2f}s")


if __name__ == '__main__':
    main()
