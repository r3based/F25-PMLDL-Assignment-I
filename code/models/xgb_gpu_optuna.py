import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb


def load_data():
    df = pd.read_csv('../../data/raw/wine_quality.csv')
    X = df.drop('quality', axis=1).copy()
    y = df['quality'].copy()
    # Лёгкий фичеринг
    if 'residual.sugar' in X.columns:
        X['log_residual_sugar'] = np.log1p(X['residual.sugar'])
    if 'chlorides' in X.columns:
        X['log_chlorides'] = np.log1p(X['chlorides'])
    return X, y


def build_params(trial, num_classes):
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        # GPU-режим (современный способ)
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': 42,
        'eta': trial.suggest_float('eta', 0.02, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }
    n_estimators = trial.suggest_int('n_estimators', 400, 1400, step=200)
    return params, n_estimators


def objective(trial: optuna.Trial):
    X, y = load_data()

    # Label encoding для мультикласса 0..K-1
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    params, n_estimators = build_params(trial, num_classes)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, valid_idx in skf.split(X, y_enc):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_enc[train_idx], y_enc[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        y_prob = booster.predict(dvalid)
        y_pred = y_prob.argmax(axis=1)
        f1 = f1_score(y_valid, y_pred, average='weighted')
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def train_best_and_save(study: optuna.Study, out_dir='../../models'):
    X, y = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    best_params, n_estimators = build_params(study.best_trial, num_classes)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    booster = xgb.train(
        best_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'wine_quality_xgb_optuna_gpu.json')
    booster.save_model(model_path)

    import joblib
    le_path = os.path.join(out_dir, 'wine_quality_xgb_label_encoder.pkl')
    joblib.dump(le, le_path)

    # метрики на валидации
    y_prob = booster.predict(dvalid)
    y_pred = y_prob.argmax(axis=1)
    f1 = f1_score(y_valid, y_pred, average='weighted')

    report = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'val_weighted_f1': f1,
        'n_trials': len(study.trials),
    }
    with open(os.path.join(out_dir, 'xgb_optuna_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print('Saved best model and report to', out_dir)


def main():
    print('[Optuna-XGB-GPU] Start tuning')
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize', sampler=sampler, study_name='xgb_gpu_tuning'
    )

    n_trials = int(os.environ.get('XGB_OPTUNA_TRIALS', '60'))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f'Tuning finished in {time.time() - t0:.1f}s')
    print('Best value (weighted F1):', study.best_value)
    print('Best params:', study.best_params)

    train_best_and_save(study)


if __name__ == '__main__':
    main()
