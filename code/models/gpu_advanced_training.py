"""
GPU-оптимизированное обучение модели с использованием RTX 4060
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
import warnings
import time
warnings.filterwarnings('ignore')

def check_gpu_availability():
    """Проверяет доступность GPU"""
    print("🔍 ПРОВЕРКА GPU")
    print("=" * 25)
    
    # Проверяем CUDA
    try:
        import cupy as cp
        print(f"✅ CuPy доступен")
        print(f"   CUDA версия: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"   Количество GPU: {cp.cuda.runtime.getDeviceCount()}")
        
        # Информация о GPU
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"   GPU {i}: {props['name'].decode()}")
            print(f"   Память: {props['totalGlobalMem'] / 1024**3:.1f} GB")
        
        return True
    except Exception as e:
        print(f"❌ CuPy недоступен: {e}")
        return False

def load_and_analyze_data():
    """Загружает и анализирует данные"""
    print("\n📊 ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
    print("=" * 50)
    
    # Загружаем данные
    df = pd.read_csv('../../data/raw/wine_quality.csv')
    
    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    print(f"Пропущенные значения: {df.isnull().sum().sum()}")
    print(f"Дубликаты: {df.duplicated().sum()}")
    
    # Анализ целевой переменной
    print(f"\n🎯 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (quality)")
    print("=" * 30)
    print(df['quality'].value_counts().sort_index())
    print(f"Среднее качество: {df['quality'].mean():.2f}")
    print(f"Стандартное отклонение: {df['quality'].std():.2f}")
    
    return df

def advanced_preprocessing(df):
    """Продвинутая предобработка данных"""
    print("\n🔧 ПРОДВИНУТАЯ ПРЕДОБРАБОТКА")
    print("=" * 40)
    
    # Создаем копию данных
    df_processed = df.copy()
    
    # 1. Удаляем дубликаты
    initial_size = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    print(f"Удалено дубликатов: {initial_size - len(df_processed)}")
    
    # 2. Обработка выбросов с помощью IQR
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    outliers_removed = 0
    
    for col in numeric_columns:
        if col != 'quality':  # Не обрабатываем целевую переменную
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            outliers_removed += outliers
            
            # Удаляем выбросы
            df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
    
    print(f"Удалено выбросов: {outliers_removed}")
    print(f"Размер после очистки: {df_processed.shape}")
    
    # 3. Создание новых признаков
    print("Создание новых признаков...")
    
    # Соотношения
    df_processed['acid_ratio'] = df_processed['fixed.acidity'] / (df_processed['volatile.acidity'] + 1e-8)
    df_processed['sulfur_ratio'] = df_processed['free.sulfur.dioxide'] / (df_processed['total.sulfur.dioxide'] + 1e-8)
    df_processed['alcohol_density_ratio'] = df_processed['alcohol'] / df_processed['density']
    
    # Логарифмические преобразования
    df_processed['log_residual_sugar'] = np.log1p(df_processed['residual.sugar'])
    df_processed['log_chlorides'] = np.log1p(df_processed['chlorides'])
    
    # Полиномиальные признаки
    df_processed['alcohol_squared'] = df_processed['alcohol'] ** 2
    df_processed['ph_squared'] = df_processed['pH'] ** 2
    
    # Категоризация алкоголя
    df_processed['alcohol_category'] = pd.cut(df_processed['alcohol'], 
                                            bins=[0, 10, 12, 15], 
                                            labels=['low', 'medium', 'high'])
    
    # One-hot encoding для категориальных признаков
    df_processed = pd.get_dummies(df_processed, columns=['alcohol_category'], prefix='alcohol')
    
    print(f"Размер после feature engineering: {df_processed.shape}")
    
    return df_processed

def feature_selection(X, y):
    """Отбор наиболее важных признаков"""
    print("\n🎯 ОТБОР ПРИЗНАКОВ")
    print("=" * 25)
    
    # 1. Univariate feature selection
    selector_univariate = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector_univariate.fit_transform(X, y)
    selected_features = X.columns[selector_univariate.get_support()]
    print(f"Отобрано признаков (univariate): {len(selected_features)}")
    print(f"Признаки: {list(selected_features)}")
    
    return X_selected, selected_features

def train_gpu_models(X_train, X_test, y_train, y_test):
    """Обучение GPU-оптимизированных моделей"""
    print("\n🚀 ОБУЧЕНИЕ GPU-ОПТИМИЗИРОВАННЫХ МОДЕЛЕЙ")
    print("=" * 55)
    
    models = {}
    training_times = {}
    
    # 1. XGBoost с GPU
    print("1. XGBoost (GPU)...")
    start_time = time.time()
    
    xgb_params = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'tree_method': ['gpu_hist'],
        'gpu_id': [0],
        'random_state': [42]
    }
    
    xgb_model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        gpu_id=0,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_grid = GridSearchCV(
        xgb_model, 
        xgb_params, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=1,  # GPU не поддерживает параллелизм
        verbose=1
    )
    
    xgb_grid.fit(X_train, y_train)
    models['XGBoost_GPU'] = xgb_grid.best_estimator_
    training_times['XGBoost_GPU'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {xgb_grid.best_params_}")
    print(f"   Лучший score: {xgb_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['XGBoost_GPU']:.2f} сек")
    
    # 2. LightGBM с GPU
    print("2. LightGBM (GPU)...")
    start_time = time.time()
    
    lgb_params = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'device': ['gpu'],
        'random_state': [42]
    }
    
    lgb_model = lgb.LGBMClassifier(
        device='gpu',
        random_state=42,
        verbose=-1
    )
    
    lgb_grid = GridSearchCV(
        lgb_model, 
        lgb_params, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=1,
        verbose=1
    )
    
    lgb_grid.fit(X_train, y_train)
    models['LightGBM_GPU'] = lgb_grid.best_estimator_
    training_times['LightGBM_GPU'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {lgb_grid.best_params_}")
    print(f"   Лучший score: {lgb_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['LightGBM_GPU']:.2f} сек")
    
    # 3. CatBoost с GPU
    print("3. CatBoost (GPU)...")
    start_time = time.time()
    
    catboost_params = {
        'iterations': [500, 1000, 1500],
        'depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'task_type': ['GPU'],
        'devices': ['0'],
        'random_seed': [42]
    }
    
    catboost_model = CatBoostClassifier(
        task_type='GPU',
        devices='0',
        random_seed=42,
        verbose=False
    )
    
    catboost_grid = GridSearchCV(
        catboost_model, 
        catboost_params, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=1,
        verbose=1
    )
    
    catboost_grid.fit(X_train, y_train)
    models['CatBoost_GPU'] = catboost_grid.best_estimator_
    training_times['CatBoost_GPU'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {catboost_grid.best_params_}")
    print(f"   Лучший score: {catboost_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['CatBoost_GPU']:.2f} сек")
    
    # 4. CPU модели для сравнения
    print("4. Random Forest (CPU) для сравнения...")
    start_time = time.time()
    
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    models['RandomForest_CPU'] = rf_grid.best_estimator_
    training_times['RandomForest_CPU'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {rf_grid.best_params_}")
    print(f"   Лучший score: {rf_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['RandomForest_CPU']:.2f} сек")
    
    return models, training_times

def evaluate_models(models, X_test, y_test, training_times):
    """Оценка всех моделей"""
    print("\n📊 ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 30)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Предсказания
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        prediction_time = time.time() - start_time
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'training_time': training_times[name],
            'prediction_time': prediction_time
        }
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-score: {f1:.4f}")
        print(f"   Время обучения: {training_times[name]:.2f} сек")
        print(f"   Время предсказания: {prediction_time:.4f} сек")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_weighted')
        print(f"   CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def save_best_model(models, results, selected_features):
    """Сохраняет лучшую модель"""
    print("\n💾 СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 35)
    
    # Находим лучшую модель по F1-score
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    best_score = results[best_model_name]['f1_score']
    
    print(f"Лучшая модель: {best_model_name}")
    print(f"F1-score: {best_score:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"Время обучения: {results[best_model_name]['training_time']:.2f} сек")
    
    # Создаем директорию
    os.makedirs('../../models', exist_ok=True)
    
    # Сохраняем модель
    model_path = '../../models/wine_quality_gpu_model.pkl'
    joblib.dump(best_model, model_path)
    
    # Сохраняем информацию о признаках
    features_path = '../../models/gpu_selected_features.pkl'
    joblib.dump(selected_features, features_path)
    
    # Сохраняем метрики
    metrics_path = '../../models/gpu_model_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"GPU Wine Quality Model\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"F1-score: {best_score:.4f}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
        f.write(f"Training Time: {results[best_model_name]['training_time']:.2f} сек\n")
        f.write(f"Prediction Time: {results[best_model_name]['prediction_time']:.4f} сек\n")
        f.write(f"Selected Features: {len(selected_features)}\n")
        f.write(f"Features: {list(selected_features)}\n")
        
        # Сравнение всех моделей
        f.write(f"\nВсе модели:\n")
        for name, result in results.items():
            f.write(f"{name}: F1={result['f1_score']:.4f}, Acc={result['accuracy']:.4f}, Time={result['training_time']:.2f}s\n")
    
    print(f"Модель сохранена в {model_path}")
    print(f"Признаки сохранены в {features_path}")
    print(f"Метрики сохранены в {metrics_path}")
    
    return best_model_name, best_model

def main():
    """Основная функция"""
    print("🚀 GPU-ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ МОДЕЛИ КАЧЕСТВА ВИНА")
    print("=" * 70)
    
    # 1. Проверка GPU
    gpu_available = check_gpu_availability()
    if not gpu_available:
        print("⚠️  GPU недоступен, но продолжим с CPU версиями")
    
    # 2. Загрузка и анализ данных
    df = load_and_analyze_data()
    
    # 3. Продвинутая предобработка
    df_processed = advanced_preprocessing(df)
    
    # 4. Подготовка данных
    X = df_processed.drop('quality', axis=1)
    y = df_processed['quality']
    
    # 5. Отбор признаков
    X_selected, selected_features = feature_selection(X, y)
    
    # 6. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 25)
    print(f"Train set: {X_train.shape[0]} образцов")
    print(f"Test set: {X_test.shape[0]} образцов")
    print(f"Признаков: {X_train.shape[1]}")
    
    # 7. Обучение GPU моделей
    models, training_times = train_gpu_models(X_train, X_test, y_train, y_test)
    
    # 8. Оценка моделей
    results = evaluate_models(models, X_test, y_test, training_times)
    
    # 9. Сохранение лучшей модели
    best_model_name, best_model = save_best_model(models, results, selected_features)
    
    print(f"\n🎉 GPU ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 40)
    print(f"Лучшая модель: {best_model_name}")
    print(f"F1-score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"Ускорение: {results['RandomForest_CPU']['training_time'] / results[best_model_name]['training_time']:.2f}x")

if __name__ == "__main__":
    main()

