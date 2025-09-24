"""
Оптимизированное обучение модели с улучшенной точностью
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
import time
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Загружает и анализирует данные"""
    print("📊 ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
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

def train_optimized_models(X_train, X_test, y_train, y_test):
    """Обучение оптимизированных моделей"""
    print("\n🚀 ОБУЧЕНИЕ ОПТИМИЗИРОВАННЫХ МОДЕЛЕЙ")
    print("=" * 50)
    
    models = {}
    training_times = {}
    
    # 1. XGBoost (CPU оптимизированный)
    print("1. XGBoost (оптимизированный)...")
    start_time = time.time()
    
    # Кодируем целевую переменную для XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    xgb_params = {
        'n_estimators': [500, 1000],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'random_state': [42]
    }
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    xgb_grid = GridSearchCV(
        xgb_model, 
        xgb_params, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1,
        verbose=1
    )
    
    xgb_grid.fit(X_train, y_train_encoded)
    models['XGBoost'] = xgb_grid.best_estimator_
    training_times['XGBoost'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {xgb_grid.best_params_}")
    print(f"   Лучший score: {xgb_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['XGBoost']:.2f} сек")
    
    # 2. LightGBM
    print("2. LightGBM...")
    start_time = time.time()
    
    lgb_params = {
        'n_estimators': [500, 1000],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'random_state': [42]
    }
    
    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    lgb_grid = GridSearchCV(
        lgb_model, 
        lgb_params, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1,
        verbose=1
    )
    
    lgb_grid.fit(X_train, y_train)
    models['LightGBM'] = lgb_grid.best_estimator_
    training_times['LightGBM'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {lgb_grid.best_params_}")
    print(f"   Лучший score: {lgb_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['LightGBM']:.2f} сек")
    
    # 3. Random Forest (оптимизированный)
    print("3. Random Forest (оптимизированный)...")
    start_time = time.time()
    
    rf_params = {
        'n_estimators': [300, 500, 1000],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    models['RandomForest'] = rf_grid.best_estimator_
    training_times['RandomForest'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {rf_grid.best_params_}")
    print(f"   Лучший score: {rf_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['RandomForest']:.2f} сек")
    
    # 4. Gradient Boosting
    print("4. Gradient Boosting...")
    start_time = time.time()
    
    gb_params = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [6, 8, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    models['GradientBoosting'] = gb_grid.best_estimator_
    training_times['GradientBoosting'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {gb_grid.best_params_}")
    print(f"   Лучший score: {gb_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['GradientBoosting']:.2f} сек")
    
    # 5. Extra Trees
    print("5. Extra Trees...")
    start_time = time.time()
    
    et_params = {
        'n_estimators': [300, 500, 1000],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    et = ExtraTreesClassifier(random_state=42, n_jobs=-1)
    et_grid = GridSearchCV(et, et_params, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    et_grid.fit(X_train, y_train)
    models['ExtraTrees'] = et_grid.best_estimator_
    training_times['ExtraTrees'] = time.time() - start_time
    
    print(f"   Лучшие параметры: {et_grid.best_params_}")
    print(f"   Лучший score: {et_grid.best_score_:.4f}")
    print(f"   Время обучения: {training_times['ExtraTrees']:.2f} сек")
    
    # 6. Ensemble (Voting Classifier)
    print("6. Ensemble (Voting)...")
    start_time = time.time()
    
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', models['XGBoost']),
            ('lgb', models['LightGBM']),
            ('rf', models['RandomForest']),
            ('gb', models['GradientBoosting']),
            ('et', models['ExtraTrees'])
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    models['Ensemble'] = voting_clf
    training_times['Ensemble'] = time.time() - start_time
    
    print(f"   Время обучения: {training_times['Ensemble']:.2f} сек")
    
    return models, training_times, le

def evaluate_models(models, X_test, y_test, training_times, le):
    """Оценка всех моделей"""
    print("\n📊 ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 30)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Предсказания
        start_time = time.time()
        if name == 'XGBoost':
            y_pred_encoded = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_encoded)
            y_pred_proba = model.predict_proba(X_test)
        else:
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
        if name == 'XGBoost':
            cv_scores = cross_val_score(model, X_test, le.transform(y_test), cv=5, scoring='f1_weighted')
        else:
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_weighted')
        
        print(f"   CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def save_best_model(models, results, selected_features, le):
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
    model_path = '../../models/wine_quality_optimized_model.pkl'
    joblib.dump(best_model, model_path)
    
    # Сохраняем label encoder для XGBoost
    if best_model_name == 'XGBoost':
        le_path = '../../models/label_encoder.pkl'
        joblib.dump(le, le_path)
        print(f"Label encoder сохранен в {le_path}")
    
    # Сохраняем информацию о признаках
    features_path = '../../models/optimized_selected_features.pkl'
    joblib.dump(selected_features, features_path)
    
    # Сохраняем метрики
    metrics_path = '../../models/optimized_model_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Optimized Wine Quality Model\n")
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
    print("🚀 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ МОДЕЛИ КАЧЕСТВА ВИНА")
    print("=" * 60)
    
    # 1. Загрузка и анализ данных
    df = load_and_analyze_data()
    
    # 2. Продвинутая предобработка
    df_processed = advanced_preprocessing(df)
    
    # 3. Подготовка данных
    X = df_processed.drop('quality', axis=1)
    y = df_processed['quality']
    
    # 4. Отбор признаков
    X_selected, selected_features = feature_selection(X, y)
    
    # 5. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 25)
    print(f"Train set: {X_train.shape[0]} образцов")
    print(f"Test set: {X_test.shape[0]} образцов")
    print(f"Признаков: {X_train.shape[1]}")
    
    # 6. Обучение оптимизированных моделей
    models, training_times, le = train_optimized_models(X_train, X_test, y_train, y_test)
    
    # 7. Оценка моделей
    results = evaluate_models(models, X_test, y_test, training_times, le)
    
    # 8. Сохранение лучшей модели
    best_model_name, best_model = save_best_model(models, results, selected_features, le)
    
    print(f"\n🎉 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print(f"Лучшая модель: {best_model_name}")
    print(f"F1-score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Сравнение с базовой моделью
    baseline_accuracy = 0.6454  # Из предыдущей модели
    improvement = (results[best_model_name]['accuracy'] - baseline_accuracy) / baseline_accuracy * 100
    print(f"Улучшение точности: {improvement:.1f}%")

if __name__ == "__main__":
    main()

