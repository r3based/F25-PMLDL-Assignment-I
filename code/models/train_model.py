"""
Скрипт для обучения модели на датасете Wine Quality
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_preprocess_data():
    """Загружает и предобрабатывает данные"""
    
    # Загружаем данные
    df = pd.read_csv('../../data/raw/wine_quality.csv')
    
    # Разделяем на признаки и целевую переменную
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Переименовываем колонки для удобства
    X.columns = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'ph', 'sulphates', 'alcohol'
    ]
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Обучает модель Random Forest"""
    
    # Создаем и обучаем модель
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Оценивает модель"""
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy

def save_model(model, accuracy):
    """Сохраняет модель"""
    
    # Создаем директорию если не существует
    os.makedirs('../../models', exist_ok=True)
    
    # Сохраняем модель
    model_path = '../../models/wine_quality_model.pkl'
    joblib.dump(model, model_path)
    
    # Сохраняем метрики
    metrics_path = '../../models/model_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    
    print(f"Модель сохранена в {model_path}")
    print(f"Метрики сохранены в {metrics_path}")

def main():
    """Основная функция"""
    
    print("Загружаем и предобрабатываем данные...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Обучаем модель...")
    model = train_model(X_train, y_train)
    
    print("Оцениваем модель...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    print("Сохраняем модель...")
    save_model(model, accuracy)
    
    print("Обучение завершено!")

if __name__ == "__main__":
    main()
