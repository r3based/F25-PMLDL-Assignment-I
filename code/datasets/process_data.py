"""
DVC пайплайн для обработки данных Wine Quality
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

def process_data():
    """Обрабатывает данные: загрузка, очистка, разделение"""
    
    # Загружаем данные
    print("Загружаем данные...")
    df = pd.read_csv('data/raw/wine_quality.csv')
    print(f"Исходный размер данных: {df.shape}")
    
    # Очистка данных
    print("Очищаем данные...")
    
    # Проверяем на пропущенные значения
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Найдены пропущенные значения: {missing_values[missing_values > 0]}")
        # Заполняем пропущенные значения медианой
        for col in missing_values[missing_values > 0].index:
            df[col] = df[col].fillna(df[col].median())
    
    # Удаляем выбросы (используем IQR метод)
    print("Удаляем выбросы...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col != 'quality':  # Не удаляем выбросы в целевой переменной
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"Размер данных после очистки: {df.shape}")
    
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
    print("Разделяем данные на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Создаем директории если не существуют
    os.makedirs('data/processed', exist_ok=True)
    
    # Сохраняем обработанные данные
    print("Сохраняем обработанные данные...")
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Обработка данных завершена!")
    print(f"Train set: {X_train.shape[0]} образцов")
    print(f"Test set: {X_test.shape[0]} образцов")

if __name__ == "__main__":
    process_data()
