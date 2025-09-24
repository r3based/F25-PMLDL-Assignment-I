"""
Скрипт для загрузки датасета Wine Quality
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os

def download_wine_data():
    """Загружает датасет Wine Quality и сохраняет в data/raw"""
    
    # Создаем директорию если не существует
    os.makedirs('../../data/raw', exist_ok=True)
    
    # Загружаем датасет Wine Quality
    print("Загружаем датасет Wine Quality...")
    wine_data = fetch_openml('wine_quality', version=1, as_frame=True)
    
    # Получаем данные и целевые значения
    X = wine_data.data
    y = wine_data.target
    
    # Объединяем в один DataFrame
    df = pd.concat([X, y], axis=1)
    
    # Сохраняем в CSV
    df.to_csv('../../data/raw/wine_quality.csv', index=False)
    print(f"Датасет сохранен в ../../data/raw/wine_quality.csv")
    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    download_wine_data()
