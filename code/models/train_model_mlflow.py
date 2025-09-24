"""
Скрипт для обучения модели с логированием в MLflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

def train_model_with_mlflow():
    """Обучает модель с логированием в MLflow"""
    
    # Настраиваем MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("wine_quality_prediction")
    
    with mlflow.start_run(run_name=f"wine_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Загружаем обработанные данные
        print("Загружаем обработанные данные...")
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
        y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Параметры модели
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Логируем параметры
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Создаем и обучаем модель
        print("Обучаем модель...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        
        # Детальный отчет классификации
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Логируем метрики для каждого класса
        for class_label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name != 'support':
                        mlflow.log_metric(f"{class_label}_{metric_name}", value)
        
        # Логируем матрицу ошибок как артефакт
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=[f"true_{i}" for i in range(len(cm))], 
                            columns=[f"pred_{i}" for i in range(len(cm))])
        cm_path = "models/confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Сохраняем модель
        model_path = "models/wine_quality_model_mlflow.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        
        # Логируем модель
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="WineQualityModel"
        )
        
        # Логируем артефакты
        mlflow.log_artifact(model_path)
        
        # Сохраняем метрики в файл
        metrics_path = "models/model_metrics_mlflow.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"Model: Random Forest Classifier\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Training Date: {datetime.now()}\n")
            f.write(f"Train Size: {len(X_train)}\n")
            f.write(f"Test Size: {len(X_test)}\n")
        
        mlflow.log_artifact(metrics_path)
        
        print(f"Модель сохранена в {model_path}")
        print(f"Метрики сохранены в {metrics_path}")
        print("Эксперимент записан в MLflow!")
        
        return model, accuracy

if __name__ == "__main__":
    train_model_with_mlflow()
