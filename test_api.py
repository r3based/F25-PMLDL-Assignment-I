"""
Скрипт для тестирования API без Docker
"""
import requests
import json

# URL API (локальный запуск)
API_URL = "http://localhost:8000"

def test_api():
    """Тестирует все эндпоинты API"""
    
    print("🧪 Тестирование Wine Quality Prediction API")
    print("=" * 50)
    
    # Тест 1: Проверка здоровья API
    print("\n1. Проверка здоровья API...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API здоров")
            print(f"   Ответ: {response.json()}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return
    
    # Тест 2: Информация о модели
    print("\n2. Получение информации о модели...")
    try:
        response = requests.get(f"{API_URL}/model_info")
        if response.status_code == 200:
            print("✅ Информация о модели получена")
            model_info = response.json()
            print(f"   Тип модели: {model_info['model_type']}")
            print(f"   Количество деревьев: {model_info['n_estimators']}")
            print(f"   Максимальная глубина: {model_info['max_depth']}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 3: Предсказание
    print("\n3. Тестирование предсказания...")
    
    # Пример данных для предсказания
    test_data = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.3,
        "citric_acid": 0.3,
        "residual_sugar": 2.0,
        "chlorides": 0.05,
        "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 50.0,
        "density": 0.997,
        "ph": 3.2,
        "sulphates": 0.6,
        "alcohol": 10.5
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Предсказание успешно")
            print(f"   Качество вина: {result['prediction']}/10")
            print(f"   Уверенность: {result['confidence']:.2%}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"   Ответ: {response.text}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 4: Несколько предсказаний
    print("\n4. Тестирование нескольких предсказаний...")
    
    test_cases = [
        {
            "name": "Хорошее вино",
            "data": {
                "fixed_acidity": 8.0,
                "volatile_acidity": 0.2,
                "citric_acid": 0.4,
                "residual_sugar": 1.5,
                "chlorides": 0.04,
                "free_sulfur_dioxide": 20.0,
                "total_sulfur_dioxide": 60.0,
                "density": 0.995,
                "ph": 3.3,
                "sulphates": 0.7,
                "alcohol": 12.0
            }
        },
        {
            "name": "Среднее вино",
            "data": {
                "fixed_acidity": 6.5,
                "volatile_acidity": 0.4,
                "citric_acid": 0.2,
                "residual_sugar": 3.0,
                "chlorides": 0.06,
                "free_sulfur_dioxide": 10.0,
                "total_sulfur_dioxide": 40.0,
                "density": 0.998,
                "ph": 3.1,
                "sulphates": 0.5,
                "alcohol": 9.5
            }
        }
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   {test_case['name']}: {result['prediction']}/10 (уверенность: {result['confidence']:.2%})")
            else:
                print(f"   {test_case['name']}: ❌ Ошибка {response.status_code}")
        except Exception as e:
            print(f"   {test_case['name']}: ❌ Ошибка {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_api()
