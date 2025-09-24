"""
Streamlit приложение для предсказания качества вина
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Заголовок
st.title("🍷 Предсказание Качества Вина")
st.markdown("---")

# URL API (будет работать в Docker)
API_URL = "http://api:8000"

# Функция для проверки здоровья API
def check_api_health():
    """Проверяет доступность API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Функция для получения предсказания
def get_prediction(features):
    """Отправляет запрос к API и получает предсказание"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None

# Основной интерфейс
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Введите характеристики вина")
    
    # Создаем форму для ввода данных
    with st.form("wine_features_form"):
        # Разделяем поля на две колонки
        col_left, col_right = st.columns(2)
        
        with col_left:
            fixed_acidity = st.number_input(
                "Фиксированная кислотность (g/dm³)",
                min_value=0.0,
                max_value=20.0,
                value=7.0,
                step=0.1,
                help="Количество фиксированных кислот в вине"
            )
            
            volatile_acidity = st.number_input(
                "Летучая кислотность (g/dm³)",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.01,
                help="Количество летучих кислот"
            )
            
            citric_acid = st.number_input(
                "Лимонная кислота (g/dm³)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="Количество лимонной кислоты"
            )
            
            residual_sugar = st.number_input(
                "Остаточный сахар (g/dm³)",
                min_value=0.0,
                max_value=20.0,
                value=2.0,
                step=0.1,
                help="Количество сахара после ферментации"
            )
            
            chlorides = st.number_input(
                "Хлориды (g/dm³)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.001,
                help="Количество хлоридов"
            )
            
            free_sulfur_dioxide = st.number_input(
                "Свободный диоксид серы (mg/dm³)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                help="Количество свободного SO2"
            )
        
        with col_right:
            total_sulfur_dioxide = st.number_input(
                "Общий диоксид серы (mg/dm³)",
                min_value=0.0,
                max_value=300.0,
                value=50.0,
                step=1.0,
                help="Общее количество SO2"
            )
            
            density = st.number_input(
                "Плотность (g/cm³)",
                min_value=0.9,
                max_value=1.1,
                value=0.997,
                step=0.001,
                help="Плотность вина"
            )
            
            ph = st.number_input(
                "pH",
                min_value=2.0,
                max_value=5.0,
                value=3.2,
                step=0.01,
                help="Кислотность вина"
            )
            
            sulphates = st.number_input(
                "Сульфаты (g/dm³)",
                min_value=0.0,
                max_value=2.0,
                value=0.6,
                step=0.01,
                help="Количество сульфатов"
            )
            
            alcohol = st.number_input(
                "Алкоголь (% vol)",
                min_value=8.0,
                max_value=15.0,
                value=10.5,
                step=0.1,
                help="Содержание алкоголя"
            )
        
        # Кнопка для предсказания
        submitted = st.form_submit_button("🔮 Предсказать качество вина", use_container_width=True)

with col2:
    st.header("📈 Результат")
    
    # Проверяем доступность API
    if check_api_health():
        st.success("✅ API доступен")
    else:
        st.error("❌ API недоступен")
        st.stop()
    
    # Обрабатываем предсказание
    if submitted:
        # Подготавливаем данные для API
        features = {
            "fixed_acidity": fixed_acidity,
            "volatile_acidity": volatile_acidity,
            "citric_acid": citric_acid,
            "residual_sugar": residual_sugar,
            "chlorides": chlorides,
            "free_sulfur_dioxide": free_sulfur_dioxide,
            "total_sulfur_dioxide": total_sulfur_dioxide,
            "density": density,
            "ph": ph,
            "sulphates": sulphates,
            "alcohol": alcohol
        }
        
        # Показываем спиннер во время запроса
        with st.spinner("Анализируем характеристики вина..."):
            result = get_prediction(features)
        
        if result:
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            # Отображаем результат
            st.markdown("### 🎯 Предсказание")
            
            # Цветовая индикация качества
            if prediction >= 7:
                st.success(f"**Качество: {prediction}/10** ⭐")
                st.success("Отличное вино!")
            elif prediction >= 5:
                st.warning(f"**Качество: {prediction}/10** ⭐")
                st.warning("Хорошее вино")
            else:
                st.error(f"**Качество: {prediction}/10** ⭐")
                st.error("Требует улучшения")
            
            # Уверенность модели
            st.markdown("### 📊 Уверенность модели")
            confidence_percent = confidence * 100
            st.progress(confidence)
            st.write(f"{confidence_percent:.1f}%")
            
            # Дополнительная информация
            st.markdown("### ℹ️ Дополнительная информация")
            st.info(f"Модель предсказывает качество вина по шкале от 0 до 10, где 10 - наивысшее качество.")

# Информация о модели
st.markdown("---")
st.header("🤖 О модели")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **Алгоритм:** Random Forest Classifier
    
    **Датасет:** Wine Quality Dataset
    
    **Признаки:**
    - Физико-химические свойства вина
    - 11 числовых характеристик
    """)

with col_info2:
    st.markdown("""
    **Цель:** Предсказание качества вина
    
    **Шкала качества:** 0-10 баллов
    
    **Точность модели:** ~65-70%
    """)

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🍷 Wine Quality Prediction App | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
