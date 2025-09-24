# Используем свежий Python 3.11 slim
FROM python:3.11-slim

WORKDIR /app

# Копируем requirements.txt
COPY code/deployment/app/requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY code/deployment/app/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
