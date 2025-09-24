# Используем свежий Python 3.11 slim
FROM python:3.11-slim

WORKDIR /app

# Копируем requirements.txt
COPY code/deployment/api/requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем код API
COPY code/deployment/api/main.py .

# Копируем обученную модель
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
