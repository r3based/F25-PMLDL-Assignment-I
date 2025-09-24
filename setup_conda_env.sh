#!/bin/bash

# 🍷 Wine Quality MLOps Pipeline - Автоматическая настройка Conda окружения
# Этот скрипт создает и настраивает виртуальное окружение для проекта

set -e  # Остановка при ошибке

echo "🍷 Wine Quality MLOps Pipeline - Настройка Conda окружения"
echo "=========================================================="

# Проверяем наличие conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda не найден. Установите Anaconda или Miniconda."
    exit 1
fi

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не найден. Установите Docker."
    exit 1
fi

echo "✅ Conda и Docker найдены"

# Имя окружения
ENV_NAME="wine_mlops"

# Проверяем, существует ли окружение
if conda env list | grep -q $ENV_NAME; then
    echo "⚠️  Окружение '$ENV_NAME' уже существует."
    read -p "Удалить и пересоздать? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Удаляем существующее окружение..."
        conda env remove -n $ENV_NAME -y
    else
        echo "📝 Активируем существующее окружение..."
        conda activate $ENV_NAME
        echo "✅ Окружение активировано!"
        exit 0
    fi
fi

# Создаем новое окружение
echo "🔧 Создаем виртуальное окружение '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.9 -y

# Активируем окружение
echo "⚡ Активируем окружение..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Проверяем активацию
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "❌ Не удалось активировать окружение"
    exit 1
fi

echo "✅ Окружение '$ENV_NAME' активировано!"

# Устанавливаем зависимости
echo "📦 Устанавливаем зависимости..."
pip install -r requirements.txt

# Проверяем установку
echo "🔍 Проверяем установку..."
python --version
echo "✅ Python версия: $(python --version)"

# Проверяем ключевые пакеты
echo "🔍 Проверяем ключевые пакеты..."
python -c "import fastapi; print('✅ FastAPI:', fastapi.__version__)"
python -c "import streamlit; print('✅ Streamlit:', streamlit.__version__)"
python -c "import dvc; print('✅ DVC:', dvc.__version__)"
python -c "import mlflow; print('✅ MLflow:', mlflow.__version__)"
python -c "import airflow; print('✅ Airflow:', airflow.__version__)"

# Инициализируем DVC
echo "🔧 Инициализируем DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "✅ DVC инициализирован"
else
    echo "✅ DVC уже инициализирован"
fi

# Создаем необходимые директории
echo "📁 Создаем необходимые директории..."
mkdir -p data/raw data/processed models services/airflow/dags services/airflow/logs

# Загружаем данные
echo "📊 Загружаем данные..."
cd code/datasets
python download_data.py
cd ../..

# Обучаем модель
echo "🤖 Обучаем модель..."
cd code/models
python train_model.py
cd ../..

# Запускаем DVC пайплайн
echo "🔄 Запускаем DVC пайплайн..."
dvc repro

echo ""
echo "🎉 Настройка завершена успешно!"
echo "=========================================================="
echo "📋 Следующие шаги:"
echo "1. Активируйте окружение: conda activate $ENV_NAME"
echo "2. Запустите Docker: cd code/deployment && sudo docker-compose up --build -d"
echo "3. Откройте веб-приложение: http://localhost:8501"
echo "4. Откройте API документацию: http://localhost:8000/docs"
echo "5. Запустите Airflow: cd services/airflow && python start_airflow.py"
echo ""
echo "🌐 Веб-интерфейсы:"
echo "   • Streamlit приложение: http://localhost:8501"
echo "   • FastAPI документация: http://localhost:8000/docs"
echo "   • Airflow UI: http://localhost:8080 (admin/admin)"
echo "   • MLflow UI: mlflow ui (запустите в отдельном терминале)"
echo ""
echo "📚 Документация:"
echo "   • README_CONDA_SETUP.md - Подробная инструкция"
echo "   • README_MLOPS_PIPELINE.md - Описание MLOps пайплайна"
echo ""
echo "✅ Готово к работе!"
