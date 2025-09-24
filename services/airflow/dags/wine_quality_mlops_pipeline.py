"""
Airflow DAG для автоматизированного MLOps пайплайна Wine Quality Prediction
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os

# Параметры по умолчанию
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'wine_quality_mlops_pipeline',
    default_args=default_args,
    description='Автоматизированный MLOps пайплайн для предсказания качества вина',
    schedule_interval='*/5 * * * *',
    catchup=False,
    tags=['mlops', 'wine_quality', 'mlflow', 'dvc'],
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))

def check_data_freshness():
    import os
    from datetime import datetime, timedelta
    
    data_path = os.path.join(PROJECT_ROOT, 'data/raw/wine_quality.csv')
    if os.path.exists(data_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
        if datetime.now() - mod_time < timedelta(hours=24):
            print("Данные свежие, пропускаем загрузку")
            return False
    print("Данные устарели, требуется обновление")
    return True

# Этап 1: Data Engineering
data_engineering = BashOperator(
    task_id='data_engineering',
    bash_command=f"""
    cd {PROJECT_ROOT} && \
    echo "=== Этап 1: Data Engineering ===" && \
    dvc repro process_data
    """,
    dag=dag,
)

# Этап 2: Model Engineering
model_engineering = BashOperator(
    task_id='model_engineering',
    bash_command=f"""
    cd {PROJECT_ROOT} && \
    echo "=== Этап 2: Model Engineering ===" && \
    dvc repro train_model
    """,
    dag=dag,
)

# Этап 3: Deployment
deployment = BashOperator(
    task_id='deployment',
    bash_command=f"""
    cd {PROJECT_ROOT}/code/deployment && \
    echo "=== Этап 3: Deployment ===" && \
    docker-compose down || true && \
    docker-compose up --build -d
    """,
    dag=dag,
)

# Задача проверки здоровья API
health_check = BashOperator(
    task_id='health_check',
    bash_command="""
    echo "Проверяем здоровье API..." && \
    sleep 30 && \
    curl -f http://localhost:8000/health || exit 1
    """,
    dag=dag,
)

# Логирование результатов
log_results = PythonOperator(
    task_id='log_results',
    python_callable=lambda: print("=== MLOps пайплайн завершен успешно! ==="),
    dag=dag,
)

# Определяем зависимости
data_engineering >> model_engineering >> deployment >> health_check >> log_results
