"""
Скрипт для запуска Airflow с настройками MLOps пайплайна
"""
import os
import subprocess
import sys
from pathlib import Path

def setup_airflow():
    """Настраивает Airflow для MLOps пайплайна"""
    
    # Устанавливаем переменные окружения
    os.environ['AIRFLOW_HOME'] = str(Path(__file__).parent)
    os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = str(Path(__file__).parent / 'dags')
    os.environ['AIRFLOW__CORE__BASE_LOG_FOLDER'] = str(Path(__file__).parent / 'logs')
    os.environ['AIRFLOW__CORE__EXECUTOR'] = 'LocalExecutor'
    os.environ['AIRFLOW__CORE__SQL_ALCHEMY_CONN'] = f'sqlite:///{Path(__file__).parent}/airflow.db'
    os.environ['AIRFLOW__WEBSERVER__WEB_SERVER_PORT'] = '8080'
    os.environ['AIRFLOW__WEBSERVER__WEB_SERVER_HOST'] = '0.0.0.0'
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    
    print("🔧 Настройка Airflow для MLOps пайплайна...")
    print(f"📁 AIRFLOW_HOME: {os.environ['AIRFLOW_HOME']}")
    print(f"📁 DAGS_FOLDER: {os.environ['AIRFLOW__CORE__DAGS_FOLDER']}")
    
    # Создаем необходимые директории
    Path(__file__).parent.mkdir(exist_ok=True)
    (Path(__file__).parent / 'dags').mkdir(exist_ok=True)
    (Path(__file__).parent / 'logs').mkdir(exist_ok=True)
    
    return True

def init_airflow():
    """Инициализирует базу данных Airflow"""
    try:
        print("🗄️ Инициализация базы данных Airflow...")
        subprocess.run(['airflow', 'db', 'init'], check=True)
        print("✅ База данных инициализирована!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка инициализации базы данных: {e}")
        return False

def create_admin_user():
    """Создает администратора Airflow"""
    try:
        print("👤 Создание администратора...")
        subprocess.run([
            'airflow', 'users', 'create',
            '--username', 'admin',
            '--firstname', 'Admin',
            '--lastname', 'User',
            '--role', 'Admin',
            '--email', 'admin@example.com',
            '--password', 'admin'
        ], check=True)
        print("✅ Администратор создан!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Администратор уже существует или ошибка: {e}")
        return True  # Не критично

def start_airflow():
    """Запускает Airflow"""
    print("🚀 Запуск Airflow...")
    print("📊 Веб-интерфейс будет доступен по адресу: http://localhost:8080")
    print("👤 Логин: admin, Пароль: admin")
    print("🔄 MLOps пайплайн будет запускаться каждые 5 минут")
    print("⏹️ Для остановки нажмите Ctrl+C")
    
    try:
        # Запускаем планировщик в фоне
        scheduler_process = subprocess.Popen(['airflow', 'scheduler'])
        
        # Запускаем веб-сервер
        webserver_process = subprocess.Popen(['airflow', 'webserver'])
        
        # Ждем завершения
        scheduler_process.wait()
        webserver_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Остановка Airflow...")
        scheduler_process.terminate()
        webserver_process.terminate()
        print("✅ Airflow остановлен!")

def main():
    """Основная функция"""
    print("🍷 Wine Quality MLOps Pipeline - Airflow Setup")
    print("=" * 50)
    
    if not setup_airflow():
        print("❌ Ошибка настройки Airflow")
        sys.exit(1)
    
    if not init_airflow():
        print("❌ Ошибка инициализации Airflow")
        sys.exit(1)
    
    if not create_admin_user():
        print("❌ Ошибка создания администратора")
        sys.exit(1)
    
    start_airflow()

if __name__ == "__main__":
    main()
