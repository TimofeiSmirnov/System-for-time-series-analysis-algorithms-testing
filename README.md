# Система для тестирования алгоритмов анализа временных рядов

Flask-приложение для анализа и визуализации временных рядов.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ваш-username/ваш-репозиторий.git
   cd ваш-репозиторий
   ```
2. Создайте виртуальное окружение (опционально):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Создайте файл app/config.py со следующим наполнением:
5. ```bash
   import os
   class Config:
       SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
       UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
   ```

## Запуск
1. Запустите приложение:
   ```bash
   python run.py
   ```
2. Откройте в браузере:
   ```bash
   http://127.0.0.1:5000
   ```

## Лицензия


---

### Итог

Надеюсь, вам был полезен данный файлик 😊
