import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")