import os

from flask import Flask
from app.config import Config
from app.routes import init_routes


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'data', 'userTS')


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app = create_app()

    init_routes(app)

    return app
