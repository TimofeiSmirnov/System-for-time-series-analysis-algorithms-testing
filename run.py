import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
