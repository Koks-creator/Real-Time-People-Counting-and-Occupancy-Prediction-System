import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from webapp import app
from config import Config


if __name__ == '__main__':
    app.run(host=Config.WEB_APP_HOST,
            port=Config.WEB_APP_PORT,
            debug=Config.WEB_APP_DEBUG)