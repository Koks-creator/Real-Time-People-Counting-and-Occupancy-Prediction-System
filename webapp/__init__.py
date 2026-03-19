import sys
from pathlib import Path
import logging
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))
from glob import glob
import json
from dotenv import load_dotenv
from flask import Flask

from config import Config
from area_monitor import AreaMonitorSystem

load_dotenv()

def setup_logging(app: Flask) -> None:
    """Configure logging for the application"""
    log_dir = os.path.dirname(Config.WEB_APP_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(Config.WEB_APP_LOG_FILE)
    file_handler.setLevel(Config.WEB_APP_LOG_LEVEL)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(Config.WEB_APP_LOG_LEVEL)
    app.logger.propagate = False  # Prevent duplicate logs

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
setup_logging(app)
app.logger.info("Starting web app")
# by zrobic to pod wiele sesji, by nalezalo wyciagnac instancje yolo_detector na zewnatrz
# by ladowac model tylko raz i robic instancje area_monitora per sesja
area_monitor_tool = AreaMonitorSystem()

video_area_mapping = {}
webcam_area_mapping = {}

areas_files = [os.path.split(f_path)[1].replace(".json", "") for f_path in glob(f"{Config.AREAS_FOLDER}/*.*")]
videos_files = [os.path.split(f_path)[1] for f_path in glob(f"{Config.VIDEOS_FOLDER}/*.*")]
for ar in areas_files:
    if "cam" in ar:
        _, cam_id = ar.split("_")
        webcam_area_mapping[cam_id] = str(Path(Config.AREAS_FOLDER) / f"{ar}.json")
    for video_f in videos_files:
        file_str = os.path.splitext(video_f)[0]
        if file_str == ar:
            video_area_mapping[str(Path(Config.VIDEOS_FOLDER) / video_f)] = str(Path(Config.AREAS_FOLDER) / f"{ar}.json")

app.logger.info(json.dumps(video_area_mapping, indent=4))
app.logger.info(json.dumps(webcam_area_mapping, indent=4))
from webapp import routes