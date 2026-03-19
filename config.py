from pathlib import Path
import json
from typing import Union
import os
import logging


class Config:
    # Overall
    ROOT_PATH: str = Path(__file__).resolve().parent

    # Folders
    RAW_DATA_PATH: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/RawData"
    CLEANED_DATA_PATH: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/DataCleaned"
    TRAIN_IMAGES_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/images/train"
    VAL_IMAGES_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/images/val"
    TRAIN_LABELS_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/labels/train"
    VAL_LABELS_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/train_data/labels/val"
    TEST_DATA_FOLDER: Union[str, os.PathLike, Path] = fr"{ROOT_PATH}/TestData"
    VIDEOS_FOLDER: Union[str, os.PathLike, Path] =  fr"{ROOT_PATH}\Videos"

    # OVERALL SYSTEM PARAMS
    AREAS_FOLDER: Union[str, os.PathLike, Path] = rf"{ROOT_PATH}\areas"
    ALPHA: float = 0.6

    # SORT
    SORT_MAX_AGE: int = 50
    SORT_MIN_HITS: int = 1
    SORT_IOU_THRESHOLD: float = .3

    # SAHI MODEL PARAMS
    USE_SAHI: bool = False
    SAHI_CONF_THRESH: float = .2
    SAHI_SLICE_HEIGHT: int = 480
    SAHI_SLICE_WIDTH: int = 480
    SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2
    SAHI_OVERLAP_WIDTH_RATIO: float = 0.2

    # YOLO Model
    YOLO_MODELS_FOLDER_PATH: str = f"{ROOT_PATH}/models/yolo_models/"
    YOLO_MODEL_PATH: str = f"{YOLO_MODELS_FOLDER_PATH}/best2.pt"
    YOLO_CLASSES_FILE: str = f"{YOLO_MODELS_FOLDER_PATH}/classes.txt"
    YOLO_DEVICE: str = "cpu"
    YOLO_IOU: float = .2
    YOLO_CONF_THRESH: float = .2
    YOLO_AUGMENT: bool = True
    YOLO_AGNOSTIC_NMS: bool = True

    # ZONE MODEL
    ZONE_MODELS_FOLDER_PATH: str = f"{ROOT_PATH}/models/zone_model/"
    ZONE_MODEL_PATH: str = f"{ZONE_MODELS_FOLDER_PATH}/zone_model.h5"
    ZONE_MODEL_SEQ_LEN: int = 30

    # API
    API_PORT: int = 5000
    API_HOST: str = "127.0.0.1"
    MAX_IMAGE_FILES: int = 5
    API_LOG_FILE: str = f"{ROOT_PATH}/logs/api_logs.log"

    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = True
    WEB_APP_LOG_FILE: str = f"{ROOT_PATH}/logs/web_app.logs"
    WEB_APP_TEMP_UPLOADS_FOLDER = f"{ROOT_PATH}/ocr_webapp/static/temp_uploads"
    WEB_APP_FILES_LIFE_TIME: int = 300
    WEB_APP_USE_SSL: bool = False
    WEB_APP_SSL_FOLDER: str = f"{ROOT_PATH}/ocr_webapp/ssl_cert"
    WEB_APP_TESTING: bool = False
    WEB_APP_LOG_LEVEL: int = logging.DEBUG

    # Dataset filtering
    MIN_HEIGHT: int = 150
    MIN_WIDTH: int = 150
    MAX_HEIGHT: int = 5000
    MAX_WIDTH: int = 5000
    ALLOWED_EXTENSIONS: tuple = (".jpg", ".png", ".jpeg")

    # LOGGER
    UVICORN_LOG_CONFIG_PATH: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/ocr_api/uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] = f"{Config.ROOT_PATH}/logs/api_logs.log"
            return log_config