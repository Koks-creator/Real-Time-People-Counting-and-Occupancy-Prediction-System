from dataclasses import dataclass
from typing import Union, List, Tuple
from pathlib import Path
import numpy as np
from tensorflow import keras

from custom_decorators import timeit, log_call
from custom_logger import CustomLogger
from config import Config


logger = CustomLogger(logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL,
                      log_file_name=fr"{Config.ROOT_PATH}/logs/zone_predictor_logs.log"
                      ).create_logger()


@dataclass
class ZonePredictor:
    model_path: Union[str, Path]
    _horizons: Tuple[int] = (30, 60)

    def __post_init__(self) -> None:
        logger.info("Initializing ZonePredictor: \n" \
        f"- {self.model_path=}\n"
        )
        self.model = keras.models.load_model('zone_model.h5', compile=False)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("Model loaded")

    def predict_zone(self, recent_counts: list, duration_s: float, horizon_s: int, capacity: int) -> dict:
        n = len(recent_counts)
        seq = np.array(
            [[i / (n - 1), c / capacity] for i, c in enumerate(recent_counts)],
            dtype=np.float32
        )[np.newaxis, ...]

        meta = np.array([[duration_s / 600, horizon_s / 600, capacity / 200]], dtype=np.float32)

        pred = float(self.model.predict({'sequence': seq, 'metadata': meta}, verbose=0)[0, 0]) * capacity
        return {
            'predicted_count': round(pred),
            'occupancy_pct':   round(pred / capacity * 100, 1),
            'horizon_s':       horizon_s,
        }
    
    @log_call(logger=logger, log_params=["duration_s", "capacity"], hide_res=True, log_debug=True)
    @timeit(logger=logger)
    def get_zone_predictions(self, capacity: int, duration_s: int, recent_counts: List) -> List[dict]:
        res = []
        for horizon in self._horizons:
            res.append(
                self.predict_zone(capacity=capacity,
                                  duration_s=duration_s,
                                  recent_counts=recent_counts,
                                  horizon_s=horizon
                                  )
            )
        return res
    

if __name__ == "__main__":
    zone_pred = ZonePredictor(
        model_path="zone_model.h5"
    )
    counts_down = [20, 20, 19, 19, 18, 17, 17, 16, 15, 15,
               14, 13, 13, 12, 11, 11, 10, 10, 9, 8,
               8, 7, 7, 6, 5, 5, 4, 4, 3, 3]
    
    x = zone_pred.get_zone_predictions(
        capacity=36,
        duration_s=len(counts_down) * 2.54,
        recent_counts=counts_down
    )
    print(x)