from time import time, sleep
from dataclasses import dataclass
from typing import Union, List, Generator, Tuple
from pathlib import Path
import numpy as np
import cv2
import json
from collections import deque, defaultdict
import csv

from yolo_detector import YoloDetector
from zone_predictor import ZonePredictor
from custom_decorators import timeit, log_call
from custom_logger import CustomLogger
from config import Config
from sort_tracker import Sort

logger = CustomLogger(logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL,
                      log_file_name=fr"{Config.ROOT_PATH}/logs/area_monitor_logs.log"
                      ).create_logger()


@dataclass(frozen=True)
class FrameDetectionData:
    original_frame: np.ndarray
    detection_frame: np.ndarray
    detections_data: Generator


@dataclass
class AreaMonitorSystem():
    yolo_model_path: Union[str, Path] = Config.YOLO_MODEL_PATH
    classes_path: Union[str, Path] = Config.YOLO_CLASSES_FILE
    device: str = Config.YOLO_DEVICE  # Domyślnie "auto", może być też "cpu", "cuda:0" itp.
    sort_max_age: int = Config.SORT_MAX_AGE
    sort_min_hits: int = Config.SORT_MIN_HITS
    sort_iou_threshold: float = Config.SORT_IOU_THRESHOLD
    zone_model_path: Union[str, Path] = Config.ZONE_MODEL_PATH
    _seq_len: int = Config.ZONE_MODEL_SEQ_LEN

    def __post_init__(self) -> None:
        logger.info("Initializing AreaMonitorSystem: \n" \
        f"- {self.yolo_model_path=}\n"
        f"- {self.classes_path=}\n"
        f"- {self.device=}\n"
        f"- {self.sort_max_age=}\n"
        f"- {self.sort_min_hits=}\n"
        f"- {self.sort_iou_threshold=}\n"
        f"- {self.zone_model_path=}"
        )

        self.yolo_tool = YoloDetector(
            model_path=self.yolo_model_path,
            classes_path=self.classes_path,
            device=self.device
        )

        self.sorttr = Sort(
            max_age=self.sort_max_age,
            min_hits=self.sort_min_hits,
            iou_threshold=self.sort_iou_threshold
        )

        self.zone_pred = ZonePredictor(
            model_path=self.zone_model_path
        )

        with open(self.classes_path) as f:
            self.class_names = f.read().split("\n")

        self.areas = {}
        self._streaming = False
        self.funi_img = cv2.imread(fr"{Config.ROOT_PATH}\funi.png")

        
    def load_areas(self, areas_json_path: Union[str, Path]) -> dict:
        self.areas = {}
        
        with open(areas_json_path) as f:
            areas_conf = json.load(f)

        for area_name, area_items in areas_conf.items():
            self.areas[area_name] = {
                "area": area_items["area"],
                "direction": "down",
                "person_ids": [],
                "person_data": [],
                "count": 0,
                "occupancy_pct": 0.0,
                "pred": defaultdict(dict),
                "capacity": area_items["capacity"]
            }

        for area_name, area_items in self.areas.items():
            M = cv2.moments(np.array(area_items["area"]))
            if M['m00'] != 0:
                area_cx = int(M['m10'] / M['m00'])
                area_cy = int(M['m01'] / M['m00'])
                area_items["center"] = (area_cx, area_cy)
        
        self.history = {area: deque(maxlen=300) for area in self.areas}
    
    # @log_call(logger=logger, log_params=["conf", "iou", "augment", "agnostic_nms"], hide_res=True)
    @timeit(logger=logger)
    def yolo_detect(self,
                    images: List[np.ndarray],
                    conf: float = .2,
                    iou: float = .35,
                    augment: bool = True,
                    agnostic_nms: bool = True,
                    use_sahi: bool = False,
                    sahi_conf: float = 0.2, 
                    sahi_slice_height: int = 256, 
                    sahi_slice_width: int = 256, 
                    sahi_overlap_height_ratio: float = 0.2, 
                    sahi_overlap_width_ratio: float = 0.2
        ) -> Tuple[List[Generator], List[np.ndarray],  List[np.ndarray]]:
        if use_sahi:
            res = self.yolo_tool.detect_with_sahi(
                images=images,
                conf=sahi_conf, 
                slice_height=sahi_slice_height, 
                slice_width=sahi_slice_width, 
                overlap_height_ratio=sahi_overlap_height_ratio,
                overlap_width_ratio=sahi_overlap_width_ratio
            )
            detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
            detection_generators = [self.yolo_tool.yield_sahi_data(sahi_result=detection_res)
                                    for detection_res in detection_results]
        else:
            res = self.yolo_tool.detect(images=images,
                                            conf=conf,
                                            iou=iou,
                                            augment=augment,
                                            agnostic_nms=agnostic_nms)
            detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
            detection_generators = [self.yolo_tool.yield_data(bbox=detection_res) for detection_res in detection_results]

        return detection_generators, detection_frames, images

    def normalize_yolo_predictions(self, detection_generators: List[Generator],
                                   detection_frames: List[np.ndarray],
                                   images: List[np.ndarray]
                                   ) -> List[FrameDetectionData]:
        result = []
        for detection_generator, detection_frame, frame in zip(detection_generators, detection_frames, images):
            result.append(
                FrameDetectionData(
                    original_frame=frame,
                    detection_frame=detection_frame,
                    detections_data=detection_generator
                )
            )
        
        return result
    
    @log_call(logger=logger, log_params=[""], hide_res=True)
    @timeit(logger=logger)
    def set_object_ids(self, detections: List[FrameDetectionData]) -> List[np.ndarray]:
        total_track_data = []
        for detection_obj in detections:
            detection_gen = detection_obj.detections_data
            track_data = []
            for detection in detection_gen:
                class_id, _, conf, x1, y1, x2, y2 = *detection[:3], *detection[3]
                track_data.append([x1, y1, x2, y2, conf, class_id])
            
            updated_tracks = self.sorttr.update(track_data).astype(float)
            total_track_data.append(updated_tracks)
        
        return total_track_data
    
    def draw_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, class_name: str, obj_id: int) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 50, 80), 2, 1)
        cv2.putText(frame, f"{class_name} ID: {obj_id}", (x1, y1-15), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 50, 80), 2)
        cv2.putText(frame, f"{class_name}", (x1, y1-15), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 50, 80), 2)
    
    @staticmethod
    def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return bbox[0] + (abs(bbox[2]-bbox[0])//2), bbox[1] + (abs(bbox[3]-bbox[1])//2)

    def draw_areas_summary(self, image: np.ndarray) -> None:
        for area_name, area_items in self.areas.items():
            area_cx, area_cy = area_items["center"]
            cv2.putText(image, area_name, (area_cx, area_cy), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 50, 80), 2)
            cv2.putText(image, f"Count: {area_items['count']} - {area_items['occupancy_pct']}%",
                        (area_cx, area_cy + 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 50, 80), 2)
            
            start_y = area_cy + 50
            if area_items["pred"]:
                for horizont, pred_num in area_items["pred"].items():
                    cv2.putText(image, f"{horizont}: {pred_num['val']} - {pred_num['occupancy_pct']}%", (area_cx, start_y),
                                cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 50, 80), 2)
                    start_y += 25

    @staticmethod
    def area_check(area_points: np.array, bbox_center: Tuple[int, int]) -> bool:
        if cv2.pointPolygonTest(area_points, bbox_center, False) > -1:
            return True
        return False
    
    # @log_call(logger=logger, log_params=["conf", "iou", "augment", "agnostic_nms"], hide_res=True)
    @timeit(return_val=True)
    def process_images(self,
                       images: List[np.ndarray],
                       conf: float = .2,
                       iou: float = .35,
                       augment: bool = True,
                       agnostic_nms: bool = True,
                       alpha: float = 0.6,
                       use_sahi: bool = False,
                       sahi_conf: float = 0.2, 
                       sahi_slice_height: int = 480, 
                       sahi_slice_width: int = 480, 
                       sahi_overlap_height_ratio: float = 0.2, 
                       sahi_overlap_width_ratio: float = 0.2
            ) -> np.ndarray:
        detection_generators, detection_frames, images = self.yolo_detect(
            images=images,
            conf=conf,
            iou=iou,
            augment=augment,
            agnostic_nms=agnostic_nms,
            use_sahi=use_sahi,
            sahi_conf=sahi_conf,
            sahi_slice_height=sahi_slice_height,
            sahi_slice_width=sahi_slice_width,
            sahi_overlap_height_ratio=sahi_overlap_height_ratio,
            sahi_overlap_width_ratio=sahi_overlap_width_ratio
        )
        detections = self.normalize_yolo_predictions(
            detection_generators=detection_generators,
            detection_frames=detection_frames,
            images=images
        )
        
        total_track_data = self.set_object_ids(detections=detections)
        main_image = images[-1]
        overlay = images[-1].copy()
        for area_name, area_items in self.areas.items():
            area_points = area_items["area"]
            area_items["person_ids"] = []
            area_items["person_data"] = []

            cv2.polylines(overlay, [np.array(area_points)], True, (255, 150, 100), 4)
            cv2.polylines(overlay, [np.array(area_points)], True, (255, 200, 150), 4)

            for object_track_data in total_track_data:
                for track_data in object_track_data:
                    x1, y1, x2, y2, conf, obj_id, class_id = track_data
                    conf = round(track_data[-3], 1)
                    x1, y1, x2, y2, obj_id, class_id = int(x1), int(y1), int(x2), int(y2), int(obj_id), int(class_id)
                    class_name =  self.class_names[class_id]
                    person_cp = self.get_center(bbox=(x1, y1, x2, y2))


                    if self.area_check(area_points=np.array(area_points),
                                        bbox_center=person_cp):
                        self.draw_bbox(frame=main_image,
                                        x1=x1, y1=y1,
                                        x2=x2, y2=y2,
                                        class_name=class_name,
                                        obj_id=obj_id)

                        if obj_id not in area_items["person_ids"]:
                            area_items["person_ids"].append(obj_id)
                            area_items["person_data"].append({"person_id": obj_id, "cp": person_cp})
            area_items["count"] = len(area_items["person_data"])
            area_items["occupancy_pct"] = round(
                area_items["count"] / area_items["capacity"] * 100, 1
            )
    
        final_img = cv2.addWeighted(main_image, alpha, overlay, 1 - alpha, 0)

        return final_img

    def process_video(self, 
                      video_input: Union[int, str, Path],
                      conf: float = .2,
                      iou: float = .35, 
                      augment: bool = True, 
                      agnostic_nms: bool = True,
                      alpha: float = 0.6, 
                      target_interval: int = None, 
                      gather_data_file: Union[str, Path] = None,
                      use_sahi: bool = False,
                      sahi_conf: float = 0.2, 
                      sahi_slice_height: int = 480, 
                      sahi_slice_width: int = 480, 
                      sahi_overlap_height_ratio: float = 0.2, 
                      sahi_overlap_width_ratio: float = 0.2
            ) -> None:
        
        cap = cv2.VideoCapture(video_input)
        p_time = 0
        buffers = {area: deque(maxlen=self._seq_len) for area in self.areas} # czyli 30s, bo co 1s sprawdzamy

        if gather_data_file:
            csv_file = open(gather_data_file, "a", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp", "zone", "count"])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            final_img, time_elapsed = self.process_images(
                images=[frame],
                conf=conf,
                iou=iou,
                augment=augment,
                agnostic_nms=agnostic_nms,
                alpha=alpha,
                use_sahi=use_sahi,
                sahi_conf=sahi_conf,
                sahi_slice_height=sahi_slice_height,
                sahi_slice_width=sahi_slice_width,
                sahi_overlap_height_ratio=sahi_overlap_height_ratio,
                sahi_overlap_width_ratio=sahi_overlap_width_ratio
            )
            if target_interval:
                remaining = target_interval - time_elapsed
                if remaining > 0:
                    sleep(remaining)

            for area_name, area_items in self.areas.items():
                current_count = len(area_items["person_data"])
                buffers[area_name].append(current_count)
                if gather_data_file:
                    writer.writerow([time(), area_name, current_count])
                    csv_file.flush()
                if len(buffers[area_name]) >= self._seq_len:
                    res = self.zone_pred.get_zone_predictions(
                        capacity=area_items["capacity"],
                        duration_s=len(buffers[area_name]),
                        recent_counts=buffers[area_name]
                    )
                    for r in res:
                        self.areas[area_name]["pred"][r["horizon_s"]]["val"] = r["predicted_count"]
                        self.areas[area_name]["pred"][r["horizon_s"]]["occupancy_pct"] = r["occupancy_pct"]

            key = cv2.waitKey(1)
            if key == 27:
                break
            
            self.draw_areas_summary(image=final_img)
            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time
            cv2.putText(final_img, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
            cv2.imshow("res", final_img)

        cv2.destroyAllWindows()
        cap.release()

    def stream_busy_response(self):
        
        cv2.putText(self.funi_img, "Stream is already busy", (20, 80),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (0, 0, 200), 2)
        cv2.putText(self.funi_img, "Refresh page and try again later", (20, 130),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 200), 1)
        _, buffer = cv2.imencode(".jpg", self.funi_img)
        yield (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    
    def stream_video(self, 
                    video_path: Union[int, str, Path],
                    conf: float = .2,
                    iou: float = .35, 
                    augment: bool = True, 
                    agnostic_nms: bool = True,
                    alpha: float = 0.6, 
                    use_sahi: bool = False,
                    sahi_conf: float = 0.2, 
                    sahi_slice_height: int = 480, 
                    sahi_slice_width: int = 480, 
                    sahi_overlap_height_ratio: float = 0.2, 
                    sahi_overlap_width_ratio: float = 0.2):
        """for flask"""
        self._streaming = True

        cap = cv2.VideoCapture(video_path)
        buffers = {area: deque(maxlen=self._seq_len) for area in self.areas}
        p_time = 0
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
    
                final_img, time_elapsed = self.process_images(
                    images=[frame],
                    conf=conf,
                    iou=iou,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    alpha=alpha,
                    use_sahi=use_sahi,
                    sahi_conf=sahi_conf,
                    sahi_slice_height=sahi_slice_height,
                    sahi_slice_width=sahi_slice_width,
                    sahi_overlap_height_ratio=sahi_overlap_height_ratio,
                    sahi_overlap_width_ratio=sahi_overlap_width_ratio
                )
    
                for area_name, area_items in self.areas.items():
                    current_count = len(area_items["person_data"])
                    buffers[area_name].append(current_count)
                    self.history[area_name].append((time(), current_count))
    
                    if len(buffers[area_name]) >= self._seq_len:
                        res = self.zone_pred.get_zone_predictions(
                            capacity=area_items["capacity"],
                            duration_s=len(buffers[area_name]) * time_elapsed,
                            recent_counts=buffers[area_name]
                        )
                        for r in res:
                            self.areas[area_name]["pred"][r["horizon_s"]]["val"] = r["predicted_count"]
                            self.areas[area_name]["pred"][r["horizon_s"]]["occupancy_pct"] = r["occupancy_pct"]
    
                self.draw_areas_summary(image=final_img)
                c_time = time()
                fps = int(1 / (c_time - p_time))
                p_time = c_time
                cv2.putText(final_img, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
    
                _, buffer = cv2.imencode(".jpg", final_img)
                frame_data = buffer.tobytes()

                yield (b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
    
        finally:
            cap.release()
            self._streaming = False


if __name__ == "__main__":
    gowno = AreaMonitorSystem()
    gowno.load_areas(f"{Config.AREAS_FOLDER}/video3.json")
    gowno.process_video(
        video_input=r"C:\Users\table\PycharmProjects\MojeCos\bombon\Videos\video3.mp4",
        conf=Config.YOLO_CONF_THRESH,
        iou=Config.YOLO_IOU,
        augment=Config.YOLO_AUGMENT,
        agnostic_nms=Config.YOLO_AGNOSTIC_NMS,
        alpha=Config.ALPHA, 
        target_interval=None, 
        gather_data_file= None,
        use_sahi=Config.USE_SAHI,
        sahi_conf=Config.SAHI_CONF_THRESH,
        sahi_slice_height=Config.SAHI_SLICE_HEIGHT, 
        sahi_slice_width=Config.SAHI_SLICE_WIDTH, 
        sahi_overlap_height_ratio=Config.SAHI_OVERLAP_HEIGHT_RATIO, 
        sahi_overlap_width_ratio=Config.SAHI_OVERLAP_WIDTH_RATIO

    )
