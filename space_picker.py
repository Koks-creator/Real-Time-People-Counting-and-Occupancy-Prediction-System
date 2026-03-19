from dataclasses import dataclass
from typing import Tuple, Union
import cv2
import numpy as np
import json
from collections import defaultdict

from config import Config


@dataclass
class SpacePicker:
    max_corner_number: int = 20
    areas_res_path: str = ""

    def __post_init__(self) -> None:
        self.temp_points = []
        self.areas = []

    @staticmethod
    def check_inside2p(point: Tuple[int, int], top_left_p: Tuple[int, int], bot_right_p: Tuple[int, int]) -> bool:
        if top_left_p[0] < point[0] < bot_right_p[0] and top_left_p[1] < point[1] < bot_right_p[1]:
            return True
        return False

    def mouse_click(self, event, x, y, flags, params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.temp_points.append([x, y])
        if event == cv2.EVENT_RBUTTONDOWN:
            for index, region in enumerate(self.areas):
                ppt = cv2.pointPolygonTest(region, (x, y), False)
                if ppt in (1, 0):
                    self.areas.pop(index)

    @staticmethod
    def __nothing(x) -> None:
        pass

    def run(self, image_path: str) -> None:
        cv2.namedWindow("Options")
        cv2.createTrackbar("MaxNumberOfCorners", "Options", 4, self.max_corner_number, self.__nothing)
        cv2.setTrackbarMin("MaxNumberOfCorners", "Options", 3)

        # print(self.__temp_points)
        # if not self.output_file_path:
        #     self.output_file_path = Config.AREAS_FILE
        while True:
            max_corners = cv2.getTrackbarPos("MaxNumberOfCorners", "Options")
            img = cv2.imread(image_path)
            img = cv2.resize(img, (1280, 720))

            cv2.putText(img, "Press 's' to save", (15, 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 225), 2)

            for region in self.areas:
                # print(region)
                cv2.polylines(img, [np.array([region])], True, (255, 255, 255), 4)
                # cv2.rectangle(img, (region[0]), (region[-1]), (255, 0, 255), 2)

            if len(self.temp_points) == max_corners:
                self.areas.append(self.temp_points)
                self.temp_points = []
            for point in self.temp_points:
                cv2.circle(img, point, 8, (255, 0, 200), -1)

            key = cv2.waitKey(1)

            if key == 27:
                break

            if key == ord("s"):
                res_json = defaultdict(dict)
                for ind, points in enumerate(self.areas):
                    res_json[f"area{ind+1}"]["area"] = points
                    res_json[f"area{ind+1}"]["capacity"] = 0

                res_json = json.dumps(res_json, indent=4)
                with open(self.areas_res_path, "w") as f:
                    f.write(res_json)
                print(f"saved to {self.areas_res_path}")

            cv2.imshow("res", img)
            cv2.setMouseCallback("res", self.mouse_click)


if __name__ == '__main__':
    picker = SpacePicker(areas_res_path=f"{Config.AREAS_FOLDER}/video3.json")
    picker.run(
        image_path=r"Screenshot_5.png")