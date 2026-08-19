"""
Detector module - YOLOv8 object detection for vehicles and traffic signals.
Detects: cars, motorcycles, buses, trucks, traffic lights, stop signs.
"""
import numpy as np
from ultralytics import YOLO

# COCO class IDs we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRAFFIC_LIGHT_CLASS = 9
STOP_SIGN_CLASS = 11


class Detector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.4):
        """
        Initialize YOLOv8 model.
        Models: yolov8n.pt (nano, fast), yolov8s.pt (small, balanced), yolov8m.pt (medium, accurate)
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        print("Model loaded successfully.")

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns: {
            'vehicles': [{class_name, confidence, bbox: [x1,y1,x2,y2]}],
            'traffic_lights': [{confidence, bbox, state: 'red'|'yellow'|'green'}],
            'stop_signs': [{confidence, bbox}]
        }
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        vehicles = []
        traffic_lights = []
        stop_signs = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                if cls_id in VEHICLE_CLASSES:
                    vehicles.append({
                        "class_name": VEHICLE_CLASSES[cls_id],
                        "confidence": round(conf, 3),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)]
                    })
                elif cls_id == TRAFFIC_LIGHT_CLASS:
                    state = self._detect_traffic_light_state(frame, [x1, y1, x2, y2])
                    traffic_lights.append({
                        "confidence": round(conf, 3),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "state": state
                    })
                elif cls_id == STOP_SIGN_CLASS:
                    stop_signs.append({
                        "confidence": round(conf, 3),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)]
                    })

        return {
            "vehicles": vehicles,
            "traffic_lights": traffic_lights,
            "stop_signs": stop_signs
        }

    def _detect_traffic_light_state(self, frame, bbox):
        """
        Analyze the traffic light bounding box to determine its state.
        Looks at the red, yellow, and green channel intensity in the region.
        Returns: 'red', 'yellow', 'green', or 'unknown'
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define color ranges (HSV)
        red_lower1 = np.array([0, 80, 80])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 80, 80])
        red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([20, 80, 80])
        yellow_upper = np.array([35, 255, 255])
        green_lower = np.array([40, 80, 80])
        green_upper = np.array([90, 255, 255])

        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        red_pixels = int(np.sum(red_mask > 0))
        yellow_pixels = int(np.sum(yellow_mask > 0))
        green_pixels = int(np.sum(green_mask > 0))

        # Threshold: need enough colored pixels to be confident
        min_pixels = max(10, int(roi.shape[0] * roi.shape[1] * 0.02))

        if red_pixels > min_pixels and red_pixels >= yellow_pixels and red_pixels >= green_pixels:
            return "red"
        elif yellow_pixels > min_pixels and yellow_pixels >= red_pixels and yellow_pixels >= green_pixels:
            return "yellow"
        elif green_pixels > min_pixels and green_pixels >= red_pixels and green_pixels >= yellow_pixels:
            return "green"
        else:
            return "unknown"


import cv2  # imported here to avoid circular import issues at module level
