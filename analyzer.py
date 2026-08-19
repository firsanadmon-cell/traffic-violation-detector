"""
Analyzer module - Traffic violation detection logic.
Analyzes detections frame by frame and determines if vehicles are
committing traffic violations.
"""


class ViolationAnalyzer:
    """
    Analyzes vehicle tracks and traffic signal detections to identify violations.
    
    Supported violations:
    - Red light violation: vehicle moves through an intersection while traffic light is red
    - Stop sign violation: vehicle passes a stop sign without stopping
    """

    def __init__(self, frame_skip=3, fps=30):
        """
        Args:
            frame_skip: process every N-th frame (for performance)
            fps: video frames per second (used for timestamp calculation)
        """
        self.frame_skip = frame_skip
        self.fps = fps
        self.violations = []
        self.violated_vehicles = set()  # track IDs that already have a violation (to avoid duplicates)
        self.frame_number = 0
        self.evidence_frames = []  # frames where violations occurred (for later extraction)

        # For red light: track which vehicles were near a red light and moving
        self.red_light_vehicles = {}  # vehicle_id -> frames_near_red_light

        # For stop sign: track which vehicles were near a stop sign
        self.stop_sign_vehicles = {}  # vehicle_id -> {approaching, stopped}

    def analyze_frame(self, frame_number, tracked_vehicles, traffic_lights, stop_signs):
        """
        Analyze a single frame for violations.
        
        Args:
            frame_number: current frame number
            tracked_vehicles: dict of {id: {class_name, bbox, centroid, velocity, is_stopped}}
            traffic_lights: list of {confidence, bbox, state}
            stop_signs: list of {confidence, bbox}
        
        Returns:
            list of new violations detected in this frame
        """
        self.frame_number = frame_number
        new_violations = []
        timestamp = frame_number / self.fps if self.fps > 0 else 0

        # --- RED LIGHT VIOLATION ---
        red_lights = [tl for tl in traffic_lights if tl["state"] == "red"]

        if red_lights:
            for vehicle_id, vehicle in tracked_vehicles.items():
                for tl in red_lights:
                    if self._is_near_traffic_light(vehicle["bbox"], tl["bbox"]):
                        # Vehicle is near a red light
                        if vehicle_id not in self.red_light_vehicles:
                            self.red_light_vehicles[vehicle_id] = 0
                        self.red_light_vehicles[vehicle_id] += 1

                        # If vehicle is moving and has been near red light for a few frames
                        if (not vehicle["is_stopped"] and
                            vehicle["velocity"] > 3.0 and
                            self.red_light_vehicles[vehicle_id] >= 2 and
                            vehicle_id not in self.violated_vehicles):

                            violation = {
                                "type": "red_light_violation",
                                "vehicle_id": vehicle_id,
                                "vehicle_type": vehicle["class_name"],
                                "timestamp": round(timestamp, 2),
                                "frame_number": frame_number,
                                "traffic_light_bbox": tl["bbox"],
                                "vehicle_bbox": vehicle["bbox"],
                                "description": f"Vehículo ({vehicle['class_name']}) cruzó semáforo en rojo",
                                "severity": "alta"
                            }
                            new_violations.append(violation)
                            self.violations.append(violation)
                            self.violated_vehicles.add(vehicle_id)
        else:
            # No red lights visible - reset tracking
            self.red_light_vehicles.clear()

        # --- STOP SIGN VIOLATION ---
        if stop_signs:
            for vehicle_id, vehicle in tracked_vehicles.items():
                for ss in stop_signs:
                    if self._is_near_stop_sign(vehicle["bbox"], ss["bbox"]):
                        if vehicle_id not in self.stop_sign_vehicles:
                            self.stop_sign_vehicles[vehicle_id] = {"approaching": True, "ever_stopped": False}

                        track = self.stop_sign_vehicles[vehicle_id]

                        if vehicle["is_stopped"]:
                            track["ever_stopped"] = True

                        # If vehicle has passed the stop sign and never stopped
                        if (self._has_passed_stop_sign(vehicle["bbox"], ss["bbox"]) and
                            not track["ever_stopped"] and
                            vehicle_id not in self.violated_vehicles and
                            vehicle["velocity"] > 2.0):

                            violation = {
                                "type": "stop_sign_violation",
                                "vehicle_id": vehicle_id,
                                "vehicle_type": vehicle["class_name"],
                                "timestamp": round(timestamp, 2),
                                "frame_number": frame_number,
                                "stop_sign_bbox": ss["bbox"],
                                "vehicle_bbox": vehicle["bbox"],
                                "description": f"Vehículo ({vehicle['class_name']}) no se detuvo en señal de PARE",
                                "severity": "media"
                            }
                            new_violations.append(violation)
                            self.violations.append(violation)
                            self.violated_vehicles.add(vehicle_id)

        # --- EXCESSIVE SPEED (estimated) ---
        # Simple heuristic: if a vehicle's bbox shrinks rapidly (approaching camera fast)
        # or grows rapidly, flag as potential speed violation
        # This is a rough estimate and should be calibrated with real-world data
        for vehicle_id, vehicle in tracked_vehicles.items():
            if vehicle["velocity"] > 50.0 and vehicle_id not in self.violated_vehicles:
                violation = {
                    "type": "potential_speeding",
                    "vehicle_id": vehicle_id,
                    "vehicle_type": vehicle["class_name"],
                    "timestamp": round(timestamp, 2),
                    "frame_number": frame_number,
                    "vehicle_bbox": vehicle["bbox"],
                    "estimated_velocity": round(vehicle["velocity"], 2),
                    "description": f"Vehículo ({vehicle['class_name']}) con velocidad elevada detectada",
                    "severity": "media"
                }
                new_violations.append(violation)
                self.violations.append(violation)
                self.violated_vehicles.add(vehicle_id)

        return new_violations

    def _is_near_traffic_light(self, vehicle_bbox, light_bbox, threshold=200):
        """Check if a vehicle is near a traffic light (within threshold pixels)."""
        vx_center = (vehicle_bbox[0] + vehicle_bbox[2]) / 2
        lx_center = (light_bbox[0] + light_bbox[2]) / 2
        vy_center = (vehicle_bbox[1] + vehicle_bbox[3]) / 2
        ly_center = (light_bbox[1] + light_bbox[3]) / 2
        dist = ((vx_center - lx_center) ** 2 + (vy_center - ly_center) ** 2) ** 0.5
        return dist < threshold

    def _is_near_stop_sign(self, vehicle_bbox, sign_bbox, threshold=150):
        """Check if a vehicle is near a stop sign."""
        vx_center = (vehicle_bbox[0] + vehicle_bbox[2]) / 2
        sx_center = (sign_bbox[0] + sign_bbox[2]) / 2
        vy_center = (vehicle_bbox[1] + vehicle_bbox[3]) / 2
        sy_center = (sign_bbox[1] + sign_bbox[3]) / 2
        dist = ((vx_center - sx_center) ** 2 + (vy_center - sy_center) ** 2) ** 0.5
        return dist < threshold

    def _has_passed_stop_sign(self, vehicle_bbox, sign_bbox):
        """Check if vehicle has passed (is below/beyond) the stop sign."""
        vehicle_bottom = vehicle_bbox[3]
        sign_center_y = (sign_bbox[1] + sign_bbox[3]) / 2
        return vehicle_bottom > sign_center_y + 20

    def get_all_violations(self):
        """Return all violations detected so far."""
        return self.violations

    def get_summary(self):
        """Return a summary of all violations by type."""
        summary = {}
        for v in self.violations:
            vtype = v["type"]
            if vtype not in summary:
                summary[vtype] = {"count": 0, "violations": []}
            summary[vtype]["count"] += 1
            summary[vtype]["violations"].append(v)
        return summary
