"""
Tracker module - Simple centroid-based vehicle tracking.
Assigns IDs to vehicles and tracks their position across frames.
"""
import math
from collections import defaultdict


class VehicleTracker:
    """
    Centroid-based tracker. Assigns a unique ID to each vehicle
    and tracks its position across frames. Calculates velocity.
    """

    def __init__(self, max_distance=80, max_disappeared=15):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracks = {}  # id -> {bbox, centroid, prev_centroid, velocity, disappeared, stopped}
        self.frame_count = 0

    def update(self, detections):
        """
        Update tracker with new detections from current frame.
        detections: list of {class_name, confidence, bbox}
        Returns: dict of tracked vehicles {id: {class_name, bbox, centroid, velocity, is_stopped, appeared_frame}}
        """
        self.frame_count += 1
        current_centroids = []
        current_detections = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_centroids.append((cx, cy))
            current_detections.append(det)

        # First frame - register all
        if len(self.tracks) == 0:
            for i, det in enumerate(current_detections):
                self._register(det, current_centroids[i])
        else:
            # Match existing tracks to new detections
            used_detections = set()
            used_tracks = set()

            # Calculate distances
            for track_id, track in self.tracks.items():
                if track["disappeared"] >= self.max_disappeared:
                    continue

                best_dist = float("inf")
                best_det_idx = -1

                for i, (cx, cy) in enumerate(current_centroids):
                    if i in used_detections:
                        continue

                    dist = math.hypot(cx - track["centroid"][0], cy - track["centroid"][1])
                    if dist < best_dist and dist < self.max_distance:
                        best_dist = dist
                        best_det_idx = i

                if best_det_idx >= 0:
                    used_detections.add(best_det_idx)
                    used_tracks.add(track_id)
                    self._update_track(track_id, current_detections[best_det_idx], current_centroids[best_det_idx])

            # Register unmatched detections as new vehicles
            for i, (cx, cy) in enumerate(current_centroids):
                if i not in used_detections:
                    self._register(current_detections[i], (cx, cy))

            # Mark unmatched tracks as disappeared
            for track_id in self.tracks:
                if track_id not in used_tracks:
                    self.tracks[track_id]["disappeared"] += 1

        # Remove tracks that disappeared too long
        to_remove = [tid for tid, t in self.tracks.items() if t["disappeared"] >= self.max_disappeared]
        for tid in to_remove:
            self._log_track_history(tid)
            del self.tracks[tid]

        # Return active tracks
        active = {}
        for tid, track in self.tracks.items():
            if track["disappeared"] == 0:
                active[tid] = {
                    "class_name": track["class_name"],
                    "bbox": track["bbox"],
                    "centroid": track["centroid"],
                    "velocity": track["velocity"],
                    "is_stopped": track["is_stopped"],
                    "appeared_frame": track["appeared_frame"]
                }
        return active

    def _register(self, det, centroid):
        self.tracks[self.next_id] = {
            "class_name": det["class_name"],
            "bbox": det["bbox"],
            "centroid": centroid,
            "prev_centroid": centroid,
            "velocity": 0,
            "disappeared": 0,
            "is_stopped": False,
            "appeared_frame": self.frame_count,
            "stop_frames": 0
        }
        self.next_id += 1

    def _update_track(self, track_id, det, centroid):
        track = self.tracks[track_id]
        prev = track["centroid"]
        dist = math.hypot(centroid[0] - prev[0], centroid[1] - prev[1])

        # Smooth velocity (moving average)
        track["velocity"] = track["velocity"] * 0.6 + dist * 0.4
        track["prev_centroid"] = prev
        track["centroid"] = centroid
        track["bbox"] = det["bbox"]
        track["disappeared"] = 0

        # Track if vehicle is stopped (velocity below threshold for several frames)
        if track["velocity"] < 2.0:
            track["stop_frames"] += 1
            track["is_stopped"] = track["stop_frames"] >= 3
        else:
            track["stop_frames"] = 0
            track["is_stopped"] = False

    def _log_track_history(self, track_id):
        pass  # Could log track history for debugging
