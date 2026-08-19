"""
Traffic Violation Detector API
FastAPI service that receives videos and returns detected traffic violations.

Endpoints:
  POST /analyze    - Upload a video and get violations detected
  GET  /health     - Health check
  GET  /           - API info
"""
import os
import tempfile
import uuid
import json
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from detector import Detector
from tracker import VehicleTracker
from analyzer import ViolationAnalyzer

app = FastAPI(
    title="Traffic Violation Detector API",
    description="Analiza videos y detecta infracciones de tránsito usando YOLOv8 + OpenCV",
    version="1.0.0"
)

# CORS - allow all origins for prototype
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (loads YOLO model on startup)
detector = None


@app.on_event("startup")
async def startup_event():
    global detector
    model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
    conf = float(os.getenv("CONF_THRESHOLD", "0.4"))
    detector = Detector(model_name=model_name, conf_threshold=conf)


@app.get("/")
async def root():
    return {
        "service": "Traffic Violation Detector",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze - Upload video file",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": detector is not None}


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    frame_skip: int = 5,
    save_evidence: bool = False
):
    """
    Analyze a video file for traffic violations.

    Args:
        file: Video file (mp4, avi, mov, etc.)
        frame_skip: Process every N-th frame (default 5 for performance)
        save_evidence: If true, saves frames where violations occur

    Returns:
        JSON with:
        - total_violations: count
        - violations: list of violation objects
        - summary: violations grouped by type
        - video_info: fps, total_frames, duration
        - detections_summary: total vehicles, traffic lights, stop signs detected
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")

    # Save uploaded video to temp file
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Initialize tracker and analyzer
    tracker = VehicleTracker(max_distance=80, max_disappeared=15)
    analyzer = ViolationAnalyzer(frame_skip=frame_skip, fps=fps if fps > 0 else 30)

    # Evidence frames directory
    evidence_dir = None
    if save_evidence:
        evidence_dir = os.path.join(temp_dir, "evidence")
        os.makedirs(evidence_dir, exist_ok=True)

    # Stats
    frame_count = 0
    processed_frames = 0
    total_vehicles_detected = 0
    total_lights_detected = 0
    total_stop_signs_detected = 0
    all_violations = []

    print(f"Processing video: {file.filename} | FPS: {fps} | Total frames: {total_frames} | Duration: {duration:.1f}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for performance
        if frame_count % frame_skip != 0:
            continue

        processed_frames += 1

        # Run detection
        detections = detector.detect(frame)

        # Track vehicles
        tracked = tracker.update(detections["vehicles"])

        # Analyze for violations
        new_violations = analyzer.analyze_frame(
            frame_count,
            tracked,
            detections["traffic_lights"],
            detections["stop_signs"]
        )

        # Save evidence frames
        if save_evidence and new_violations and evidence_dir:
            for v in new_violations:
                evidence_path = os.path.join(evidence_dir, f"violation_{v['type']}_frame_{frame_count}.jpg")
                cv2.imwrite(evidence_path, frame)

        # Update stats
        total_vehicles_detected = max(total_vehicles_detected, len(tracked))
        total_lights_detected += len(detections["traffic_lights"])
        total_stop_signs_detected += len(detections["stop_signs"])

        if new_violations:
            all_violations.extend(new_violations)
            for v in new_violations:
                print(f"  [VIOLATION] Frame {frame_count} | {v['type']} | {v['description']}")

    cap.release()

    # Build response
    summary = analyzer.get_summary()

    result = {
        "video_info": {
            "filename": file.filename,
            "fps": round(fps, 2) if fps > 0 else 0,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2) if duration > 0 else 0,
            "processed_frames": processed_frames,
            "frame_skip": frame_skip
        },
        "detections_summary": {
            "max_vehicles_in_frame": total_vehicles_detected,
            "traffic_light_detections": total_lights_detected,
            "stop_sign_detections": total_stop_signs_detected
        },
        "total_violations": len(all_violations),
        "violations": all_violations,
        "summary_by_type": summary
    }

    print(f"Analysis complete: {len(all_violations)} violations found")

    # Cleanup
    try:
        os.remove(video_path)
    except Exception:
        pass

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
