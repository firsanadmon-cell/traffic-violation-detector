"""
Test script - verify the API is working.
Run: python test_api.py
"""
import requests
import sys

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_root():
    """Test root endpoint."""
    print("\nTesting /...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_analyze(video_path):
    """Test analyze endpoint with a video file."""
    print(f"\nTesting /analyze with {video_path}...")
    try:
        with open(video_path, "rb") as f:
            files = {"file": (video_path, f, "video/mp4")}
            data = {"frame_skip": "5", "save_evidence": "true"}
            print("  Uploading and processing (this may take a while)...")
            r = requests.post(f"{BASE_URL}/analyze", files=files, data=data, timeout=300)

        print(f"  Status: {r.status_code}")
        result = r.json()

        print(f"\n  Video info: {result['video_info']}")
        print(f"  Detections: {result['detections_summary']}")
        print(f"  Total violations: {result['total_violations']}")

        if result["violations"]:
            print("\n  Violations found:")
            for v in result["violations"]:
                print(f"    [{v['severity'].upper()}] {v['type']} at {v['timestamp']}s - {v['description']}")

        print(f"\n  Summary by type: {result['summary_by_type']}")
        return r.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Traffic Violation Detector - API Tests")
    print("=" * 50)

    ok1 = test_health()
    ok2 = test_root()

    if len(sys.argv) > 1:
        ok3 = test_analyze(sys.argv[1])
    else:
        print("\n(No video file provided. Usage: python test_api.py <video.mp4>)")
        ok3 = True

    print("\n" + "=" * 50)
    if ok1 and ok2 and ok3:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    print("=" * 50)
