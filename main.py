import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO # type: ignore
import os
import json
from dotenv import load_dotenv

from telegram_bot import send_telegram_photo

# Load environment variables
load_dotenv()

# Fix PyTorch CPU compatibility issues on VMs
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
# Disable oneDNN for CPU compatibility
torch.backends.mkldnn.enabled = False

# Load areas from JSON config
with open('config.json', 'r') as f:
    config = json.load(f)

# Parse areas dynamically
AREAS = []
for area_config in config['areas']:
    AREAS.append({
        'id': area_config['id'],
        'name': area_config['name'],
        'color': tuple(area_config['color']),
        'polygon': np.array(area_config['polygon'], dtype=np.int32)
    })

print(f"Loaded {len(AREAS)} parking areas: {', '.join([a['name'] for a in AREAS])}")

# -------------------------
# CONFIG
# -------------------------
RTSP_URL = os.getenv("RTSP_URL")
CAPTURE_INTERVAL = 60  # seconds - capture, analyze, and send every minute
RTSP_RETRY_TIMEOUT = 10  # seconds to wait before retrying failed RTSP connection

# YOLO model: use yolov8m.pt for good detection with CPU compatibility
MODEL_PATH = "yolov8m.pt"  # Medium model - good balance for CPU
CONF = 0.01  # Absolute minimum confidence for maximum detection
IOU_THRESHOLD = 0.10  # Maximum overlapping detection tolerance
IMG_SIZE = 640  # Standard size for stable CPU inference

# COCO class ids: 3=motorcycle
DETECT_CLASSES = [3]  # Motorcycle only
MOTORCYCLE_CLASS = 3

# Advanced detection settings
USE_TTA = False  # Test-Time Augmentation disabled for CPU inference stability
MAX_DET = 1000  # Maximum possible detections for extremely dense parking
MIN_AREA = 150  # Very small minimum area for distant/partial motorcycles

# Where to save telegram snapshot
SNAPSHOT_PATH = "/tmp/parking_status.jpg"


# -------------------------
# GEOMETRY HELPERS
# -------------------------
def box_center_xy(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_poly(pt, poly):
    # poly is Nx2 np.int32
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def count_motorcycles_in_area(dets_xyxy, poly):
    c = 0
    for box in dets_xyxy:
        cx, cy = box_center_xy(box)
        if point_in_poly((cx, cy), poly):
            c += 1
    return c


def count_motorcycles_per_area(dets):
    """Count motorcycles in each defined area"""
    area_counts = []
    for area in AREAS:
        count = count_motorcycles_in_area(dets, area['polygon'])
        status = "DETECTED" if count > 0 else "EMPTY"
        area_counts.append({
            'id': area['id'],
            'name': area['name'],
            'count': count,
            'status': status
        })
    return area_counts


# -------------------------
# RTSP CAPTURE HELPERS
# -------------------------
def open_capture(url):
    cap = cv2.VideoCapture(url)
    # optional tuning
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def read_frame(cap, url, max_retries=3):
    """Attempt to read frame with retry logic"""
    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap, frame, True
        
        print(f"  ⚠ RTSP read failed (attempt {attempt + 1}/{max_retries})")
        cap.release()
        time.sleep(2)
        
        print(f"  ↻ Reconnecting to RTSP...")
        cap = open_capture(url)
        time.sleep(1)
    
    print("  ✗ RTSP connection failed after all retries")
    return cap, None, False


def preprocess_frame(frame):
    """Minimal preprocessing for CPU compatibility"""
    # Return frame as-is to avoid CPU compatibility issues
    return frame

def add_timestamp_overlay(frame):
    """Add timestamp overlay to frame"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add semi-transparent background for timestamp
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 350, 10), (w - 10, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Add timestamp text
    cv2.putText(frame, timestamp, (w - 340, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


# -------------------------
# DETECTION & ANALYSIS
# -------------------------
def detect_motorcycles(model, frame):
    """Run YOLO detection and return filtered detections within defined areas"""
    # Preprocess frame for better detection
    processed_frame = preprocess_frame(frame)

    # YOLO inference with advanced settings
    results = model(
        processed_frame,
        conf=CONF,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        classes=DETECT_CLASSES,
        verbose=False,
        agnostic_nms=False,
        max_det=MAX_DET,
        augment=USE_TTA,
        retina_masks=True
    )

    # Collect all boxes with confidence scores
    dets = []
    confs = []
    classes = []
    for r in results:
        if r.boxes is None:
            continue
        dets.extend(r.boxes.xyxy.cpu().numpy())
        confs.extend(r.boxes.conf.cpu().numpy())
        classes.extend(r.boxes.cls.cpu().numpy())

    dets = np.array(dets) if len(dets) else np.empty((0, 4), dtype=np.float32)
    confs = np.array(confs) if len(confs) else np.empty(0, dtype=np.float32)
    classes = np.array(classes) if len(classes) else np.empty(0, dtype=np.float32)

    # Filter by minimum area
    if len(dets) > 0:
        widths = dets[:, 2] - dets[:, 0]
        heights = dets[:, 3] - dets[:, 1]
        areas_size = widths * heights
        valid_mask = areas_size > MIN_AREA
        dets = dets[valid_mask]
        confs = confs[valid_mask]
        classes = classes[valid_mask]
    
    # Filter to only keep detections within any defined area
    if len(dets) > 0:
        in_area_mask = np.zeros(len(dets), dtype=bool)
        for i, box in enumerate(dets):
            cx, cy = box_center_xy(box)
            for area in AREAS:
                if point_in_poly((cx, cy), area['polygon']):
                    in_area_mask[i] = True
                    break
        
        dets = dets[in_area_mask]
        confs = confs[in_area_mask]
        classes = classes[in_area_mask]

    return dets, confs, classes


def analyze_areas(dets, confs, classes):
    """Count motorcycles in each area and determine status"""
    motorcycles = np.sum(classes == MOTORCYCLE_CLASS) if len(classes) > 0 else 0
    avg_conf = np.mean(confs) if len(confs) > 0 else 0

    area_counts = count_motorcycles_per_area(dets)

    return {
        'motorcycles': motorcycles,
        'avg_conf': avg_conf,
        'areas': area_counts
    }


def annotate_frame(frame, dets, confs, area_counts):
    """Add all visual annotations to the frame"""
    # Draw bounding boxes with confidence
    for i, box in enumerate(dets):
        x1, y1, x2, y2 = map(int, box)
        conf = confs[i] if i < len(confs) else 0
        color = (0, 255, 0)  # Green for motorcycles
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"M: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw polygons for all areas
    for area in AREAS:
        cv2.polylines(frame, [area['polygon']], True, area['color'], 3)

    # Add status text at bottom-left for all areas
    h, w = frame.shape[:2]
    y_base = h - (40 * (len(area_counts) + 1))  # Dynamic spacing based on number of areas
    
    cv2.putText(frame, f"Total detections: {len(dets)}",
                (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for i, area_data in enumerate(area_counts):
        y_pos = y_base + 40 * (i + 1)
        area = next(a for a in AREAS if a['id'] == area_data['id'])
        text = f"{area['name']}: {area_data['status']} ({area_data['count']})"
        cv2.putText(frame, text, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, area['color'], 2)
    
    # Add timestamp overlay
    frame = add_timestamp_overlay(frame)
    
    return frame


def send_notification(area_counts):
    """Save annotated frame and send Telegram notification"""
    msg_lines = ["Parking Status\n"]
    
    for area_data in area_counts:
        area = next(a for a in AREAS if a['id'] == area_data['id'])
        # Use emoji based on area name or id
        emoji = "🟧" if 'orange' in area['id'].lower() else "🟥" if 'red' in area['id'].lower() else "🔷"
        msg_lines.append(
            f"{emoji} {area['name']}: {area_data['status']} ({area_data['count']} motorcycles)"
        )
    
    msg = "\n".join(msg_lines)
    
    try:
        send_telegram_photo(SNAPSHOT_PATH, msg)
        print(f"  ✓ Telegram notification sent successfully")
    except Exception as e:
        print(f"  ✗ Telegram error: {e}")


# -------------------------
# MODEL MANAGEMENT
# -------------------------
def check_and_download_model():
    """Check if model exists, download if needed"""
    model_path = MODEL_PATH
    
    # Check if model file exists
    if os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model found: {model_path} ({file_size_mb:.1f} MB)")
        return True
    
    print(f"Model not found: {model_path}")
    print(f"Downloading YOLOv8 Medium model... (this may take a few minutes)")
    
    try:
        # YOLO() will automatically download if missing
        model = YOLO(model_path)
        print(f"✓ Model downloaded successfully: {model_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


# -------------------------
# MAIN
# -------------------------
def main():
    # Check/download model before starting
    print("\n" + "="*60)
    print("Initializing YOLOv8 Parking Monitor")
    print("="*60)
    
    if not check_and_download_model():
        print("Failed to initialize model. Exiting.")
        return
    
    model = YOLO(MODEL_PATH)
    cap = open_capture(RTSP_URL)
    
    consecutive_failures = 0
    max_consecutive_failures = 5

    print("="*60)
    print("Starting monitoring loop - Capture every 60 seconds")
    print("="*60)
    print(f"Capture interval: {CAPTURE_INTERVAL}s")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence: {CONF} | IOU: {IOU_THRESHOLD}")
    print("="*60 + "\n")

    while True:
        cycle_start = time.time()
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting capture cycle...")

        # Capture frame from RTSP
        cap, frame, success = read_frame(cap, RTSP_URL)
        
        if not success or frame is None:
            consecutive_failures += 1
            print(f"  ✗ Frame capture failed ({consecutive_failures}/{max_consecutive_failures})")
            
            if consecutive_failures >= max_consecutive_failures:
                print(f"  ⚠ Too many consecutive failures. Waiting {RTSP_RETRY_TIMEOUT}s before retry...")
                time.sleep(RTSP_RETRY_TIMEOUT)
                cap.release()
                cap = open_capture(RTSP_URL)
                consecutive_failures = 0
            
            # Wait before next cycle
            time.sleep(CAPTURE_INTERVAL)
            continue
        
        # Reset failure counter on success
        consecutive_failures = 0
        print(f"  ✓ Frame captured successfully")

        # Detect motorcycles
        dets, confs, classes = detect_motorcycles(model, frame)

        # Analyze areas and get status
        analysis = analyze_areas(dets, confs, classes)
        
        print(f"[{time.strftime('%H:%M:%S')}] Detected: {analysis['motorcycles']} motorcycles | Avg conf: {analysis['avg_conf']:.2f}")
        for area_data in analysis['areas']:
            print(f"  {area_data['name']}: {area_data['status']} ({area_data['count']} motorcycles)")

        # Annotate frame with detections and status
        frame = annotate_frame(frame, dets, confs, analysis['areas'])
        
        # Save snapshot
        cv2.imwrite(SNAPSHOT_PATH, frame)
        print(f"  ✓ Snapshot saved: {SNAPSHOT_PATH}")

        # Send notification
        send_notification(analysis['areas'])
        
        # Wait for next cycle
        cycle_duration = time.time() - cycle_start
        sleep_time = max(0, CAPTURE_INTERVAL - cycle_duration)
        
        if sleep_time > 0:
            print(f"  ⏳ Cycle completed in {cycle_duration:.1f}s. Waiting {sleep_time:.1f}s until next capture...")
            time.sleep(sleep_time)
        else:
            print(f"  ⚠ Cycle took {cycle_duration:.1f}s (longer than {CAPTURE_INTERVAL}s interval)")


if __name__ == "__main__":
    main()
