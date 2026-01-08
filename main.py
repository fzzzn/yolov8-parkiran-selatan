import time
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
import os
from dotenv import load_dotenv

from areas import ORANGE_AREA, RED_AREA
from telegram_bot import send_telegram_photo

# Load environment variables
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
RTSP_URL = os.getenv("RTSP_URL")
CAPTURE_INTERVAL = 60  # seconds - capture, analyze, and send every minute
RTSP_RETRY_TIMEOUT = 10  # seconds to wait before retrying failed RTSP connection

# YOLO model: use yolov8x.pt for best detection in extremely crowded scenes
MODEL_PATH = "yolov8x.pt"  # Extra large model for maximum accuracy
CONF = 0.01  # Absolute minimum confidence for maximum detection
IOU_THRESHOLD = 0.10  # Maximum overlapping detection tolerance
IMG_SIZE = 1536  # Even larger for better small/distant object detection

# COCO class ids: 3=motorcycle
DETECT_CLASSES = [3]  # Motorcycle only
MOTORCYCLE_CLASS = 3

# Advanced detection settings
USE_TTA = True  # Test-Time Augmentation
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
    """Enhanced preprocessing for extremely crowded motorcycle parking"""
    # 1. Resize to higher resolution if needed for better detection
    h, w = frame.shape[:2]
    if w < 2048:  # Upscale to even higher resolution
        scale = 2048 / w
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Light denoising (faster than full denoising)
    denoised = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # 3. More aggressive CLAHE for better contrast in crowded areas
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # Increased clip limit
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 4. Even stronger sharpening for better edge and detail detection
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1,  2,  2,  2, -1],
                       [-1,  2, 10,  2, -1],  # Increased center weight
                       [-1,  2,  2,  2, -1],
                       [-1, -1, -1, -1, -1]]) / 10.0
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 5. Additional unsharp masking for even more detail
    gaussian = cv2.GaussianBlur(sharpened, (0, 0), 2.0)
    sharpened = cv2.addWeighted(sharpened, 1.5, gaussian, -0.5, 0)
    
    return sharpened

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
# MAIN
# -------------------------
def main():
    model = YOLO(MODEL_PATH)
    cap = open_capture(RTSP_URL)
    
    consecutive_failures = 0
    max_consecutive_failures = 5

    print("\n" + "="*60)
    print("YOLOv8 Parking Monitor - Every Minute Capture")
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
            retina_masks=True  # Better segmentation for overlapping objects
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

        dets = np.array(dets) if len(
            dets) else np.empty((0, 4), dtype=np.float32)
        confs = np.array(confs) if len(
            confs) else np.empty(0, dtype=np.float32)
        classes = np.array(classes) if len(
            classes) else np.empty(0, dtype=np.float32)

        # Apply additional filtering
        # Keep only detections with reasonable size
        if len(dets) > 0:
            widths = dets[:, 2] - dets[:, 0]
            heights = dets[:, 3] - dets[:, 1]
            areas = widths * heights
            # Filter: area > MIN_AREA pixels (reduced for crowded scenes)
            valid_mask = areas > MIN_AREA
            dets = dets[valid_mask]
            confs = confs[valid_mask]
            classes = classes[valid_mask]
        
        # Filter to only keep detections within orange or red areas
        if len(dets) > 0:
            in_area_mask = np.zeros(len(dets), dtype=bool)
            for i, box in enumerate(dets):
                cx, cy = box_center_xy(box)
                if point_in_poly((cx, cy), ORANGE_AREA) or point_in_poly((cx, cy), RED_AREA):
                    in_area_mask[i] = True
            
            dets = dets[in_area_mask]
            confs = confs[in_area_mask]
            classes = classes[in_area_mask]

        # Debug: print total detections with details
        motorcycles = np.sum(classes == MOTORCYCLE_CLASS) if len(
            classes) > 0 else 0
        avg_conf = np.mean(confs) if len(confs) > 0 else 0

        print(f"[{time.strftime('%H:%M:%S')}] Detected: {motorcycles} motorcycles | Avg conf: {avg_conf:.2f}")

        orange_count = count_motorcycles_in_area(dets, ORANGE_AREA)
        red_count = count_motorcycles_in_area(dets, RED_AREA)
        
        # Simple presence detection
        orange_status = "DETECTED" if orange_count > 0 else "EMPTY"
        red_status = "DETECTED" if red_count > 0 else "EMPTY"

        print(f"  Orange Area: {orange_status} ({orange_count} motorcycles)")
        print(f"  Red Area: {red_status} ({red_count} motorcycles)")

        # Annotate frame for snapshot
        # draw bounding boxes with confidence
        for i, box in enumerate(dets):
            x1, y1, x2, y2 = map(int, box)
            conf = confs[i] if i < len(confs) else 0

            # Green for motorcycles
            color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add confidence label
            label = f"M: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # draw polygons
        cv2.polylines(frame, [ORANGE_AREA], True, (0, 165, 255), 3)
        cv2.polylines(frame, [RED_AREA], True, (0, 0, 255), 3)

        # label status text at bottom-left
        h, w = frame.shape[:2]
        y_base = h - 80  # Start from bottom
        
        cv2.putText(frame, f"Total detections: {len(dets)}",
                    (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"RED Area: {red_status} ({red_count})",
                    (20, y_base + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"ORANGE Area: {orange_status} ({orange_count})",
                    (20, y_base + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Add timestamp overlay
        frame = add_timestamp_overlay(frame)
        
        # Save and send to Telegram
        cv2.imwrite(SNAPSHOT_PATH, frame)
        print(f"  ✓ Snapshot saved: {SNAPSHOT_PATH}")

        msg = (
            "Parking Status\n\n"
            f"🟧 Orange Area: {orange_status} ({orange_count} motorcycles)\n"
            f"🟥 Red Area: {red_status} ({red_count} motorcycles)\n"
        )

        try:
            send_telegram_photo(SNAPSHOT_PATH, msg)
            print(f"  ✓ Telegram notification sent successfully")
        except Exception as e:
            print(f"  ✗ Telegram error: {e}")
        
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
