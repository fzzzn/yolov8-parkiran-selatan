import cv2
import numpy as np
import json
import os
from dotenv import load_dotenv

# Configuration
AREA_DEFINITIONS = [
    {"id": "orange", "name": "Orange Area", "color": [0, 165, 255], "emoji": "🟧"},
    {"id": "red", "name": "Red Area", "color": [0, 0, 255], "emoji": "🟥"},
    # Add more areas as needed:
    # {"id": "green", "name": "Green Area", "color": [0, 255, 0], "emoji": "🟢"},
]

# Global variables
points = []
current_area_index = 0
areas = {area["id"]: [] for area in AREA_DEFINITIONS}
img_display = None
img_original = None
scale_factor = 1.0
dragging_point = None  # (area_id, point_index)
drag_mode = False

def get_current_area():
    """Get current area configuration"""
    return AREA_DEFINITIONS[current_area_index] if current_area_index < len(AREA_DEFINITIONS) else None

def get_area_config(area_id):
    """Get area configuration by ID"""
    return next((a for a in AREA_DEFINITIONS if a["id"] == area_id), None)

def mouse_callback(event, x, y, flags, param):
    global points, img_display, current_area_index, scale_factor, dragging_point, drag_mode
    
    # Convert display coordinates to original image coordinates
    orig_x = int(x / scale_factor)
    orig_y = int(y / scale_factor)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking near an existing point in completed areas
        for area_id, area_points in areas.items():
            if area_points:
                for i, (px, py) in enumerate(area_points):
                    dist = np.sqrt((orig_x - px)**2 + (orig_y - py)**2)
                    if dist < 20:  # Within 20 pixels
                        dragging_point = (area_id, i)
                        drag_mode = True
                        area_config = get_area_config(area_id)
                        if area_config:
                            print(f"Dragging {area_config['name']} point {i+1}")
                        return
        
        # Check if clicking near current points
        for i, (px, py) in enumerate(points):
            dist = np.sqrt((orig_x - px)**2 + (orig_y - py)**2)
            if dist < 20:
                dragging_point = ("current", i)
                drag_mode = True
                print(f"Dragging current point {i+1}")
                return
        
        # Not dragging, add new point
        if not drag_mode:
            current_area = get_current_area()
            if current_area:
                points.append((orig_x, orig_y))
                print(f"{current_area['name']} point {len(points)}: ({orig_x}, {orig_y})")
                
                # Close polygon if 4 points
                if len(points) == 4:
                    areas[current_area['id']] = points.copy()
                    print(f"\n{current_area['emoji']} {current_area['name']} completed!")
                    print(f"Coordinates: {points}\n")
                    
                    current_area_index += 1
                    if current_area_index < len(AREA_DEFINITIONS):
                        next_area = AREA_DEFINITIONS[current_area_index]
                        print(f"Now define {next_area['emoji']} {next_area['name']} (click 4 corners)")
                    else:
                        print("\n✓ All areas defined!")
                        save_to_json()
                    
                    points = []
                
                redraw_display()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_mode and dragging_point:
            area_id, point_idx = dragging_point
            if area_id == "current":
                points[point_idx] = (orig_x, orig_y)
            else:
                areas[area_id][point_idx] = (orig_x, orig_y)
            redraw_display()
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drag_mode and dragging_point:
            area_id, point_idx = dragging_point
            if area_id == "current":
                points[point_idx] = (orig_x, orig_y)
                print(f"Moved current point {point_idx+1} to ({orig_x}, {orig_y})")
            else:
                areas[area_id][point_idx] = (orig_x, orig_y)
                area_config = get_area_config(area_id)
                if area_config:
                    print(f"Moved {area_config['name']} point {point_idx+1} to ({orig_x}, {orig_y})")
                # Check if all areas are complete
                if all(areas[a['id']] for a in AREA_DEFINITIONS):
                    save_to_json()
            dragging_point = None
            drag_mode = False
            redraw_display()

def redraw_display():
    global img_display, img_original, scale_factor, points, areas, current_area_index
    
    # Start with scaled original
    img_display = cv2.resize(img_original, None, fx=scale_factor, fy=scale_factor)
    
    # Draw completed areas
    for area_config in AREA_DEFINITIONS:
        area_id = area_config["id"]
        area_points = areas.get(area_id, [])
        if area_points:
            color = tuple(area_config["color"])
            pts = np.array([(int(x*scale_factor), int(y*scale_factor)) for x, y in area_points])
            cv2.polylines(img_display, [pts], True, color, 3)
            # Fill with semi-transparent
            overlay = img_display.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, img_display, 0.8, 0, img_display)
            
            # Draw draggable points for completed areas
            for i, (px, py) in enumerate(area_points):
                disp_x = int(px * scale_factor)
                disp_y = int(py * scale_factor)
                cv2.circle(img_display, (disp_x, disp_y), 8, color, -1)
                cv2.circle(img_display, (disp_x, disp_y), 10, (255, 255, 255), 2)
    
    # Draw current area points
    if points:
        color = (0, 255, 0)
        for i, (px, py) in enumerate(points):
            disp_x = int(px * scale_factor)
            disp_y = int(py * scale_factor)
            cv2.circle(img_display, (disp_x, disp_y), 8, color, -1)
            cv2.circle(img_display, (disp_x, disp_y), 10, (255, 255, 255), 2)
            cv2.putText(img_display, str(i+1), (disp_x+10, disp_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if i > 0:
                prev_x = int(points[i-1][0] * scale_factor)
                prev_y = int(points[i-1][1] * scale_factor)
                cv2.line(img_display, (prev_x, prev_y), (disp_x, disp_y), color, 2)
    
    # Add instructions
    current_area = get_current_area()
    if current_area:
        status_text = f"Defining: {current_area['emoji']} {current_area['name']} ({len(points)}/4)"
    else:
        status_text = "All areas defined! Press 's' to save"
    
    cv2.putText(img_display, status_text, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_display, "Click to add, drag to move | 'r' reset | 's' save | 'q' quit", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Define Areas", img_display)

def capture_frame_from_rtsp(rtsp_url, output_path="test_frame.jpg"):
    """Capture a frame from RTSP stream"""
    print(f"Connecting to RTSP stream...")
    print(f"URL: {rtsp_url[:20]}...")
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("ERROR: Could not connect to RTSP stream")
        print("Please check:")
        print("  1. RTSP_URL in .env file is correct")
        print("  2. Camera is accessible from this network")
        print("  3. Credentials are valid")
        return False
    
    print("Connected! Capturing frame...")
    
    # Discard first few frames to get fresh image
    for _ in range(5):
        cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("ERROR: Could not read frame from stream")
        return False
    
    cv2.imwrite(output_path, frame)
    h, w = frame.shape[:2]
    print(f"✓ Frame captured: {w}x{h} saved to {output_path}\n")
    return True

def save_to_json():
    """Save areas to config.json"""
    config = {"areas": []}
    
    for area_config in AREA_DEFINITIONS:
        area_id = area_config["id"]
        if areas.get(area_id):
            config["areas"].append({
                "id": area_id,
                "name": area_config["name"],
                "color": area_config["color"],
                "polygon": areas[area_id]
            })
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ Configuration saved to config.json")
    print("="*60)
    print("="*60)

def main():
    global img_display, img_original, scale_factor
    
    # Load environment variables
    load_dotenv()
    
    # Check if test frame exists, if not try to capture it
    if not os.path.exists("test_frame.jpg"):
        print("test_frame.jpg not found. Attempting to capture from RTSP stream...\n")
        rtsp_url = os.getenv("RTSP_URL")
        
        if not rtsp_url:
            print("ERROR: RTSP_URL not found in .env file")
            print("Please create a .env file with RTSP_URL=your_camera_url")
            return
        
        if not capture_frame_from_rtsp(rtsp_url):
            print("\nFailed to capture frame. Please:")
            print("  1. Check your RTSP camera connection")
            print("  2. Or manually save a frame as test_frame.jpg")
            return
    
    # Load image
    img = cv2.imread("test_frame.jpg")
    if img is None:
        print("ERROR: Could not load test_frame.jpg")
        return
    
    img_original = img.copy()
    h, w = img.shape[:2]
    
    # Scale to fit 1080p screen (leave margin)
    max_height = 1000
    max_width = 1800
    
    scale_h = max_height / h
    scale_w = max_width / w
    scale_factor = min(scale_h, scale_w, 1.0)  # Don't upscale
    
    print(f"Original image: {w}x{h}")
    print(f"Display scale: {scale_factor:.2f}x")
    print(f"Display size: {int(w*scale_factor)}x{int(h*scale_factor)}\n")
    
    print("="*60)
    print("INTERACTIVE AREA DEFINITION")
    print("="*60)
    print(f"\nTotal areas to define: {len(AREA_DEFINITIONS)}")
    for area in AREA_DEFINITIONS:
        print(f"  {area['emoji']} {area['name']}")
    print("\nInstructions:")
    print("1. Click 4 corners for each area (top-left, top-right, bottom-right, bottom-left)")
    print("2. Drag any point to adjust position")
    print("3. Press 's' to save to config.json")
    print("4. Press 'r' to reset, 'q' to quit\n")
    print(f"Starting with {AREA_DEFINITIONS[0]['emoji']} {AREA_DEFINITIONS[0]['name']}...\n")
    
    cv2.namedWindow("Define Areas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Define Areas", int(w*scale_factor), int(h*scale_factor))
    cv2.setMouseCallback("Define Areas", mouse_callback)
    redraw_display()
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current configuration
            if any(areas.values()):
                save_to_json()
                print("\n✓ Configuration saved! You can close the window or continue editing.")
        elif key == ord('r'):
            print(f"\nReset! Starting over with {AREA_DEFINITIONS[0]['name']}...")
            points.clear()
            for area_id in areas:
                areas[area_id].clear()
            current_area_index = 0
            redraw_display()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()