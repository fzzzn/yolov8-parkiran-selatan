import cv2
import numpy as np

# Global variables
points = []
current_area = "ORANGE"
areas = {"ORANGE": [], "RED": []}
img_display = None
img_original = None
scale_factor = 1.0
dragging_point = None  # (area_name, point_index)
drag_mode = False

def mouse_callback(event, x, y, flags, param):
    global points, img_display, current_area, scale_factor, dragging_point, drag_mode
    
    # Convert display coordinates to original image coordinates
    orig_x = int(x / scale_factor)
    orig_y = int(y / scale_factor)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking near an existing point in completed areas
        for area_name, area_points in areas.items():
            if area_points:
                for i, (px, py) in enumerate(area_points):
                    dist = np.sqrt((orig_x - px)**2 + (orig_y - py)**2)
                    if dist < 20:  # Within 20 pixels
                        dragging_point = (area_name, i)
                        drag_mode = True
                        print(f"Dragging {area_name} point {i+1}")
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
            points.append((orig_x, orig_y))
            print(f"{current_area} point {len(points)}: ({orig_x}, {orig_y})")
            
            # Close polygon if 4 points
            if len(points) == 4:
                areas[current_area] = points.copy()
                print(f"\n{current_area} area completed!")
                print(f"Coordinates: {points}\n")
                
                if current_area == "ORANGE":
                    print("Now define RED area (click 4 corners)")
                    current_area = "RED"
                else:
                    print("\nAll areas defined!")
                    print_areas_code()
                
                points = []
            
            redraw_display()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_mode and dragging_point:
            area_name, point_idx = dragging_point
            if area_name == "current":
                points[point_idx] = (orig_x, orig_y)
            else:
                areas[area_name][point_idx] = (orig_x, orig_y)
            redraw_display()
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drag_mode and dragging_point:
            area_name, point_idx = dragging_point
            if area_name == "current":
                points[point_idx] = (orig_x, orig_y)
                print(f"Moved current point {point_idx+1} to ({orig_x}, {orig_y})")
            else:
                areas[area_name][point_idx] = (orig_x, orig_y)
                print(f"Moved {area_name} point {point_idx+1} to ({orig_x}, {orig_y})")
                if area_name == "RED" and all(areas.values()):
                    print_areas_code()
            dragging_point = None
            drag_mode = False
            redraw_display()

def redraw_display():
    global img_display, img_original, scale_factor, points, areas, current_area
    
    # Start with scaled original
    img_display = cv2.resize(img_original, None, fx=scale_factor, fy=scale_factor)
    
    # Draw completed areas
    for area_name, area_points in areas.items():
        if area_points:
            color = (0, 165, 255) if area_name == "ORANGE" else (0, 0, 255)
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
    cv2.putText(img_display, f"Defining: {current_area} ({len(points)}/4)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_display, "Click to add, drag to move | 'r' reset | 'q' quit", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Define Areas", img_display)

def print_areas_code():
    print("\n" + "="*60)
    print("COPY THIS TO areas.py:")
    print("="*60)
    print("import numpy as np\n")
    
    if areas["ORANGE"]:
        print("# 🟧 ORANGE AREA (upper parking section)")
        print("ORANGE_AREA = np.array([")
        for i, (x, y) in enumerate(areas["ORANGE"]):
            label = ["top-left", "top-right", "bottom-right", "bottom-left"][i]
            print(f"    ({x:4d}, {y:4d}),    # {label}")
        print("], dtype=np.int32)\n")
    
    if areas["RED"]:
        print("# 🟥 RED AREA (lower parking section)")
        print("RED_AREA = np.array([")
        for i, (x, y) in enumerate(areas["RED"]):
            label = ["top-left", "top-right", "bottom-right", "bottom-left"][i]
            print(f"    ({x:4d}, {y:4d}),    # {label}")
        print("], dtype=np.int32)")
    
    print("="*60)

def main():
    global img_display, img_original, scale_factor
    
    # Load image
    img = cv2.imread("test_frame.jpg")
    if img is None:
        print("ERROR: test_frame.jpg not found")
        print("Run: python capture_frame.py first")
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
    print("\nInstructions:")
    print("1. Click 4 corners of ORANGE area (top-left, top-right, bottom-right, bottom-left)")
    print("2. Then click 4 corners of RED area")
    print("3. Drag any point to adjust position")
    print("4. Press 'r' to reset, 'q' to quit\n")
    print("Starting with ORANGE area...\n")
    
    cv2.namedWindow("Define Areas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Define Areas", int(w*scale_factor), int(h*scale_factor))
    cv2.setMouseCallback("Define Areas", mouse_callback)
    redraw_display()
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\nReset! Starting over with ORANGE area...")
            points.clear()
            areas["ORANGE"].clear()
            areas["RED"].clear()
            current_area = "ORANGE"
            redraw_display()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
