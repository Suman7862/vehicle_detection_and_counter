import cv2
import time
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------- CONFIG --------------------
MODEL_PATH = "models/yolov8n.pt"
VIDEO_SOURCE = "data/traffic.mp4"  # Use 0 for local webcam
WINDOW_NAME = "Vehicle Detection & Counting (Final)"

# Detection/Tracking tuning
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
FRAME_SKIP = 5
SPEED_FACTOR = 2.0  # 1.0 = normal, 2.0 = 2x faster
INFER_IMGSZ = 640
TARGET_WIDTH = 960
COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


# -------------------- MODEL & TRACKER --------------------
model = YOLO(MODEL_PATH)
tracker = DeepSort(
    max_age=50,
    n_init=2,
    max_iou_distance=0.9,
)


# -------------------- VIDEO SOURCE --------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(
        f"Unable to open video source: {VIDEO_SOURCE}. "
        "Check camera URL/network or set VIDEO_SOURCE = 0 for webcam."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25

if SPEED_FACTOR <= 0:
    raise ValueError("SPEED_FACTOR must be > 0")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


# -------------------- HELPERS --------------------
prev_positions = {}  # track_id -> previous y
track_labels = {}  # track_id -> object label

detected_ids = set()
counted_ids = set()

detected_count = 0
crossed_count = 0
line_y = None

frame_count = 0
prev_frame_time = None


# -------------------- MAIN LOOP --------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize large frames before inference/tracking for better real-time performance.
        if TARGET_WIDTH and frame.shape[1] > TARGET_WIDTH:
            scale = TARGET_WIDTH / frame.shape[1]
            frame = cv2.resize(
                frame,
                (TARGET_WIDTH, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        if line_y is None:
            line_y = int(frame.shape[0] * 0.75)

        # YOLOv8 detection
        detections = []
        results = model(
            frame,
            imgsz=INFER_IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=COCO_VEHICLE_CLASS_IDS,
            device=DEVICE,
            half=HALF_PRECISION,
            verbose=False,
        )

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])
                detections.append(([x1, y1, w, h], conf, label))

        # Deep SORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            detected_ids.add(track_id)

            x1, y1, x2, y2 = map(int, track.to_ltrb())

            if track_id not in track_labels:
                det_class = track.get_det_class()
                track_labels[track_id] = det_class if det_class else "vehicle"

            label = track_labels.get(track_id, "vehicle")

            # Draw detection box and centroid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Count only top-to-bottom crossing once per track
            if track_id in prev_positions:
                prev_y = prev_positions[track_id]
                if prev_y < line_y <= cy and track_id not in counted_ids:
                    crossed_count += 1
                    counted_ids.add(track_id)

            prev_positions[track_id] = cy

        detected_count = len(detected_ids)

        # Draw counters and guide line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Detected: {detected_count}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Crossed: {crossed_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        now = time.time()
        if prev_frame_time is None:
            fps_curr = 0.0
        else:
            dt = now - prev_frame_time
            fps_curr = 1.0 / dt if dt > 0 else 0.0
        prev_frame_time = now

        cv2.putText(
            frame,
            f"FPS: {int(fps_curr)}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        delay = int(1000 / (fps * SPEED_FACTOR))
        delay = max(1, delay)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
finally:
    print("\n========== VEHICLE COUNT SUMMARY ==========")
    print(f"TOTAL DETECTED VEHICLES : {detected_count}")
    print(f"TOTAL CROSSED VEHICLES  : {crossed_count}")
    print("===========================================\n")

    cap.release()
    cv2.destroyAllWindows()
