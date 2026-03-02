# Vehicle Detection and Counter

Real-time vehicle detection, tracking, and counting using **YOLOv8 + Deep SORT + OpenCV**.

## Features
- Detects vehicles: `car`, `motorcycle`, `bus`, `truck`
- Tracks each vehicle with a unique ID (Deep SORT)
- Counts:
  - Total unique detected vehicles
  - Vehicles crossing a counting line (top-to-bottom)
- Live on-screen overlays: bounding boxes, labels, counters, FPS

## Tech Stack
- Python
- Ultralytics YOLOv8
- Deep SORT Realtime
- OpenCV
- PyTorch

## Project Structure
```text
vehicle_detection_and_counter/
├── src/
│   └── main.py
├── models/
│   ├── yolov8n.pt
│   └── yolov8m.pt
├── data/
│   ├── traffic.mp4
│   └── coco.names
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup
1. Create and activate virtual environment:

```terminal
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```terminal
pip install -r requirements.txt
```

## Run
```terminal
python src/main.py
```

Press `q` to quit the window.

## Configuration
Edit values in `src/main.py` if needed:
- `MODEL_PATH` (default: `models/yolov8n.pt`)
- `VIDEO_SOURCE` (default: `data/traffic.mp4`)
- `CONF_THRESHOLD`
- `FRAME_SKIP`

## Output
The app shows `Detected`, `Crossed`, and `FPS` on screen.
When you close it, a short summary is printed in terminal.

## Notes
- If video does not open, check `VIDEO_SOURCE` path.
- Keep large files in `.gitignore`.
