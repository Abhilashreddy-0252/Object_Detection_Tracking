# Object Detection and Tracking System

A complete object detection and tracking pipeline using **YOLOv8** for real-time object detection and **SORT (Simple Online and Realtime Tracking)** for multiple object tracking.

## Objectives
- Real-time video processing using OpenCV.
- High accuracy object detection across multiple COCO classes using a lightweight YOLOv8 nano model.
- Robust tracking of detected objects across frames using the SORT algorithm (Kalman Filter + Hungarian Algorithm).
- Modular and extensible architecture suitable for college project demonstrations.

## Key Features
- **Multiple Input Sources:** Switch seamlessly between live Webcam and local Video Files via a terminal menu.
- **Robust Tracking:** Assigns and maintains unique IDs for tracked objects.
- **Performance Metrics:** Real-time display of FPS and total tracked object count.
- **Interactive Controls:**
  - `Q` or `ESC` : Quit the application
  - `P` : Pause/Resume the video feed
  - `R` : Reset tracking algorithms
- **Modular Codebase:** Cleanly separated concerns (Detector, Tracker, VideoSource, Visualizer).

## Technologies
- **Python 3.x**
- **OpenCV** - For video ingestion, frame manipulation, and rendering.
- **YOLOv8 (Ultralytics)** - Pre-trained deep learning model for high-speed object detection.
- **SORT Algorithm** - Using `FilterPy` (Kalman Filters) and `SciPy` (Hungarian matching algorithm).
- **NumPy** - For array operations and tensor manipulations.

## Project Structure
```text
Object_Detection_Tracking/
├── models/
│   └── yolov8n.pt             # YOLOv8 nano model weights (auto-downloaded on first run)
├── videos/
│   └── sample.mp4             # Directory for video files
├── src/
│   ├── __init__.py
│   ├── detector.py            # YOLOv8 integration
│   ├── tracker.py             # SORT implementation
│   ├── video_source.py        # OpenCV VideoCapture wrapper
│   ├── visualizer.py          # Bounding box and label rendering
│   └── config.py              # Centralized configuration
├── main.py                    # Application entry point and orchestrator
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation & Setup

1. **Clone or Extract the Project**
   Open a terminal and navigate to the project directory:
   ```cmd
   cd path\to\Object_Detection_Tracking
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   Run the following command to install all required libraries:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Add a Sample Video (Optional)**
   Place any `.mp4` video in the `videos/` folder and name it `sample.mp4`, or provide the path to your own video when running the script.

## Usage

Run the main script to launch the interactive terminal menu:
```cmd
python main.py
```

### Modes of Operation:
- **Webcam (Real-time):** Select option `1`. The system will connect to your default webcam (usually `0`) and perform real-time detection and tracking.
- **Video File:** Select option `2`. Provide the absolute or relative path to a video file. If you press Enter without typing, it will attempt to load `videos/sample.mp4`.

### Expected Output:
A new OpenCV window will open displaying the video feed.
- Detected objects will have a bounding box.
- Above the bounding box, a label will display: `Class Name | ID: <Track_ID> | <Confidence>%`
- Top-left corner will display the current `FPS` and `Tracked Objects` count.

## Error Handling
The system handles errors gracefully:
- If a webcam cannot be opened, it will warn the user and safely exit.
- If an invalid video path is provided, it terminates cleanly.
- If the YOLO model is missing, it will automatically download it using the ultralytics library, or throw a clear error if the download fails.
