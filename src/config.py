import os

class Config:
    # Model Settings (Using YOLO-World for open-vocabulary detection)
    MODEL_NAME = "yolov8s-worldv2.pt"
    
    # Custom open-vocabulary classes for YOLO-World
    CUSTOM_CLASSES = [
        "person", "man", "woman", "human", "face", "earbuds", "charging cable", 
        "mobile phone", "laptop", "smartwatch", "headphones", "mouse", "keyboard", 
        "monitor", "tablet", "power bank", "usb drive", "camera",
        "adapter", "electronic gadget", "furniture", "clothing", "bottle", "cup"
    ]
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # Detection Settings
    CONFIDENCE_THRESHOLD = 0.30
    
    # Video Settings
    VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
    DEFAULT_VIDEO_PATH = os.path.join(VIDEO_DIR, "sample.mp4")
    
    # Output Settings
    SAVE_OUTPUT = True
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.avi")
    
    # Detection Filtering
    # Set to a list of class IDs (e.g., [0, 2] for Person and Car) or None for all classes.
    ALLOWED_CLASSES = None
    
    # SORT Tracker Settings
    MAX_AGE = 30
    MIN_HITS = 3
    IOU_THRESHOLD = 0.3
