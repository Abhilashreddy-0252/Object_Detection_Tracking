import os
from ultralytics import YOLO
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path, confidence_threshold=0.5, custom_classes=None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLOv8 model
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
        print(f"Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            
            # Set custom open-vocabulary classes if using YOLO-World
            if custom_classes and "world" in self.model_path.lower():
                print(f"Setting custom YOLO-World classes: {custom_classes}")
                self.model.set_classes(custom_classes)
                
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise e
            
        self.classes = self.model.names

    def detect(self, frame, allowed_classes=None):
        """
        Detect objects in the given frame.
        Returns a list of detections: [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            if conf >= self.confidence_threshold:
                if allowed_classes is None or cls_id in allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2, conf, cls_id])
                
        if len(detections) == 0:
            return np.empty((0, 6))
        return np.array(detections)

    def get_class_name(self, class_id):
        return self.classes.get(class_id, "Unknown")
