import cv2
import time
import numpy as np
class Visualizer:
    def __init__(self, class_names_func):
        self.class_names_func = class_names_func
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")

    def draw_detections_and_tracks(self, frame, tracks, fps=None):
        """
        Draw bounding boxes, labels, IDs, and FPS.
        tracks format: [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = map(int, track)
            
            # Select color based on track ID
            color = [int(c) for c in self.colors[track_id % len(self.colors)]]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            class_name = self.class_names_func(class_id)
            label = f"{class_name} | ID: {track_id} | {conf}%"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw total count
        count_label = f"Tracked Objects: {len(tracks)}"
        cv2.putText(frame, count_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw FPS
        if fps is not None:
            fps_label = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        return frame
