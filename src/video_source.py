import cv2

class VideoSource:
    def __init__(self, source):
        """
        Initialize video source.
        source: int for webcam (e.g., 0) or str for video file path.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video source {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30.0 # Default for webcam if not correctly reported

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
