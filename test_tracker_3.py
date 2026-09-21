import numpy as np
import traceback
from src.tracker import SortTracker

tracker = SortTracker()
detections = np.array([[10, 10, 50, 50, 0.9, 0]])
tracker.update(detections) # Adds one tracker

try:
    print("Testing with 0 detections and 1 tracker...")
    res = tracker.update(np.empty((0, 6)))
    print("Result:", res.shape)
except Exception as e:
    traceback.print_exc()
