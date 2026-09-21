import numpy as np
import traceback
from src.tracker import SortTracker

tracker = SortTracker()
detections = np.array([[10, 10, 50, 50, 0.9, 0]]) # One detection

try:
    print("Testing with 1 detection (trackers empty)...")
    res1 = tracker.update(detections)
    print("Result 1:", res1.shape)
    
    print("Testing with 1 detection (tracker exists)...")
    res2 = tracker.update(detections)
    print("Result 2:", res2.shape)
    
    print("Testing with 0 detections...")
    res3 = tracker.update(np.empty((0, 6)))
    print("Result 3:", res3.shape)
    
except Exception as e:
    traceback.print_exc()
