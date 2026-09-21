import numpy as np
import traceback
from src.tracker import SortTracker

tracker = SortTracker()
tracker.update(np.array([[10, 10, 50, 50, 0.9, 0]]))

try:
    print("Testing with shape (0,)")
    res = tracker.update(np.array([]))
    print("Result:", res.shape)
except Exception as e:
    traceback.print_exc()
