import numpy as np
import traceback
from src.tracker import associate_detections_to_trackers

dets = np.array([[10, 10, 50, 50, 0.9, 0]])
trks = np.array([[10, 10, 50, 50, 0]])

try:
    print("Testing associate_detections_to_trackers with overlapping det/trk...")
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)
    print("Matched:", matched.shape)
    
    print("Testing associate_detections_to_trackers with non-overlapping det/trk...")
    dets2 = np.array([[100, 100, 150, 150, 0.9, 0]])
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets2, trks)
    print("Matched:", matched.shape)

except Exception as e:
    traceback.print_exc()
