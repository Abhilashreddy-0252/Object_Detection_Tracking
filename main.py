import sys
import cv2
import time
from src.config import Config
from src.detector import YOLOv8Detector
from src.tracker import SortTracker
from src.video_source import VideoSource
from src.visualizer import Visualizer

def print_menu():
    print("="*50)
    print(" Object Detection and Tracking with YOLOv8 & SORT")
    print("="*50)
    print("1. Webcam (Real-time)")
    print("2. Video File")
    print("3. Exit")
    print("="*50)

def main():
    while True:
        print_menu()
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '1':
            source = 0
            break
        elif choice == '2':
            source = input(f"Enter video file path (or press Enter for default: {Config.DEFAULT_VIDEO_PATH}): ").strip()
            if not source:
                source = Config.DEFAULT_VIDEO_PATH
            break
        elif choice == '3':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid option. Please try again.\n")

    try:
        # Initialize components
        print(f"Initializing video source: {source}...")
        video_source = VideoSource(source)
        
        detector = YOLOv8Detector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD, custom_classes=Config.CUSTOM_CLASSES)
        tracker = SortTracker(max_age=Config.MAX_AGE, min_hits=Config.MIN_HITS, iou_threshold=Config.IOU_THRESHOLD)
        visualizer = Visualizer(detector.get_class_name)
        
        print("\nControls:")
        print(" 'Q' or 'ESC' - Quit")
        print(" 'P'          - Pause/Resume")
        print(" 'R'          - Reset Tracking\n")
        
        # Initialize video writer
        out = None
        if Config.SAVE_OUTPUT:
            import os
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(Config.DEFAULT_OUTPUT_PATH, fourcc, video_source.fps, (video_source.width, video_source.height))
            print(f"Saving output video to: {Config.DEFAULT_OUTPUT_PATH}\n")
            
        paused = False
        prev_time = time.time()
        
        # Setup Fullscreen Window
        window_name = "Object Detection and Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while True:
            if not paused:
                ret, frame = video_source.read_frame()
                if not ret:
                    print("End of video stream or error reading frame.")
                    break
                
                # Detection with optional filtering
                detections = detector.detect(frame, allowed_classes=Config.ALLOWED_CLASSES)
                
                # Tracking
                tracks = tracker.update(detections)
                
                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Visualization
                frame = visualizer.draw_detections_and_tracks(frame, tracks, fps)
                
                # Write to output file
                if out is not None:
                    out.write(frame)
                    
            # Display
            cv2.imshow("Object Detection and Tracking", frame)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("Paused.")
                else:
                    print("Resumed.")
                    prev_time = time.time() # Reset time to avoid FPS spike
            elif key == ord('r'):
                print("Resetting tracker...")
                tracker = SortTracker(max_age=Config.MAX_AGE, min_hits=Config.MIN_HITS, iou_threshold=Config.IOU_THRESHOLD)
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'out' in locals() and out is not None:
                out.release()
            video_source.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()
