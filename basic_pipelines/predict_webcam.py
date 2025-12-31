#!/usr/bin/env python3
"""
YOLOv11 Real-time Webcam Prediction Script for People Detection
This script performs real-time people detection using a trained YOLOv11 model on webcam feed.
"""

import os
import sys
import cv2
import logging
import argparse
import time
import numpy as np
from pathlib import Path
from collections import deque

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO
from config.settings import (
    MODEL_DIR, WEBCAM_CONFIG, CLASS_NAMES, COLORS
)
from utils.model_utils import (
    load_yolo_model, get_device, print_system_info,
    validate_model_path, draw_predictions
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebcamPredictor:
    """Real-time webcam prediction class"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 device: str = '', camera_id: int = 0, clustering_distance: float = 120.0):
        """
        Initialize webcam predictor
        
        Args:
            model_path (str): Path to trained model
            conf_threshold (float): Confidence threshold for predictions
            device (str): Device to use for inference
            camera_id (int): Camera ID (0 for default webcam)
            clustering_distance (float): Maximum distance for clustering people
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.clustering_distance = clustering_distance
        self.device = device or get_device()
        self.camera_id = camera_id
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)  # Track last 30 frames for FPS
        self.frame_count = 0
        self.start_time = time.time()
        
        # Object tracking variables
        self.selected_trackers = {}  # Dict of tracker_id: tracker_object
        self.selected_boxes = {}     # Dict of tracker_id: (x1, y1, x2, y2)
        self.current_detections = [] # Current frame detections for click detection
        self.next_tracker_id = 0
        self.mouse_callback_set = False
        self.current_frame = None    # Store current frame for mouse operations
        
        # Set class names and colors
        self.class_names = CLASS_NAMES
        self.colors = COLORS
        
        # Load model
        self._load_model()
        
        # Initialize camera
        self._initialize_camera()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = load_yolo_model(self.model_path, self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _initialize_camera(self):
        """Initialize the camera"""
        try:
            # Try several backends depending on platform
            backends = []
            
            # Raspberry Pi specific backends
            if sys.platform.startswith('linux'):
                # V4L2 is the standard Linux video backend - works best on Raspberry Pi
                if hasattr(cv2, 'CAP_V4L2'):
                    backends.append(('V4L2', cv2.CAP_V4L2))
                # GStreamer backend - good for Pi Camera
                if hasattr(cv2, 'CAP_GSTREAMER'):
                    backends.append(('GStreamer', cv2.CAP_GSTREAMER))
            # Windows backends
            elif sys.platform.startswith('win'):
                if hasattr(cv2, 'CAP_DSHOW'):
                    backends.append(('DSHOW', cv2.CAP_DSHOW))
                if hasattr(cv2, 'CAP_MSMF'):
                    backends.append(('MSMF', cv2.CAP_MSMF))
            
            # Always try default backend last
            backends.append(('Default', None))

            self.cap = None
            for backend_name, backend in backends:
                try:
                    if backend is None:
                        logger.info(f"Trying {backend_name} VideoCapture backend...")
                        cap = cv2.VideoCapture(self.camera_id)
                    else:
                        logger.info(f"Trying {backend_name} VideoCapture backend ({backend})...")
                        cap = cv2.VideoCapture(self.camera_id, backend)

                    if cap is not None and cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"Successfully opened camera with {backend_name} backend")
                            self.cap = cap
                            break
                        else:
                            logger.warning(f"{backend_name} backend opened but cannot read frames")
                            cap.release()
                    else:
                        if cap is not None:
                            cap.release()
                except Exception as e:
                    logger.warning(f"Failed to open camera with {backend_name}: {e}")
                    if cap is not None:
                        try:
                            cap.release()
                        except:
                            pass

            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id} with any tested backend")

            # Set camera properties (best-effort)
            # Raspberry Pi cameras often work better with specific resolutions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Set buffer size to 1 to reduce latency on Raspberry Pi
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                # Some backends don't allow setting FPS; ignore
                pass

            # Get actual camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {width}x{height} @ {fps} FPS")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            # Provide additional diagnostic hints
            logger.error("Hints: \n  - Check that a webcam is connected and not in use by another app.\n  - Try a different camera id with --camera (e.g. 1, 2).\n  - On Windows, ensure Camera privacy settings allow apps to use the camera.\n  - If using a USB camera, try reconnecting the device or using a different USB port.")
            raise
    
    def _calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        if hasattr(self, 'last_time'):
            frame_time = current_time - self.last_time
            if frame_time > 0:
                fps = 1.0 / frame_time
                self.fps_queue.append(fps)
        
        self.last_time = current_time
        
        # Return average FPS
        if self.fps_queue:
            return sum(self.fps_queue) / len(self.fps_queue)
        return 0
    
    def _draw_info_overlay(self, frame: np.ndarray, fps: float, 
                          detection_count: int) -> np.ndarray:
        """
        Draw information overlay on frame
        
        Args:
            frame (np.ndarray): Input frame
            fps (float): Current FPS
            detection_count (int): Number of detections
            
        Returns:
            np.ndarray: Frame with overlay
        """
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw background rectangle for info (made taller for more info)
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text information
        info_text = [
            f"FPS: {fps:.1f}",
            f"Detections: {detection_count}",
            f"Confidence: {self.conf_threshold:.2f} ",
            f"Clustering: {self.clustering_distance:.0f}px ",
            f"Model: {Path(self.model_path).name}",
            f"Device: {self.device.upper()}"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 25 + i * 18
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def _process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame with YOLO detection and visualization"""
        results = self.model(frame)
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Update detection count
                num_detections = len(result.boxes)
                self.detection_count = num_detections
                
                # Store current detections for tracking
                self.current_detections = []
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    self.current_detections.append({
                        'id': i,
                        'box': box,
                        'confidence': conf
                    })
                
                # Update trackers
                self._update_trackers(frame)
                
                # Draw predictions with tracking information
                annotated_frame = draw_predictions(
                    frame, 
                    results, 
                    self.class_names, 
                    self.colors,
                    self.clustering_distance,
                    self.selected_boxes,
                    self.current_detections
                )
                
                return annotated_frame, num_detections
        
        # If no detections, reset count and return original frame
        self.detection_count = 0
        self.current_detections = []
        return frame, 0
    
    def run(self, save_video: bool = False, output_path: str = None):
        """
        Run real-time prediction
        
        Args:
            save_video (bool): Whether to save video output
            output_path (str): Output video path
        """
        try:
            # Setup video writer if saving
            video_writer = None
            if save_video and output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_path, fourcc, 20.0, 
                    (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )
                logger.info(f"Saving video to: {output_path}")
            
            logger.info("Starting webcam prediction...")
            logger.info("Controls: q=quit, s=screenshot, r=reset, +/-=confidence, ,/.=clustering")
            logger.info("Mouse Controls: Left click=select object, Right click=deselect")
            logger.info("Make sure to CLICK on the video window first, then use keyboard!")
            
            screenshot_count = 0
            
            # Set up mouse callback when GUI is available
            window_name = 'YOLOv11 People Detection - Webcam'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self._mouse_callback)
            
            # For Raspberry Pi: flush initial frames from buffer
            if sys.platform.startswith('linux'):
                logger.info("Flushing camera buffer (Raspberry Pi optimization)...")
                for _ in range(5):
                    self.cap.read()
            
            failed_reads = 0
            max_failed_reads = 10
            
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    failed_reads += 1
                    logger.warning(f"Failed to read frame from camera (attempt {failed_reads}/{max_failed_reads})")
                    
                    if failed_reads >= max_failed_reads:
                        logger.error("Too many consecutive failed frame reads. Stopping.")
                        break
                    
                    # Wait a bit and try to re-grab
                    time.sleep(0.1)
                    # Try to flush buffer
                    for _ in range(3):
                        self.cap.grab()
                    continue
                
                # Reset failed reads counter on successful read
                failed_reads = 0
                
                # Validate frame
                if frame.size == 0:
                    logger.warning("Received empty frame, skipping...")
                    continue
                
                # Store current frame for mouse operations
                self.current_frame = frame.copy()
                
                # Process frame
                annotated_frame, detection_count = self._process_frame(frame)
                
                # Calculate FPS
                fps = self._calculate_fps()
                self.frame_count += 1
                
                # Draw info overlay
                final_frame = self._draw_info_overlay(
                    annotated_frame, fps, detection_count
                )
                
                # Save frame if recording
                if video_writer:
                    video_writer.write(final_frame)
                
                # Try to display frame (handle GUI not available)
                try:
                    window_name = 'YOLOv11 People Detection - Webcam'
                    cv2.imshow(window_name, final_frame)
                    
                    # Make sure window is focused and on top
                    if self.frame_count == 1:  # Only on first frame
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                        logger.info("Click on the video window to ensure keyboard input works!")
                    
                    gui_available = True
                except cv2.error as e:
                    if "not implemented" in str(e).lower():
                        # OpenCV built without GUI support
                        if not hasattr(self, 'gui_warning_shown'):
                            logger.warning("OpenCV GUI not available - running in headless mode")
                            logger.info("Frames will be saved but not displayed. Press Ctrl+C to stop.")
                            self.gui_warning_shown = True
                        gui_available = False
                    else:
                        raise
                
                # Handle key presses only if GUI is available
                if gui_available:
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Debug: print key code when any key is pressed
                    if key != 255:  # 255 means no key pressed
                        logger.info(f"Key pressed: {key} (char: {chr(key) if 32 <= key <= 126 else 'special'})")
                    
                    if key == ord('q') or key == 27:  # q or ESC
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_count += 1
                        screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
                        cv2.imwrite(screenshot_path, final_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('r'):
                        # Reset statistics
                        self.fps_queue.clear()
                        self.frame_count = 0
                        self.start_time = time.time()
                        logger.info("Statistics reset")
                    elif key == ord('+') or key == ord('=') or key == ord('u'):
                        # Increase confidence (use + or = or u key)
                        self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                        logger.info(f"Confidence increased to {self.conf_threshold:.2f}")
                    elif key == ord('-') or key == ord('_') or key == ord('d'):
                        # Decrease confidence (use - or _ or d key)
                        self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                        logger.info(f"Confidence decreased to {self.conf_threshold:.2f}")
                    elif key == ord('.') or key == ord('>') or key == ord('l'):
                        # Increase clustering distance (looser clustering) - use . or > or l key
                        self.clustering_distance = min(300.0, self.clustering_distance + 20.0)
                        logger.info(f"Clustering distance increased to {self.clustering_distance:.0f}px (looser grouping)")
                    elif key == ord(',') or key == ord('<') or key == ord('t'):
                        # Decrease clustering distance (tighter clustering) - use , or < or t key
                        self.clustering_distance = max(20.0, self.clustering_distance - 20.0)
                        logger.info(f"Clustering distance decreased to {self.clustering_distance:.0f}px (tighter grouping)")
                    elif key == ord('h'):
                        # Show help
                        logger.info("=== Keyboard Controls ===")
                        logger.info("q/ESC = Quit")
                        logger.info("s = Save screenshot")
                        logger.info("r = Reset statistics")
                        logger.info("+/=/u = Increase confidence threshold")
                        logger.info("-/_/d = Decrease confidence threshold")
                        logger.info("./>/l = Increase clustering distance (looser groups)")
                        logger.info(",/</t = Decrease clustering distance (tighter groups)")
                        logger.info("h = Show this help")
                    
                    # Try arrow keys with different key codes (varies by system)
                    elif key in [82, 65, 119]:  # Various up arrow codes or 'w'
                        self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                        logger.info(f"Confidence increased to {self.conf_threshold:.2f}")
                    elif key in [84, 66, 115]:  # Various down arrow codes or 's'
                        self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                        logger.info(f"Confidence decreased to {self.conf_threshold:.2f}")
                    elif key in [83, 67, 100]:  # Various right arrow codes or 'd'
                        self.clustering_distance = min(300.0, self.clustering_distance + 20.0)
                        logger.info(f"Clustering distance increased to {self.clustering_distance:.0f}px (looser grouping)")
                    elif key in [81, 68, 97]:   # Various left arrow codes or 'a'
                        self.clustering_distance = max(20.0, self.clustering_distance - 20.0)
                        logger.info(f"Clustering distance decreased to {self.clustering_distance:.0f}px (tighter grouping)")
                else:
                    # In headless mode, auto-save periodic screenshots
                    if self.frame_count % 30 == 0:  # Every 30 frames
                        screenshot_count += 1
                        screenshot_path = f"frame_{screenshot_count:04d}.jpg"
                        cv2.imwrite(screenshot_path, final_frame)
                        logger.info(f"Frame saved: {screenshot_path}")
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.03)  # ~30 FPS
                
                # Print periodic stats
                if self.frame_count % 100 == 0:
                    elapsed_time = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed_time
                    logger.info(f"Processed {self.frame_count} frames, "
                              f"Average FPS: {avg_fps:.2f}")
            
            # Cleanup
            if video_writer:
                video_writer.release()
                logger.info("Video saved successfully")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
        finally:
            self._cleanup()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for object selection"""
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click to select
            self._select_object_at_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to deselect
            self._deselect_object_at_point(x, y)
    
    def _select_object_at_point(self, x, y):
        """Select object at clicked point"""
        for i, (x1, y1, x2, y2) in enumerate(self.current_detections):
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Create new tracker for this detection
                tracker_id = self.next_tracker_id
                self.next_tracker_id += 1
                
                try:
                    # Use CSRT tracker (more accurate but slower)
                    tracker = cv2.TrackerCSRT_create()
                    
                    # Initialize tracker with current frame and bounding box
                    success = tracker.init(self.current_frame, (x1, y1, x2-x1, y2-y1))
                    
                    if success:
                        self.selected_trackers[tracker_id] = tracker
                        self.selected_boxes[tracker_id] = (x1, y1, x2, y2)
                        logger.info(f"Selected object {tracker_id} at ({x1}, {y1}) - ({x2}, {y2})")
                    else:
                        logger.warning(f"Failed to initialize tracker for object at ({x}, {y})")
                        
                except Exception as e:
                    logger.error(f"Error creating tracker: {e}")
                    # Fallback to KCF tracker
                    try:
                        tracker = cv2.TrackerKCF_create()
                        success = tracker.init(self.current_frame, (x1, y1, x2-x1, y2-y1))
                        if success:
                            self.selected_trackers[tracker_id] = tracker
                            self.selected_boxes[tracker_id] = (x1, y1, x2, y2)
                            logger.info(f"Selected object {tracker_id} with KCF tracker")
                    except Exception as e2:
                        logger.error(f"Failed to create fallback tracker: {e2}")
                
                break  # Only select first matching box
    
    def _deselect_object_at_point(self, x, y):
        """Deselect object at clicked point"""
        to_remove = []
        for tracker_id, (x1, y1, x2, y2) in self.selected_boxes.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            if tracker_id in self.selected_trackers:
                del self.selected_trackers[tracker_id]
            if tracker_id in self.selected_boxes:
                del self.selected_boxes[tracker_id]
            logger.info(f"Deselected object {tracker_id}")
    
    def _update_trackers(self, frame):
        """Update all active trackers"""
        to_remove = []
        
        for tracker_id, tracker in self.selected_trackers.items():
            try:
                success, bbox = tracker.update(frame)
                
                if success:
                    # Update the bounding box
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    
                    # Keep box within frame bounds
                    frame_h, frame_w = frame.shape[:2]
                    x1 = max(0, min(x1, frame_w))
                    y1 = max(0, min(y1, frame_h))
                    x2 = max(0, min(x2, frame_w))
                    y2 = max(0, min(y2, frame_h))
                    
                    self.selected_boxes[tracker_id] = (x1, y1, x2, y2)
                else:
                    # Tracking failed, mark for removal
                    to_remove.append(tracker_id)
                    logger.warning(f"Tracking failed for object {tracker_id}")
                    
            except Exception as e:
                logger.error(f"Error updating tracker {tracker_id}: {e}")
                to_remove.append(tracker_id)
        
        # Remove failed trackers
        for tracker_id in to_remove:
            if tracker_id in self.selected_trackers:
                del self.selected_trackers[tracker_id]
            if tracker_id in self.selected_boxes:
                del self.selected_boxes[tracker_id]

    def _cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error as e:
            if "not implemented" not in str(e).lower():
                raise
            # Ignore GUI not available error
        logger.info("Resources cleaned up")

def find_best_model() -> Path:
    """Find the best trained model"""
    potential_paths = [
        MODEL_DIR / "best_model.pt",
        MODEL_DIR / "best.pt"
    ]
    
    # Also check training results directory
    if (MODEL_DIR.parent / "results" / "training").exists():
        training_dirs = list((MODEL_DIR.parent / "results" / "training").glob("yolov11n_people_detection*"))
        for training_dir in training_dirs:
            weights_path = training_dir / "weights" / "best.pt"
            if weights_path.exists():
                potential_paths.append(weights_path)
    
    for path in potential_paths:
        if path.exists():
            logger.info(f"Found trained model: {path}")
            return path
    
    return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLOv11 Webcam People Detection")
    parser.add_argument('--model', type=str, default='', help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--clustering', type=float, default=120.0, help='Clustering distance in pixels')
    parser.add_argument('--save-video', action='store_true', help='Save video output')
    parser.add_argument('--output', type=str, default='webcam_output.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_best_model()
    
    if not model_path or not model_path.exists():
        logger.error("No valid model found for prediction!")
        logger.info("Please train a model first or specify a valid model path with --model")
        # Try to use the pretrained model as fallback
        pretrained_path = Path(__file__).parent.parent.parent.parent / "yolo11n.pt"
        if pretrained_path.exists():
            logger.info(f"Using pretrained model: {pretrained_path}")
            model_path = pretrained_path
        else:
            sys.exit(1)
    
    try:
        # Create predictor
        predictor = WebcamPredictor(
            model_path=str(model_path),
            conf_threshold=args.conf,
            device=args.device,
            camera_id=args.camera,
            clustering_distance=args.clustering
        )
        
        # Run prediction
        predictor.run(
            save_video=args.save_video,
            output_path=args.output if args.save_video else None
        )
        
        logger.info("Webcam prediction finished successfully!")
        
    except Exception as e:
        logger.error(f"Webcam prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()