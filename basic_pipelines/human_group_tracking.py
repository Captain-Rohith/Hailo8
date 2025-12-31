"""
Human Group Tracking - With Visual Overlays
Detects humans, clusters by proximity, shows pan/tilt to center largest group.
Draws cluster boxes, center markers, and connection line on video.

Usage:
    python human_group_tracking.py --input rpi --use-frame
"""

from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import sys
import hailo
import argparse
import cv2
import numpy as np

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# Default paths
DEFAULT_CUSTOM_HEF = "/usr/local/hailo/resources/models/hailo8l/best.hef"
DEFAULT_LABELS_JSON = str(Path(__file__).resolve().parent.parent / "local_resources" / "best_person.json")

# -----------------------------------------------------------------------------------------------
# Simple distance-based clustering (no sklearn needed)
# -----------------------------------------------------------------------------------------------
def simple_cluster(centers, eps):
    """
    Simple clustering: group points within eps distance of each other.
    Uses union-find approach for transitive clustering.
    """
    n = len(centers)
    if n == 0:
        return []
    
    # Initialize each point as its own cluster
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union points that are close enough
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < eps:
                union(i, j)
    
    # Convert to cluster labels
    root_to_label = {}
    labels = []
    next_label = 0
    
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels.append(root_to_label[root])
    
    return labels

# -----------------------------------------------------------------------------------------------
# User data class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self, eps=100):
        super().__init__()
        self.eps = eps
        self.frame_count = 0

# -----------------------------------------------------------------------------------------------
# Callback - draws clusters, centers, and pan/tilt on frame
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.frame_count += 1
    
    # Get frame dimensions
    fmt, width, height = get_caps_from_pad(pad)
    if width is None:
        width, height = 1280, 720
    
    # Get frame for drawing (only if use_frame is enabled)
    frame = None
    if user_data.use_frame and fmt is not None:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)
    
    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # All detections are persons (single class model)
    persons = list(detections)
    n_persons = len(persons)
    
    # Frame center
    frame_cx, frame_cy = width // 2, height // 2
    
    # Draw frame center marker if we have a frame
    if frame is not None:
        # Draw "+" at frame center (white with black outline)
        cv2.line(frame, (frame_cx - 20, frame_cy), (frame_cx + 20, frame_cy), (0, 0, 0), 4)
        cv2.line(frame, (frame_cx, frame_cy - 20), (frame_cx, frame_cy + 20), (0, 0, 0), 4)
        cv2.line(frame, (frame_cx - 20, frame_cy), (frame_cx + 20, frame_cy), (255, 255, 255), 2)
        cv2.line(frame, (frame_cx, frame_cy - 20), (frame_cx, frame_cy + 20), (255, 255, 255), 2)
    
    if n_persons == 0:
        if frame is not None:
            cv2.putText(frame, "No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)
        return Gst.PadProbeReturn.OK
    
    # Get centers in pixels for clustering
    centers = []
    for det in persons:
        bbox = det.get_bbox()
        cx = int((bbox.xmin() + bbox.xmax()) / 2 * width)
        cy = int((bbox.ymin() + bbox.ymax()) / 2 * height)
        centers.append((cx, cy))
    
    # Cluster persons
    labels = simple_cluster(centers, user_data.eps)
    
    # Group by cluster
    clusters = {}
    for i, lbl in enumerate(labels):
        if lbl not in clusters:
            clusters[lbl] = []
        clusters[lbl].append(i)
    
    # Find largest cluster (with 2+ members)
    largest_id = -1
    largest_size = 0
    for cid, members in clusters.items():
        if len(members) >= 2 and len(members) > largest_size:
            largest_size = len(members)
            largest_id = cid
    
    # Colors for clusters
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    largest_group_center = None
    
    if frame is not None:
        # Draw each cluster
        for cid, members in clusters.items():
            if len(members) < 2:
                continue
            
            color = colors[cid % len(colors)]
            is_largest = (cid == largest_id)
            
            # Calculate cluster bounding box
            min_x = min_y = float('inf')
            max_x = max_y = 0
            
            for idx in members:
                bbox = persons[idx].get_bbox()
                min_x = min(min_x, int(bbox.xmin() * width))
                min_y = min(min_y, int(bbox.ymin() * height))
                max_x = max(max_x, int(bbox.xmax() * width))
                max_y = max(max_y, int(bbox.ymax() * height))
            
            # Add padding
            pad = 15
            min_x = max(0, min_x - pad)
            min_y = max(0, min_y - pad)
            max_x = min(width, max_x + pad)
            max_y = min(height, max_y + pad)
            
            # Draw cluster box
            thickness = 4 if is_largest else 2
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, thickness)
            
            # Cluster center
            group_cx = (min_x + max_x) // 2
            group_cy = (min_y + max_y) // 2
            
            # Draw "+" at group center
            size = 15 if is_largest else 10
            cv2.line(frame, (group_cx - size, group_cy), (group_cx + size, group_cy), color, 3)
            cv2.line(frame, (group_cx, group_cy - size), (group_cx, group_cy + size), color, 3)
            
            # Label
            label = f"GROUP {len(members)}" if is_largest else f"grp {len(members)}"
            cv2.putText(frame, label, (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if is_largest:
                largest_group_center = (group_cx, group_cy)
        
        # Draw line from largest group center to frame center
        if largest_group_center is not None:
            cv2.line(frame, largest_group_center, (frame_cx, frame_cy), (0, 255, 255), 2)
            
            # Calculate pan/tilt
            pan_px = largest_group_center[0] - frame_cx
            tilt_px = largest_group_center[1] - frame_cy
            pan_pct = (pan_px / width) * 100
            tilt_pct = (tilt_px / height) * 100
            
            pan_dir = "R" if pan_px > 0 else "L"
            tilt_dir = "D" if tilt_px > 0 else "U"
            
            # Draw pan/tilt info at top
            info_text = f"Pan: {abs(pan_pct):.0f}% {pan_dir}  Tilt: {abs(tilt_pct):.0f}% {tilt_dir}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Persons: {n_persons}  Groups: {len([c for c in clusters.values() if len(c)>=2])}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Persons: {n_persons} (no groups)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Convert and set frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Human Group Tracking")
    parser.add_argument('--clustering-eps', type=float, default=100,
                        help='Max distance (pixels) between persons to form a group (default: 100)')
    args, remaining = parser.parse_known_args()
    return args, remaining

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set env
    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    
    # Parse our args
    args, remaining = parse_args()
    
    print("=" * 60)
    print("Human Group Tracking")
    print(f"Clustering distance: {args.clustering_eps} pixels")
    print("=" * 60)
    
    # Rebuild sys.argv for GStreamerDetectionApp
    sys.argv = [sys.argv[0]] + remaining
    
    # Add --use-frame for visual overlays (required for drawing)
    if '--use-frame' not in sys.argv:
        sys.argv.append('--use-frame')
    
    # Add custom HEF if not specified
    if '--hef-path' not in sys.argv:
        sys.argv.extend(['--hef-path', DEFAULT_CUSTOM_HEF])
        print(f"Using model: {DEFAULT_CUSTOM_HEF}")
    
    # Add labels JSON if not specified
    if '--labels-json' not in sys.argv and Path(DEFAULT_LABELS_JSON).exists():
        sys.argv.extend(['--labels-json', DEFAULT_LABELS_JSON])
    
    # Create callback class
    user_data = user_app_callback_class(eps=args.clustering_eps)
    
    # Run
    print("\nStarting... (Ctrl+C to stop)\n")
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
