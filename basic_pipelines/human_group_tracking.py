"""
Human Group Tracking Pipeline
- Detects humans and draws bounding boxes
- Clusters groups of humans based on spatial density
- Marks center of largest group and frame center
- Calculates pan/tilt adjustments to center the largest group
"""

from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import argparse
from sklearn.cluster import DBSCAN

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self, clustering_eps=100, clustering_min_samples=2):
        super().__init__()
        # Clustering parameters
        self.clustering_eps = clustering_eps  # Maximum distance between two samples in a cluster
        self.clustering_min_samples = clustering_min_samples  # Minimum samples to form a cluster
        
        # Store latest frame dimensions
        self.frame_width = None
        self.frame_height = None
        
        # Store latest pan/tilt values
        self.pan_offset = 0
        self.tilt_offset = 0
        self.largest_group_center = None

    def set_clustering_params(self, eps, min_samples):
        """Update clustering parameters"""
        self.clustering_eps = eps
        self.clustering_min_samples = min_samples

    def get_frame_center(self):
        """Get the center of the frame"""
        if self.frame_width and self.frame_height:
            return (self.frame_width // 2, self.frame_height // 2)
        return None

# -----------------------------------------------------------------------------------------------
# Clustering and visualization functions
# -----------------------------------------------------------------------------------------------

def get_bbox_center(bbox):
    """Calculate center point of a bounding box"""
    x_center = bbox.xmin() + (bbox.width() / 2)
    y_center = bbox.ymin() + (bbox.height() / 2)
    return (x_center, y_center)

def cluster_humans(human_centers, eps, min_samples):
    """
    Cluster human detections using DBSCAN algorithm
    
    Args:
        human_centers: List of (x, y) center coordinates
        eps: Maximum distance between points in a cluster
        min_samples: Minimum points to form a cluster
    
    Returns:
        labels: Cluster labels for each point (-1 for noise)
        n_clusters: Number of clusters found
    """
    if len(human_centers) < min_samples:
        return np.array([-1] * len(human_centers)), 0
    
    centers_array = np.array(human_centers)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
    
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    return labels, n_clusters

def get_group_bounding_box(human_bboxes, labels, cluster_id):
    """
    Calculate bounding box encompassing all humans in a cluster
    
    Args:
        human_bboxes: List of hailo bounding boxes
        labels: Cluster labels
        cluster_id: Target cluster ID
    
    Returns:
        (x_min, y_min, x_max, y_max) of the group bounding box
    """
    group_indices = np.where(labels == cluster_id)[0]
    
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    
    for idx in group_indices:
        bbox = human_bboxes[idx]
        x_mins.append(bbox.xmin())
        y_mins.append(bbox.ymin())
        x_maxs.append(bbox.xmin() + bbox.width())
        y_maxs.append(bbox.ymin() + bbox.height())
    
    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))

def get_group_center(human_bboxes, labels, cluster_id):
    """Calculate the center of a group"""
    group_bbox = get_group_bounding_box(human_bboxes, labels, cluster_id)
    center_x = (group_bbox[0] + group_bbox[2]) / 2
    center_y = (group_bbox[1] + group_bbox[3]) / 2
    return (center_x, center_y)

def calculate_pan_tilt(group_center, frame_center, frame_width, frame_height):
    """
    Calculate pan and tilt adjustments needed to center the group
    
    Args:
        group_center: (x, y) center of the largest group (normalized 0-1)
        frame_center: (x, y) center of the frame (pixels)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        pan: Horizontal adjustment (negative = left, positive = right)
        tilt: Vertical adjustment (negative = up, positive = down)
    """
    # Convert normalized coordinates to pixels
    group_x_pixels = group_center[0] * frame_width
    group_y_pixels = group_center[1] * frame_height
    
    # Calculate offset from center (in pixels)
    pan_offset = group_x_pixels - frame_center[0]
    tilt_offset = group_y_pixels - frame_center[1]
    
    # Convert to percentage of frame dimension for easier interpretation
    pan_percent = (pan_offset / frame_width) * 100
    tilt_percent = (tilt_offset / frame_height) * 100
    
    return pan_percent, tilt_percent

# -----------------------------------------------------------------------------------------------
# Color palette for different groups
# -----------------------------------------------------------------------------------------------
GROUP_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    
    # Store frame dimensions
    if width and height:
        user_data.frame_width = width
        user_data.frame_height = height

    # Get video frame if available
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Filter for human detections only
    human_detections = []
    human_bboxes = []
    human_centers = []
    
    for detection in detections:
        label = detection.get_label()
        if label == "person":
            human_detections.append(detection)
            bbox = detection.get_bbox()
            human_bboxes.append(bbox)
            center = get_bbox_center(bbox)
            human_centers.append(center)

    # Perform clustering if we have humans
    string_to_print = f"\n{'='*60}\n"
    string_to_print += f"Frame: {user_data.get_count()} | Humans detected: {len(human_detections)}\n"
    
    labels = np.array([])
    n_clusters = 0
    largest_cluster_id = -1
    largest_cluster_size = 0
    
    if len(human_centers) > 0:
        # Scale centers to pixel coordinates for clustering
        scaled_centers = [(c[0] * width, c[1] * height) for c in human_centers]
        labels, n_clusters = cluster_humans(
            scaled_centers, 
            user_data.clustering_eps, 
            user_data.clustering_min_samples
        )
        
        string_to_print += f"Groups found: {n_clusters}\n"
        
        # Find the largest cluster
        if n_clusters > 0:
            for cluster_id in range(n_clusters):
                cluster_size = np.sum(labels == cluster_id)
                if cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    largest_cluster_id = cluster_id
            
            string_to_print += f"Largest group: {largest_cluster_id} with {largest_cluster_size} humans\n"

    # Draw on frame if available
    if user_data.use_frame and frame is not None:
        frame_center = (width // 2, height // 2)
        
        # Draw individual human bounding boxes
        for i, (detection, bbox) in enumerate(zip(human_detections, human_bboxes)):
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int((bbox.xmin() + bbox.width()) * width)
            y2 = int((bbox.ymin() + bbox.height()) * height)
            
            # Color based on cluster
            if len(labels) > i and labels[i] >= 0:
                color = GROUP_COLORS[labels[i] % len(GROUP_COLORS)]
            else:
                color = (128, 128, 128)  # Gray for unclustered
            
            # Draw human bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            confidence = detection.get_confidence()
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw group bounding boxes
        for cluster_id in range(n_clusters):
            group_bbox = get_group_bounding_box(human_bboxes, labels, cluster_id)
            
            x1 = int(group_bbox[0] * width)
            y1 = int(group_bbox[1] * height)
            x2 = int(group_bbox[2] * width)
            y2 = int(group_bbox[3] * height)
            
            color = GROUP_COLORS[cluster_id % len(GROUP_COLORS)]
            thickness = 4 if cluster_id == largest_cluster_id else 2
            
            # Add padding around group
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label the group
            group_size = np.sum(labels == cluster_id)
            label_text = f"Group {cluster_id}: {group_size} people"
            if cluster_id == largest_cluster_id:
                label_text += " (LARGEST)"
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw group center
            group_center = get_group_center(human_bboxes, labels, cluster_id)
            center_x = int(group_center[0] * width)
            center_y = int(group_center[1] * height)
            
            if cluster_id == largest_cluster_id:
                # Draw larger marker for largest group center
                cv2.circle(frame, (center_x, center_y), 15, color, -1)
                cv2.circle(frame, (center_x, center_y), 18, (255, 255, 255), 2)
                user_data.largest_group_center = (center_x, center_y)
            else:
                cv2.circle(frame, (center_x, center_y), 8, color, -1)
        
        # Draw frame center (crosshair)
        crosshair_size = 20
        cv2.line(frame, (frame_center[0] - crosshair_size, frame_center[1]), 
                (frame_center[0] + crosshair_size, frame_center[1]), (255, 255, 255), 2)
        cv2.line(frame, (frame_center[0], frame_center[1] - crosshair_size), 
                (frame_center[0], frame_center[1] + crosshair_size), (255, 255, 255), 2)
        cv2.circle(frame, frame_center, 5, (255, 255, 255), -1)
        
        # Calculate and display pan/tilt if we have a largest group
        if largest_cluster_id >= 0:
            group_center = get_group_center(human_bboxes, labels, largest_cluster_id)
            pan, tilt = calculate_pan_tilt(group_center, frame_center, width, height)
            
            user_data.pan_offset = pan
            user_data.tilt_offset = tilt
            
            # Draw line from frame center to group center
            group_center_px = (int(group_center[0] * width), int(group_center[1] * height))
            cv2.line(frame, frame_center, group_center_px, (0, 255, 255), 2)
            
            # Display pan/tilt info on frame
            pan_direction = "RIGHT" if pan > 0 else "LEFT"
            tilt_direction = "DOWN" if tilt > 0 else "UP"
            
            info_y = 30
            cv2.putText(frame, f"Pan: {abs(pan):.1f}% {pan_direction}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Tilt: {abs(tilt):.1f}% {tilt_direction}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            string_to_print += f"\n--- Pan/Tilt Adjustment ---\n"
            string_to_print += f"Pan:  {pan:+.2f}% ({pan_direction})\n"
            string_to_print += f"Tilt: {tilt:+.2f}% ({tilt_direction})\n"
        
        # Display clustering parameters
        cv2.putText(frame, f"Clustering: eps={user_data.clustering_eps}, min={user_data.clustering_min_samples}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display human and group count
        cv2.putText(frame, f"Humans: {len(human_detections)} | Groups: {n_clusters}", 
                   (10, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    string_to_print += f"{'='*60}\n"
    print(string_to_print)
    
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Human Group Tracking with Pan/Tilt Guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--clustering-eps', 
        type=float, 
        default=100,
        help='DBSCAN epsilon: Maximum distance (in pixels) between two humans to be in the same cluster. '
             'Lower values = tighter groups, higher values = looser groups.'
    )
    
    parser.add_argument(
        '--clustering-min-samples', 
        type=int, 
        default=2,
        help='Minimum number of humans required to form a group/cluster.'
    )
    
    # Parse known args to allow GStreamer to handle its own args
    args, _ = parser.parse_known_args()
    return args

# -----------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("Human Group Tracking Pipeline")
    print("="*60)
    print(f"Clustering Parameters:")
    print(f"  - eps (max distance): {args.clustering_eps} pixels")
    print(f"  - min_samples: {args.clustering_min_samples} humans")
    print("="*60 + "\n")
    
    # Create an instance of the user app callback class with clustering parameters
    user_data = user_app_callback_class(
        clustering_eps=args.clustering_eps,
        clustering_min_samples=args.clustering_min_samples
    )
    
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
