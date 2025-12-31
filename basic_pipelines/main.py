#!/usr/bin/env python3
"""
YOLOv11 People Detection - Main Launcher Script
Provides a simple interface to run training, testing, webcam prediction, or Hailo group tracking.

For Hailo-based detection with RPi camera and group tracking:
    python main.py hailo --input rpi
    
For webcam-based detection:
    python main.py webcam
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_script(script_name: str, args: list = None, use_src: bool = True):
    """
    Run a script with optional arguments
    
    Args:
        script_name (str): Name of the script to run
        args (list): Additional arguments
        use_src (bool): Whether to look in src directory
    """
    if use_src:
        script_path = Path(__file__).parent / "src" / script_name
    else:
        script_path = Path(__file__).parent / script_name
    
    cmd = [sys.executable, str(script_path)]
    
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("Script interrupted by user")
        sys.exit(1)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 People Detection - Main Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                           # Train with defaults
  python main.py train --epochs 200 --batch-size 32    # Custom training
  python main.py test                            # Test trained model  
  python main.py test --save-images              # Test and save images
  python main.py webcam                          # Real-time webcam (ultralytics)
  python main.py webcam --save-video             # Record webcam output
  
  # Hailo-based detection with RPi camera (recommended for Pi):
  python main.py hailo --input rpi               # Run with RPi camera
  python main.py hailo --input rpi --use-frame   # With frame overlay
  python main.py hailo --input usb               # Run with USB camera
  python main.py hailo --hef-path /path/to/model.hef  # Custom HEF model
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['train', 'test', 'webcam', 'hailo'],
        help='Operation mode: train, test, webcam (ultralytics), or hailo (Hailo-accelerated with group tracking)'
    )
    
    # Parse known args to allow passing through to sub-scripts
    args, unknown = parser.parse_known_args()
    
    # Map modes to scripts
    script_map = {
        'train': ('train.py', True),
        'test': ('test.py', True), 
        'webcam': ('predict_webcam.py', True),
        'hailo': ('human_group_tracking.py', False)  # In basic_pipelines, not src
    }
    
    # Print header
    print("=" * 60)
    print("🎯 YOLOv11 People Detection")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    if unknown:
        print(f"Additional arguments: {' '.join(unknown)}")
    print("=" * 60)
    
    # Run the appropriate script
    script_name, use_src = script_map[args.mode]
    run_script(script_name, unknown, use_src=use_src)
    
    print("=" * 60)
    print("✅ Operation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()