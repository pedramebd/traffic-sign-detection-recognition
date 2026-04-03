"""
Traffic Sign Detection & Recognition — Demo Script
====================================================
Run the full two-stage pipeline on any dashcam image.

Usage:
    python demo.py --image path/to/image.jpg
    python demo.py --image path/to/image.jpg --save output.png
    python demo.py --image path/to/image.jpg --conf 0.3
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.detect_and_classify import TrafficSignPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Traffic Sign Detection and Recognition Pipeline'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--detector', type=str,
                        default='weights/yolov8n_single_class.pt',
                        help='Path to YOLOv8 detector weights')
    parser.add_argument('--classifier', type=str,
                        default='weights/best_baseline_cnn.pth',
                        help='Path to CNN classifier weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save output image (optional)')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image not found at '{args.image}'")
        sys.exit(1)
    if not os.path.exists(args.detector):
        print(f"Error: Detector weights not found at '{args.detector}'")
        print("Please download or train the YOLOv8 model first.")
        sys.exit(1)
    if not os.path.exists(args.classifier):
        print(f"Error: Classifier weights not found at '{args.classifier}'")
        print("Please download or train the Baseline CNN first.")
        sys.exit(1)

    # Initialise pipeline
    print("Loading models...")
    pipeline = TrafficSignPipeline(
        detector_path=args.detector,
        classifier_path=args.classifier,
        conf_threshold=args.conf
    )

    # Run inference
    print(f"Processing: {args.image}")
    results = pipeline.visualize(args.image, save_path=args.save)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Detected {len(results)} traffic sign(s)")
    print(f"{'=' * 50}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['class_name']}")
        print(f"     Detection confidence:     {r['det_conf']:.1%}")
        print(f"     Classification confidence: {r['class_conf']:.1%}")
        print(f"     Bounding box: {r['bbox']}")

    if args.save:
        print(f"\nOutput saved to: {args.save}")


if __name__ == '__main__':
    main()
