"""
Detection + Classification Pipeline
=====================================
Two-stage architecture:
    1. YOLOv8 detects traffic signs in full scene images (single-class)
    2. Baseline CNN classifies each detected sign into 43 categories

Usage:
    from src.pipeline.detect_and_classify import TrafficSignPipeline
    
    pipeline = TrafficSignPipeline(
        detector_path='weights/yolov8n_single_class.pt',
        classifier_path='weights/best_baseline_cnn.pth'
    )
    results = pipeline.run('path/to/dashcam_image.jpg')
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from ultralytics import YOLO

from src.models.baseline_cnn import BaselineCNN


# GTSRB class names
CLASS_NAMES = [
    'Speed limit 20', 'Speed limit 30', 'Speed limit 50', 'Speed limit 60',
    'Speed limit 70', 'Speed limit 80', 'End speed limit 80', 'Speed limit 100',
    'Speed limit 120', 'No overtaking', 'No overtaking trucks',
    'Priority at next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'No trucks', 'No entry', 'General caution',
    'Dangerous curve left', 'Dangerous curve right', 'Double curve',
    'Bumpy road', 'Slippery road', 'Road narrows right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware ice/snow', 'Wild animals crossing', 'End all limits',
    'Right turn ahead', 'Left turn ahead', 'Ahead only',
    'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout', 'End no overtaking', 'End no overtaking trucks'
]

# Preprocessing transform (must match training)
CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TrafficSignPipeline:
    """End-to-end traffic sign detection and classification.

    Args:
        detector_path: path to trained YOLOv8 weights
        classifier_path: path to trained BaselineCNN weights
        device: 'cuda' or 'cpu'
        conf_threshold: minimum detection confidence
    """

    def __init__(self, detector_path, classifier_path, device=None,
                 conf_threshold=0.25):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.conf_threshold = conf_threshold

        # Load detector
        self.detector = YOLO(detector_path)

        # Load classifier
        self.classifier = BaselineCNN(num_classes=43).to(self.device)
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device)
        )
        self.classifier.eval()

    def run(self, image_path):
        """Run full pipeline on a single image.

        Args:
            image_path: path to input image

        Returns:
            img_rgb: original image as RGB numpy array
            results: list of dicts with bbox, class_name, confidences
        """
        # Step 1: Detect signs
        detections = self.detector.predict(
            image_path, conf=self.conf_threshold, imgsz=640,
            device=0 if self.device.type == 'cuda' else 'cpu',
            verbose=False
        )

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        results = []

        if len(detections[0].boxes) == 0:
            return img_rgb, results

        # Step 2: Classify each detected sign
        for box in detections[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0].cpu().numpy())

            # Pad the crop by 10%
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)

            crop = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue

            # Apply CLAHE (same as training)
            crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            crop_lab[:, :, 0] = clahe.apply(crop_lab[:, :, 0])
            crop_enhanced = cv2.cvtColor(crop_lab, cv2.COLOR_LAB2RGB)

            # Classify
            input_tensor = CLASSIFY_TRANSFORM(crop_enhanced).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.classifier(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                cls_conf, cls_pred = probabilities.max(1)

            results.append({
                'bbox': (x1, y1, x2, y2),
                'det_conf': det_conf,
                'class_id': cls_pred.item(),
                'class_name': CLASS_NAMES[cls_pred.item()],
                'class_conf': float(cls_conf.item())
            })

        return img_rgb, results

    def visualize(self, image_path, save_path=None):
        """Run pipeline and display annotated result."""
        img_rgb, results = self.run(image_path)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(img_rgb)

        for r in results:
            x1, y1, x2, y2 = r['bbox']
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            label = f"{r['class_name']} ({r['class_conf']:.0%})"
            ax.text(x1, y1 - 8, label, color='white', fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='green', alpha=0.8))

        ax.set_title(f'{len(results)} sign(s) detected', fontsize=14)
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        return results
