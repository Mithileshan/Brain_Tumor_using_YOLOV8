"""
Inference module for YOLO predictions
"""

import logging
import os
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOPredictor:
    """Make predictions using YOLO"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def predict_image(self, image_path: str) -> Dict:
        """
        Predict on single image
        
        Returns:
            dict with detection results
        """
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            imgsz=640,
            verbose=False
        )
        
        if len(results) == 0:
            return {"image_path": str(image_path), "detections": []}
        
        result = results[0]
        detections = []
        
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class": int(cls),
                    "class_name": "tumor"
                })
        
        return {
            "image_path": str(image_path),
            "num_detections": len(detections),
            "detections": detections
        }
    
    def predict_folder(self, folder_path: str, output_path: str = None) -> List[Dict]:
        """
        Predict on all images in folder
        
        Returns:
            list of predictions
        """
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        predictions = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            pred = self.predict_image(img_path)
            predictions.append(pred)
        
        return predictions
