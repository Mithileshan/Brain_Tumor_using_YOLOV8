"""
Evaluation module for YOLO models
"""

import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOEvaluator:
    """Evaluate YOLO models"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
    
    def evaluate(self, data_yaml: str):
        """
        Evaluate model
        
        Args:
            data_yaml: path to data.yaml
            
        Returns:
            evaluation results object
        """
        results = self.model.val(data=data_yaml, imgsz=640, val=True)
        return results
