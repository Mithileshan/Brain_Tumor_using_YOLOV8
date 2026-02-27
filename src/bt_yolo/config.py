"""
Configuration module for YOLO training & inference
"""

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from typing import Optional


@dataclass
class Config:
    """YOLO training configuration"""
    
    # Data
    data_yaml: str = "data.yaml"
    
    # Model
    model: str = "yolov8n.pt"  # nano, small, medium, large, xlarge
    imgsz: int = 640
    
    # Training
    epochs: int = 100
    batch_size: int = 16
    device: str = "0"  # "0" for GPU:0, "cpu" for CPU, "0,1" for multi-GPU
    
    # Optimization
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    patience: int = 50  # early stopping
    
    # Data augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    
    # Output
    project: str = "runs/detect"
    name: str = "train"
    save_period: int = -1  # disabled
    verbose: bool = True
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save config to JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def validate(self):
        """Validate configuration"""
        if not Path(self.data_yaml).exists():
            raise FileNotFoundError(f"data.yaml not found: {self.data_yaml}")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        
        if self.imgsz % 32 != 0:
            raise ValueError("imgsz must be divisible by 32")
