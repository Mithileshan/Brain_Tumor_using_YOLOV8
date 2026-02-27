"""
YOLO Evaluation script - generates metrics and sample predictions
Phase 3: Comprehensive evaluation with visualizations
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bt_yolo.config import Config
from ultralytics import YOLO
import numpy as np


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def evaluate_yolo(model_path: str, data_yaml: str = "data.yaml", output_dir: str = "reports"):
    """
    YOLO evaluation pipeline
    """
    logger.info("=" * 70)
    logger.info("BRAIN TUMOR YOLO EVALUATION - PHASE 3")
    logger.info("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Validate on test set
    logger.info("Running evaluation on test set...")
    results = model.val(data=data_yaml, imgsz=640, conf=0.5, verbose=True)
    
    # Extract metrics
    metrics_dict = {}
    if hasattr(results, 'box'):
        metrics_dict = {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
            "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else None,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
        }
    
    metrics_dict['timestamp'] = datetime.now().isoformat()
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics_yolo.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: str(v) for k, v in metrics_dict.items()}, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 70)
    for key, value in metrics_dict.items():
        if key != 'timestamp':
            logger.info(f"{key}: {value}")
    
    logger.info(f"\nMetrics saved: {metrics_path}")
    logger.info("=" * 70)
    
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", 
                       help="Model path")
    parser.add_argument("--data", type=str, default="data.yaml", help="Data yaml")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    try:
        metrics = evaluate_yolo(args.model, args.data, args.output_dir)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
