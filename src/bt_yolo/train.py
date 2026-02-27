"""
Training module for YOLOv8
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.bt_yolo.config import Config


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def train_model(config: Config) -> dict:
    """
    Train YOLO model
    
    Returns:
        dict with training results
    """
    logger.info("=" * 60)
    logger.info("BRAIN TUMOR YOLO TRAINING - PHASE 1")
    logger.info("=" * 60)
    
    # Validate config
    config.validate()
    logger.info(f"Config: {config.to_dict()}")
    
    # Load model
    logger.info(f"Loading model: {config.model}")
    model = YOLO(config.model)
    
    # Train
    logger.info("Starting training...")
    results = model.train(
        data=config.data_yaml,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch_size,
        device=config.device,
        lr0=config.lr0,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        patience=config.patience,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        flipud=config.flipud,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        project=config.project,
        name=config.name,
        verbose=config.verbose,
    )
    
    # Create output directory
    output_dir = os.path.join(config.project, config.name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and save metrics
    metrics = {}
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
    
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model'] = config.model
    metrics['epochs'] = config.epochs
    metrics['batch_size'] = config.batch_size
    metrics['imgsz'] = config.imgsz
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    
    if hasattr(results, 'box'):
        logger.info(f"mAP50: {results.box.map50:.4f}")
        logger.info(f"mAP50-95: {results.box.map:.4f}")
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    config.to_json(config_path)
    logger.info(f"Config saved: {config_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: str(v) for k, v in metrics.items()}, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")
    
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector for brain tumors")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (use 5 for smoke test)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0=GPU, cpu=CPU)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    parser.add_argument("--name", type=str, default="train", help="Experiment name")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    config = Config(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=args.verbose
    )
    
    try:
        metrics = train_model(config)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
