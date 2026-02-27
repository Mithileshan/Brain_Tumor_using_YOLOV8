#!/usr/bin/env python3
"""
YOLOv8 Training Script for Brain Tumor Detection

Usage:
    python scripts/train.py --data data.yaml --epochs 50 --imgsz 640 --batch 16
    python scripts/train.py --data data.yaml --epochs 100 --device 0  # GPU
    python scripts/train.py --help
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"❌ Error: Required packages not installed.\nRun: pip install -r requirements.txt")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for brain tumor detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --data data.yaml --epochs 50
  python scripts/train.py --data data.yaml --epochs 50 --device 0,1
  python scripts/train.py --data data.yaml --epochs 100 --batch 32 --imgsz 1024
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to dataset config file (default: data.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16). Use larger batch for more memory'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640). Options: 320, 640, 1024'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (default: 0 for GPU 0, or cpu for CPU). Use "0,1" for multi-GPU'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Model variant to use (default: yolov8n.pt). Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='brain_tumor_detector',
        help='Training run name (default: brain_tumor_detector)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory to save runs (default: runs/detect)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50 epochs)'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed training information'
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate input arguments."""
    # Check data.yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data config file not found: {args.data}")
        print(f"   Expected: {data_path.absolute()}")
        sys.exit(1)
    
    # Check model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"⚠️  Warning: Model file not found: {args.model}")
        print(f"   Will download pre-trained YOLOv8 model from ultralytics...")
    
    # Validate image size
    valid_sizes = [320, 416, 480, 512, 576, 608, 640, 1024]
    if args.imgsz not in valid_sizes:
        print(f"⚠️  Warning: Image size {args.imgsz} is unusual. Recommended: {valid_sizes}")
    
    return True


def setup_device(device_str):
    """Setup PyTorch device."""
    if device_str.lower() == 'cpu':
        return 'cpu'
    
    try:
        if torch.cuda.is_available():
            return device_str  # e.g., '0' or '0,1'
        else:
            print("⚠️  Warning: CUDA not available. Falling back to CPU")
            return 'cpu'
    except Exception as e:
        print(f"⚠️  Warning: {e}. Using CPU")
        return 'cpu'


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🧠 YOLOv8 Brain Tumor Detection - Training Script")
    print("="*70 + "\n")
    
    # Validate arguments
    validate_args(args)
    
    # Setup device
    device = setup_device(args.device)
    print(f"✓ Device: {device}")
    
    # Load model
    print(f"✓ Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Print model info
    if args.verbose:
        print(f"\nModel Summary:")
        print(model.info())
    
    # Start training
    print(f"\n✓ Starting training...")
    print(f"  - Data: {args.data}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Project: {args.project}")
    print(f"  - Name: {args.name}")
    print()
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            resume=args.resume,
            verbose=args.verbose,
            # Additional training parameters
            augment=True,
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,    # HSV-Saturation augmentation
            hsv_v=0.4,    # HSV-Value augmentation
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            # Model parameters
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            # Validation
            val=True,
        )
        
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        print(f"\n📊 Results saved to: {args.project}/{args.name}")
        print(f"🏆 Best model: {args.project}/{args.name}/weights/best.pt")
        print(f"📈 Training metrics: {args.project}/{args.name}/results.csv")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\n📋 Final Metrics:")
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
