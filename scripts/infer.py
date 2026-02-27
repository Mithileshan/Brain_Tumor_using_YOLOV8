#!/usr/bin/env python3
"""
YOLOv8 Inference Script for Brain Tumor Detection

Usage:
    python scripts/infer.py --model models/best.pt --image image.jpg
    python scripts/infer.py --model models/best.pt --image image.jpg --output results/
    python scripts/infer.py --model models/best.pt --source data/images/ --output results/
    python scripts/infer.py --help
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on images using YOLOv8 brain tumor detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scripts/infer.py --model models/best.pt --image sample.jpg
  
  # Single image with output
  python scripts/infer.py --model models/best.pt --image sample.jpg --output results/
  
  # Batch inference on directory
  python scripts/infer.py --model models/best.pt --source data/test/images/ --output results/
  
  # Batch inference with confidence threshold
  python scripts/infer.py --model models/best.pt --source data/test/images/ --conf 0.5 --output results/
  
  # GPU device
  python scripts/infer.py --model models/best.pt --image sample.jpg --device 0
  
  # CPU device
  python scripts/infer.py --model models/best.pt --image sample.jpg --device cpu
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (e.g., models/best.pt)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single image file for inference'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Path to image directory or video file for batch inference'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/',
        help='Output directory for predictions (default: results/)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5, range: 0.0-1.0)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45, range: 0.0-1.0)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (default: 0 for GPU, cpu for CPU)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for inference (default: 640)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save predicted images with bounding boxes'
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate input arguments."""
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {args.model}")
        print(f"   Expected: {model_path.absolute()}")
        sys.exit(1)
    
    # Check that either --image or --source is provided
    if not args.image and not args.source:
        print("❌ Error: Either --image or --source must be provided")
        sys.exit(1)
    
    # Check image/source exists
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Error: Image not found: {args.image}")
            sys.exit(1)
    
    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"❌ Error: Source not found: {args.source}")
            sys.exit(1)
    
    # Validate thresholds
    if not (0.0 <= args.conf <= 1.0):
        print(f"❌ Error: Confidence threshold must be 0.0-1.0, got {args.conf}")
        sys.exit(1)
    
    if not (0.0 <= args.iou <= 1.0):
        print(f"❌ Error: IoU threshold must be 0.0-1.0, got {args.iou}")
        sys.exit(1)
    
    return True


def main():
    """Main inference function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🧠 YOLOv8 Brain Tumor Detection - Inference Script")
    print("="*70 + "\n")
    
    # Validate arguments
    validate_args(args)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_path.absolute()}\n")
    
    # Load model
    print(f"✓ Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    print(f"✓ Running inference...")
    print(f"  - Confidence threshold: {args.conf}")
    print(f"  - IoU threshold: {args.iou}")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Device: {args.device}\n")
    
    try:
        # Determine source
        source = args.image if args.image else args.source
        
        # Run predictions
        results = model.predict(
            source=source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            project=str(output_path),
            name='',
            verbose=args.verbose,
        )
        
        # Process results
        print("="*70)
        print("✅ Inference Complete!")
        print("="*70 + "\n")
        
        total_detections = 0
        for idx, result in enumerate(results):
            image_name = Path(result.path).name if hasattr(result, 'path') else f"result_{idx}"
            boxes = result.boxes if hasattr(result, 'boxes') else None
            
            if boxes is not None:
                num_boxes = len(boxes)
                total_detections += num_boxes
                print(f"📷 {image_name}")
                print(f"   Detections: {num_boxes}")
                
                if num_boxes > 0 and boxes.conf is not None:
                    confidences = boxes.conf.cpu().numpy()
                    print(f"   Confidence(s): {', '.join([f'{c:.2%}' for c in confidences])}")
                print()
        
        print(f"📊 Summary:")
        print(f"   Total images processed: {len(results)}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per image: {total_detections / len(results):.2f}")
        
        if args.save:
            print(f"\n💾 Predictions saved to: {output_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
