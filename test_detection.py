#!/usr/bin/env python
"""Quick test of YOLO detection with trained model"""

from src.bt_yolo.predict import YOLOPredictor
import os
import glob

# Find first test image
test_images = glob.glob('data/test/images/*.jpg')
if test_images:
    test_img = test_images[0]
    print(f'Testing: {test_img}')
    # Use newly trained model
    predictor = YOLOPredictor('runs/detect/runs/detect/train_custom/weights/best.pt', conf_threshold=0.5)
    result = predictor.predict_image(test_img)
    print(f'Num Detections: {result["num_detections"]}')
    if result['detections']:
        for i, det in enumerate(result['detections']):
            print(f'  Detection {i+1}: class={det["class_name"]}, confidence={det["confidence"]:.4f}')
    else:
        print('  No tumors detected (model may need more training)')
else:
    print('No test images found in data/test/images/')
