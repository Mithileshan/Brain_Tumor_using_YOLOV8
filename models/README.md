# Model Checkpoints

This directory contains trained YOLOv8 models for brain tumor detection.

## Files

- **`best.pt`** - Best performing model checkpoint (lowest validation loss)
- **`last.pt`** - Last epoch checkpoint (useful for resuming training)

## Usage

### Training

```bash
python scripts/train.py --model yolov8n.pt --data data.yaml --epochs 100
```

This will create new checkpoints in `runs/detect/<run_name>/weights/`

### Inference

```bash
python scripts/infer.py --model models/best.pt --image sample.jpg
```

### GUI Application

```bash
python gui.py --model models/best.pt
```

## Model Card

| Property | Value |
|---|---|
| Architecture | YOLOv8 Nano |
| Input Size | 640×640 px |
| Classes | 1 (tumor) |
| Training Dataset | Brain Tumor MRI Images |
| Framework | PyTorch + Ultralytics |
| Weight Format | `.pt` (PyTorch) |

## Training Metrics

See `runs/detect/brain_tumor_detector/results.csv` for detailed metrics:
- `box_loss` - Bounding box regression loss
- `cls_loss` - Classification loss
- `dfl_loss` - Distribution focal loss
- `metrics/precision` - Precision on validation set
- `metrics/recall` - Recall on validation set
- `metrics/mAP50` - Mean Average Precision @ IoU=0.5
- `metrics/mAP50-95` - Mean Average Precision @ IoU=0.5:0.95

## Download Pre-trained Models

From Ultralytics Hub:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Model Versioning

To track model changes, consider:
1. Date stamps: `best_2026-02-27.pt`
2. Git tags: `model-v1.0`, `model-v1.1`
3. DVC (Data Version Control): `dvc add models/`

## Performance

Expected performance on test set (depends on dataset):
- **Precision:** 85-95%
- **Recall:** 85-95%
- **mAP50:** 90-97%

## Next Steps

- [ ] Export model to ONNX for deployment
- [ ] Quantize model for mobile inference
- [ ] Create model comparison benchmark
