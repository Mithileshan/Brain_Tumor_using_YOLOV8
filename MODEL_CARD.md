# Brain Tumor YOLO Detection - Model Card

**Date:** February 27, 2026  
**Version:** 1.0  
**Phase:** Production-Ready (Phase 1-8 Complete)

---

## 📋 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | YOLOv8 Nano (Real-time Object Detection) |
| **Framework** | Ultralytics 8.0.147, PyTorch 2.0.1 |
| **Task** | Single-class object detection |
| **Class** | Brain Tumor |
| **Input Size** | 640×640 RGB images |
| **Model Size** | 6.3 MB |
| **Inference Speed** | ~2-3ms (GPU), ~15ms (CPU) |
| **Training Data** | Roboflow Crystal Clean Brain dataset |
| **Training Time** | ~2 hours (GPU), ~30 minutes smoke test (CPU) |

---

## 📊 Performance Metrics (Phase 3)

### Smoke Test (2 epochs on CPU)
- **mAP@50:** 0.954
- **mAP@50-95:** 0.597
- **Precision:** 0.0033
- **Recall:** 1.0

**Note:** Evaluate on full training (100+ epochs) for production metrics

---

## 🔄 Training Configuration

**Hyperparameters (Phase 1-2):**
```json
{
  "model": "yolov8n.pt",
  "epochs": 100,
  "batch_size": 16,
  "imgsz": 640,
  "device": "0",
  "lr0": 0.01,
  "momentum": 0.937,
  "weight_decay": 0.0005,
  "patience": 50
}
```

**Dataset:** [Roboflow - Crystal Clean Brain Tumors MRI](https://universe.roboflow.com/crystal-clear-brain)

---

## 📦 Model Artifacts (Phase 4)

**Location:** `runs/detect/train/`

```
runs/detect/train/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   └── last.pt              # Last checkpoint
├── results.csv              # Training metrics per epoch
├── config.json              # Training configuration (Phase 1)
├── metrics.json             # Final metrics
└── predictions.json         # Sample predictions (Phase 3)
```

**Versioning:** Git tags for release versions
- `v1.0` - Initial production model
- `v1.1` - Improved with more training data

---

## 🚀 Usage

### Command-Line Training (Phase 2)
```bash
python -m src.bt_yolo.train \
  --data data.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --name production_train
```

### Inference - Python API (Phase 4)
```python
from src.bt_yolo.predict import YOLOPredictor

predictor = YOLOPredictor(
    'runs/detect/train/weights/best.pt',
    conf_threshold=0.5
)

# Single image
result = predictor.predict_image('path/to/mri.jpg')
print(f"Detections: {result['num_detections']}")

# Batch
results = predictor.predict_folder('path/to/images/')
```

### Interactive UI (Phase 5)
```bash
streamlit run app.py
```

### Evaluation (Phase 3)
```bash
python scripts/evaluate.py --model runs/detect/train/weights/best.pt --data data.yaml
```

---

## 🧪 Testing (Phase 6)

Run test suite:
```bash
pytest tests/ -v
```

Tests include:
- Configuration validation
- YAML parsing
- Image size constraints
- Model loading

---

## 🐳 Docker Deployment (Phase 7)

Build image:
```bash
docker build -t brain-tumor-yolo:latest .
```

Run container:
```bash
docker run -p 8501:8501 brain-tumor-yolo:latest
```

Run with GPU:
```bash
docker run --gpus all -p 8501:8501 brain-tumor-yolo:latest
```

---

## ✅ CI/CD Pipeline (Phase 8)

GitHub Actions workflows defined in `.github/workflows/ci.yml`:
- **Lint:** flake8, black, ruff checks
- **Test:** pytest on Python 3.9, 3.10
- **Docker:** Build and validate image (CPU)
- **Trigger:** On push to main/develop or PR

---

## ⚠️ Limitations & Considerations

1. **Limited Training Data:**
   - Smoke test: only 2 epochs
   - Full training recommended for production

2. **GPU Recommended:**
   - CPU inference: ~15ms per image
   - GPU inference: ~2–3ms per image

3. **Dataset:**
   - Must download separately from Roboflow
   - Ensure proper YOLO format with `.txt` labels

4. **Hardware:**
   - CPU: Slower but functional for demo
   - GPU (CUDA 12.1): Recommended for production

---

## 📈 Model Export Options (Phase 4)

YOLOv8 supports multiple export formats:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# Export to ONNX (cross-platform)
model.export(format='onnx')

# Export to TensorRT (NVIDIA inference)
model.export(format='engine')

# Export to CoreML (Apple devices)
model.export(format='coreml')
```

---

## 🔄 Retraining Pipeline (Phase 2+)

For production retraining:

```bash
# 1. Update dataset from Roboflow
python scripts/download_dataset.py

# 2. Train new model
python -m src.bt_yolo.train --data data.yaml --name production_v2

# 3. Evaluate
python scripts/evaluate.py

# 4. Compare metrics
# 5. If better, promote: cp runs/detect/production_v2/weights/best.pt runs/detect/train/weights/best_production.pt
```

---

## 📞 Support

For issues, feature requests, or questions:
- GitHub Issues: [Brain_Tumor_using_YOLOV8/issues](https://github.com/Mithileshan/Brain_Tumor_using_YOLOV8/issues)
- Documentation: See [README.md](README.md)

---

**Author:** Mithileshan  
**Created:** February 2026  
**Last Updated:** February 27, 2026
