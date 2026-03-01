# Brain Tumor Detection using YOLOv8

---

## Overview

This project implements **YOLOv8 Nano** for brain tumor detection in MRI scans:

**Model:** YOLOv8 Nano (6.3 MB)  
**Class:** 1 (tumor)  
**Framework:** PyTorch + Ultralytics  
**Tests:** 5/5 passing | **Performance:** mAP50 0.879 | mAP50-95 0.548

---

## Quick Start 

### 1. Install

```bash
git clone https://github.com/Mithileshan/Brain_Tumor_using_YOLOV8.git
cd Brain_Tumor_using_YOLOV8
pip install -r requirements.txt
```

### 2. Prepare Dataset

Create YOLO-format dataset in data/:

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

YOLO label format:
```
<class_id> <x_center> <y_center> <width> <height>
# Normalized coordinates (0-1)
```

### 3. Train

```bash
# Smoke test (2 epochs on CPU)
make train

# Full training (100 epochs on GPU)
make train-full

# Or directly (with CPU for this example):
python -m src.bt_yolo.train --data data/data.yaml --epochs 10 --batch 4 --device cpu
```

**Pre-trained custom model available:** `runs/detect/runs/detect/train_custom/weights/best.pt`

### 4. Inference & Interactive UI

```bash
# Evaluation (Phase 2 - Complete)
python -m src.bt_yolo.eval --model runs/detect/train/weights/best.pt

# Interactive Streamlit UI (Phase 5 - Complete)
streamlit run app.py

# Single image prediction (Phase 4 - Complete)
python -c "from src.bt_yolo.predict import YOLOPredictor; p = YOLOPredictor('runs/detect/train/weights/best.pt'); print(p.predict_image('path/to/image.jpg'))"
```

#### UI & Detection Screenshot

![Tumor Detection Result](screenshot/detected.png)

### 5. Tests

```bash
pytest tests/ -v
# Expected: 5/5 PASSED
```

### 6. Docker

```bash
docker build -t brain-tumor-yolo:latest .
docker run -p 8501:8501 brain-tumor-yolo:latest streamlit run app.py
```

---


## Configuration

data.yaml:
```yaml
path: ../data
train: images/train
val: images/val
test: images/test
nc: 1
names: ['tumor']
```

Key hyperparameters in src/bt_yolo/config.py:
- model: yolov8n.pt
- epochs: 100
- batch_size: 16
- imgsz: 640
- device: 0 (GPU) or cpu
- lr0: 0.01
- patience: 50

---

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8 Nano (68.4M params) |
| Input Size | 640x640 px |
| Classes | 1 (tumor) |
| Inference Speed | ~2-3 ms GPU / ~15-20 ms CPU |
| Pre-trained Weights | COCO dataset |
| Framework | PyTorch 2.0+ |

---

**Training Results (Smoke Test 2 epochs):**
- mAP@50: 0.879
- mAP@50-95: 0.548
- Precision: 0.522
- Recall: 1.0

**Custom Model (Trained on Local Dataset - 10 epochs CPU):**
- Test Detection: **0.9179 confidence** on glioma_tumor sample
- Model: `runs/detect/runs/detect/train_custom/weights/best.pt`
- Status:  ACTIVE in Streamlit UI

---

## Reproducibility

```bash
pip install -r requirements.txt
python -m src.bt_yolo.train --data data/data.yaml --epochs 5 --device cpu
pytest tests/ -v
```

---
## Limitations & Safety

- NOT FDA-approved. Research-only, not for diagnosis.
- Single-class detection only (binary: tumor vs. no-tumor)
- Model performance depends on training dataset quality
- Always validate predictions with radiologist review

---

## Future Enhancements (Optional)

- Deploy to Hugging Face Spaces or Streamlit Cloud
- Multi-class detection for tumor subtypes
- Batch prediction CLI
- ONNX export for cross-platform deployment
- Real-time video inference

