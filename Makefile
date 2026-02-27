.PHONY: help install train eval predict clean

help:
	@echo "Brain Tumor YOLOv8 Detection - Makefile Commands"
	@echo "================================================"
	@echo "  make install       Install dependencies"
	@echo "  make train         Train YOLOv8 (5 epochs smoke test)"
	@echo "  make train-full    Train YOLOv8 (100 epochs full)"
	@echo "  make eval          Evaluate trained model"
	@echo "  make predict       Predict on single image (IMG=path/to/image.jpg)"
	@echo "  make clean         Remove artifacts and cache"
	@echo ""

install:
	pip install -r requirements.txt

train:
	python -m src.bt_yolo.train --data data.yaml --epochs 5 --name smoke_test --device cpu

train-full:
	python -m src.bt_yolo.train --data data.yaml --epochs 100 --name full_train --device 0

eval:
	@echo "Placeholder for evaluation (Phase 2)"

predict:
	@if [ -z "$(IMG)" ]; then \
		echo "Usage: make predict IMG=path/to/image.jpg"; \
	else \
		python -c "from src.bt_yolo.predict import YOLOPredictor; p = YOLOPredictor('runs/detect/train/weights/best.pt'); print(p.predict_image('$(IMG)'))"; \
	fi

clean:
	rm -rf runs/ artifacts/ reports/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.DEFAULT_GOAL := help
