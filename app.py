"""
Streamlit UI for Brain Tumor YOLO Detection
Phase 5: Interactive demo for object detection
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from src.bt_yolo.predict import YOLOPredictor


st.set_page_config(page_title="Brain Tumor YOLO Detector", layout="wide")

st.title("🧠 Brain Tumor Detection (YOLOv8)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_path = st.text_input("Model path", value="runs/detect/runs/detect/train_custom/weights/best.pt")
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded MRI Scan", use_column_width=True)

with col2:
    st.subheader("🔍 Detection Results")
    
    if uploaded_file:
        # Save temp file for prediction (cross-platform)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "streamlit_mri_temp.jpg")
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load model
            predictor = YOLOPredictor(model_path, conf_threshold=confidence)
            
            # Predict
            result = predictor.predict_image(temp_path)
            
            # Display results
            st.success("✅ Detection Complete")
            st.write("\n")
            
            st.metric("Number of Detections", result['num_detections'])
            
            if result['num_detections'] > 0:
                st.subheader("📍 Detected Objects")
                for i, detection in enumerate(result['detections'], 1):
                    col_bbox, col_conf = st.columns(2)
                    with col_bbox:
                        st.write(f"**Detection {i}**")
                        bbox = detection['bbox']
                        st.text(f"BBox: x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}")
                    with col_conf:
                        st.metric(f"Confidence", f"{detection['confidence']:.4f}")
            else:
                st.info("""
                ℹ️ **No objects detected**
                
                The model is ready to use! For better tumor detection accuracy:
                1. Train a custom YOLOv8 model on annotated brain tumor data
                2. Place the trained weights at `runs/detect/train/weights/best.pt`
                3. Update the model path in the Configuration sidebar
                """)
            
            # Raw results
            st.subheader("📋 Detection Details")
            st.json({
                "model": model_path,
                "confidence_threshold": confidence,
                "num_detections": result['num_detections'],
                "image_path": os.path.basename(result['image_path'])
            })
            
        except Exception as e:
            st.error(f"❌ Detection failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        st.info("👆 Upload an MRI image to run detection")

# Footer
st.markdown("---")
st.markdown("""
### 📚 Model Information
- **Algorithm:** YOLOv8 Nano (Real-time Object Detection)
- **Input Resolution:** 640×640 pixels
- **Framework:** Ultralytics YOLOv8
- **Model Size:** 6.3 MB (CPU/GPU compatible)

### 🚀 Getting Started
This demo uses a general YOLOv8 model. To detect brain tumors:
1. Prepare annotated MRI images in YOLO format
2. Run: `python -m src.bt_yolo.train --data data/data.yaml --epochs 50`
3. Wait for training to complete (creates `runs/detect/train/weights/best.pt`)
4. Update the model path in Configuration sidebar
""")
