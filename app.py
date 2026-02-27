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

sys.path.insert(0, str(Path(__file__).parent))

from src.bt_yolo.predict import YOLOPredictor


st.set_page_config(page_title="Brain Tumor YOLO Detector", layout="wide")

st.title("🧠 Brain Tumor Detection (YOLOv8)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_path = st.text_input("Model path", value="runs/detect/train/weights/best.pt")
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
        # Save temp file for prediction
        temp_path = "/tmp/temp_mri.jpg"
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
                st.subheader("📍 Detected Bounding Boxes")
                for i, detection in enumerate(result['detections'], 1):
                    col_bbox, col_conf = st.columns(2)
                    with col_bbox:
                        st.write(f"**Detection {i}**")
                        st.text(f"BBox: {[round(x.item(), 2) for x in detection['bbox']]}")
                    with col_conf:
                        st.metric(f"Confidence", f"{detection['confidence']:.4f}")
            
            # Raw results
            st.subheader("📋 Raw Detection Data")
            st.json({
                "image": result['image_path'],
                "num_detections": result['num_detections'],
                "detections_count": len(result['detections'])
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
- **Class:** Brain Tumor (1-class detection)
- **Input:** 640×640 MRI images
- **Framework:** Ultralytics YOLOv8
- **Model Size:** 6.3 MB (CPU/GPU compatible)
""")
