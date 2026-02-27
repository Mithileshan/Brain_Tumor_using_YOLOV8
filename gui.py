#!/usr/bin/env python3
"""
YOLOv8 Brain Tumor Detection - GUI Application

Provides interactive interface for loading images and running inference.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from ttkthemes import ThemedStyle  # Import ThemedStyle from ttkthemes

# Default model paths to try (in order)
DEFAULT_MODELS = [
    'models/best.pt',
    'runs/detect/train/weights/best.pt',
    'yolov8n.pt',
]


class ObjectDetectionApp:
    def __init__(self, root, model_path=None):
        """Initialize the application."""
        self.root = root
        self.root.title("🧠 Brain Tumor Detection - YOLOv8")
        self.root.geometry("900x900")
        self.root.minsize(600, 600)

        # Setup theme
        try:
            style = ThemedStyle(root)
            style.set_theme("equilux")
        except Exception as e:
            print(f"Warning: Could not set theme: {e}")

        # Model state
        self.model = None
        self.model_path = model_path
        self.loaded_image = None
        self.loaded_cv_image = None

        # Build UI
        self._build_ui()
        
        # Load model
        self._load_model()

    def _build_ui(self):
        """Build the user interface."""
        # Title
        title_label = tk.Label(
            self.root,
            text="🧠 Brain Tumor Detection",
            font=("Helvetica", 22, "bold"),
        )
        title_label.pack(pady=15)

        # Model info frame
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(model_frame, text="Model:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        self.model_label = tk.Label(
            model_frame,
            text="Loading...",
            font=("Helvetica", 10, "italic"),
            fg="blue"
        )
        self.model_label.pack(side=tk.LEFT, padx=5)

        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10, padx=10, fill=tk.X)

        self.load_model_button = tk.Button(
            button_frame,
            text="Load Model...",
            command=self._select_model,
            width=15
        )
        self.load_model_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(
            button_frame,
            text="Load Image",
            command=self.load_image,
            width=15,
            bg="#4CAF50",
            fg="white"
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.detect_button = tk.Button(
            button_frame,
            text="Detect Tumor",
            command=self.detect_objects,
            width=15,
            bg="#FF9800",
            fg="white"
        )
        self.detect_button.pack(side=tk.LEFT, padx=5)

        # Confidence threshold slider
        threshold_frame = tk.Frame(self.root)
        threshold_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(threshold_frame, text="Confidence Threshold:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        self.conf_slider = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=200
        )
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT, padx=5)

        self.conf_label = tk.Label(threshold_frame, text="0.50", font=("Helvetica", 10))
        self.conf_label.pack(side=tk.LEFT, padx=5)
        self.conf_slider.config(command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"))

        # Image display
        self.image_label = tk.Label(
            self.root,
            bg="LightGray",
            padx=20,
            pady=20,
            text="Load an image to start",
            font=("Helvetica", 14)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Helvetica", 9),
            fg="green",
            justify=tk.LEFT
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def _load_model(self):
        """Load the YOLOv8 model."""
        # Determine model path
        if self.model_path and Path(self.model_path).exists():
            model_path = self.model_path
        else:
            # Try default locations
            model_path = None
            for default_path in DEFAULT_MODELS:
                if Path(default_path).exists():
                    model_path = default_path
                    break
        
        if model_path is None:
            self.model_label.config(text="No model found (will use default: yolov8n)", fg="orange")
            model_path = "yolov8n.pt"
        
        try:
            self._update_status(f"Loading model: {model_path}...")
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.model_label.config(text=str(model_path), fg="green")
            self._update_status(f"✓ Model loaded: {model_path}")
        except Exception as e:
            self.model_label.config(text="Error loading model", fg="red")
            self._update_status(f"❌ Error: {str(e)}", error=True)
            messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")

    def _select_model(self):
        """Open dialog to select a model file."""
        model_file = filedialog.askopenfilename(
            title="Select YOLOv8 Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")],
            defaultdir="models/"
        )
        if model_file:
            self.model_path = model_file
            self._load_model()

    def load_image(self):
        """Load an image from file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self._update_status(f"Loading image: {Path(file_path).name}")
                self.loaded_image = Image.open(file_path)
                self.loaded_image = self.loaded_image.resize((600, 600))
                self.loaded_cv_image = cv2.cvtColor(np.array(self.loaded_image), cv2.COLOR_RGB2BGR)
                self.display_cv_image(self.loaded_cv_image)
                self._update_status(f"✓ Image loaded: {Path(file_path).name}")
            except Exception as e:
                self._update_status(f"❌ Error loading image: {str(e)}", error=True)
                messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")

    def detect_objects(self):
        """Run inference to detect brain tumors."""
        if self.model is None:
            messagebox.showwarning("Model Not Loaded", "Please load a model first")
            return

        if self.loaded_cv_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return

        try:
            self._update_status("Running inference...")
            conf = float(self.conf_slider.get())
            
            results = self.model(
                source=self.loaded_cv_image,
                conf=conf,
                verbose=False
            )
            
            if results and len(results) > 0:
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                self.display_cv_image(res_plotted_rgb)
                
                # Get detection count
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                self._update_status(f"✓ Inference complete. Detections: {num_detections} (conf: {conf:.2f})")
            else:
                self._update_status("✓ Inference complete. No tumors detected.")
        except Exception as e:
            self._update_status(f"❌ Error during inference: {str(e)}", error=True)
            messagebox.showerror("Inference Error", f"Failed to run inference:\n{str(e)}")

    def display_cv_image(self, cv_image):
        """Display a cv2 image in the UI."""
        try:
            pil_image = Image.fromarray(cv_image)
            pil_image = pil_image.resize((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            self._update_status(f"❌ Error displaying image: {str(e)}", error=True)

    def _update_status(self, message, error=False):
        """Update status bar message."""
        color = "red" if error else "green"
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()  # Force UI refresh

    def display_message(self, message):
        """Display a message in the image area."""
        self.image_label.config(text=message)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 Brain Tumor Detection GUI")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model weights (e.g., models/best.pt)'
    )
    args = parser.parse_args()
    
    root = tk.Tk()
    app = ObjectDetectionApp(root, model_path=args.model)
    root.mainloop()