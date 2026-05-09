import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gradio as gr
import pywt

# --- WSL GPU FIX ---
cuda_path = "/usr/lib/wsl/lib"
if os.path.exists(cuda_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add project root to path to import logic utilities
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from logic import extract_7ch

# --- CONFIG ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
HYBRID_DIR = os.path.join(MODELS_DIR, "hybrid_7ch")

LEGACY_MODELS = {
    'Xception': 'Xcep_7ch_Calibrated.keras',
    'MobileNetV2': 'MobileNet_7ch_Calibrated.keras',
    'ResNet50': 'Res_7ch_Mega.keras',
    'MNet_Generalist': 'MobileNet_7ch_Mega.keras'
}

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🚀 Loading Legacy Ensemble (4 Models)...")
loaded_models = {}
for name, filename in LEGACY_MODELS.items():
    path = os.path.join(HYBRID_DIR, filename)
    if os.path.exists(path):
        print(f"  Loading {name}...")
        loaded_models[name] = keras.models.load_model(path, compile=False)
    else:
        print(f"  [SKIP] {name} not found.")

print("✅ Legacy Ensemble Ready.")

def predict_legacy(input_img):
    if input_img is None: return None, "No Image"
    
    # 1. Face Detection & Alignment
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad_w, pad_h = int(w * 0.3), int(h * 0.3)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(input_img.shape[1], x + w + pad_w), min(input_img.shape[0], y + h + pad_h)
        side = max(x2 - x1, y2 - y1)
        if x1 + side > input_img.shape[1]: x1 = input_img.shape[1] - side
        if y1 + side > input_img.shape[0]: y1 = input_img.shape[0] - side
        x1, y1 = max(0, x1), max(0, y1)
        face_crop = input_img[y1:y1+side, x1:x1+side]
    else:
        face_crop = input_img

    face_view = cv2.resize(face_crop, (256, 256))
    
    # 2. 7-Channel Extraction
    x_input = extract_7ch(face_view, 256)
    x_batch = np.expand_dims(x_input, 0)
    
    # 3. Multi-Model Inference
    results = []
    output_text = "--- Individual Scores ---\n"
    
    for name, model in loaded_models.items():
        # Handle different input names if necessary
        try:
            pred = model.predict(x_batch, verbose=0)[0][0]
        except:
            pred = model({'hybrid_7ch_input': x_batch}, training=False)[0][0]
            
        prob = float(pred)
        # Logit correction if outside [0,1]
        if prob < 0 or prob > 1:
            prob = 1.0 / (1.0 + np.exp(-prob))
            
        results.append(prob)
        label = "REAL" if prob >= 0.5 else "FAKE"
        output_text += f"{name}: {label} ({prob:.4f})\n"
        
    # 4. Simple Average Ensemble
    avg_prob = np.mean(results)
    final_label = "REAL" if avg_prob >= 0.5 else "FAKE"
    final_conf = abs(avg_prob - 0.5) * 2 * 100
    
    summary = f"\n--- ENSEMBLE VERDICT ---\nVerdict: {final_label}\nAvg Score: {avg_prob:.4f}\nConfidence: {final_conf:.2f}%"
    
    return face_view, output_text + summary

# UI
with gr.Blocks(theme="glass") as demo:
    gr.Markdown("# 🏛️ Legacy Ensemble Test (Models 2-5)")
    gr.Markdown("Testing the original 4-model system (Xception, MobileNet, ResNet).")
    
    with gr.Row():
        with gr.Column():
            in_img = gr.Image()
            btn = gr.Button("Run Legacy Analysis", variant="secondary")
        with gr.Column():
            out_crop = gr.Image(label="Processed Face")
            out_text = gr.Text(label="Ensemble Results")
            
    btn.click(predict_legacy, inputs=in_img, outputs=[out_crop, out_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
