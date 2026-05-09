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

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Swin_Diff_7Ch", "convnext_diff_7ch_v2.keras")
IMG_SIZE = 256

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_7ch_debug(img_rgb):
    # Detection
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad_w, pad_h = int(w * 0.3), int(h * 0.3)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(img_rgb.shape[1], x + w + pad_w), min(img_rgb.shape[0], y + h + pad_h)
        side = max(x2 - x1, y2 - y1)
        if x1 + side > img_rgb.shape[1]: x1 = img_rgb.shape[1] - side
        if y1 + side > img_rgb.shape[0]: y1 = img_rgb.shape[0] - side
        x1, y1 = max(0, x1), max(0, y1)
        face_crop = img_rgb[y1:y1+side, x1:x1+side]
    else:
        face_crop = img_rgb

    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    
    # 🧪 TEST: SLIGHT BLUR TO REMOVE WEBCAM NOISE
    # If the model starts saying REAL after this, it was over-sensitive to noise.
    # face_resized = cv2.GaussianBlur(face_resized, (3,3), 0)

    img_norm = face_resized.astype('float32') / 255.0
    
    # Wavelets on the RED channel (Index 0 in RGB)
    # We must ensure this matches the training exactly.
    coeffs2 = pywt.dwt2(img_norm[:,:,0], 'haar')
    LL, (LH, HL, HH) = coeffs2
    LH = cv2.resize(LH, (IMG_SIZE, IMG_SIZE))
    HL = cv2.resize(HL, (IMG_SIZE, IMG_SIZE))
    HH = cv2.resize(HH, (IMG_SIZE, IMG_SIZE))
    fused = np.sqrt(LH**2 + HL**2 + HH**2)
    
    input_7ch = np.dstack([img_norm, LH, HL, HH, fused])
    return input_7ch, face_resized

print("🚀 Loading Specialist V2...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Loaded.")

def predict(input_img):
    if input_img is None: return None, "No Image"
    
    # Preprocess
    x_input, face_view = extract_7ch_debug(input_img)
    x_batch = np.expand_dims(x_input, 0)
    
    # Inference
    raw_pred = model.predict(x_batch, verbose=0)[0][0]
    raw_prob = float(raw_pred)
    
    # PRINT TO CONSOLE FOR DEBUGGING
    print(f"DEBUG: Raw Probability = {raw_prob:.6f} (0=Fake, 1=Real)")
    
    # Verdict Logic
    # We use a slightly wider "Uncertainty" zone for V2
    if raw_prob > 0.6:
        label = "REAL HUMAN"
        color = "Green"
    elif raw_prob < 0.4:
        label = "AI GENERATED (FAKE)"
        color = "Red"
    else:
        label = "UNCERTAIN (Mixed Signals)"
        color = "Yellow"
    
    confidence = abs(raw_prob - 0.5) * 2 * 100
    res = f"Verdict: {label}\nRaw Score: {raw_prob:.4f}\nConfidence: {confidence:.2f}%"
    
    return face_view, res

# UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🛡️ Phase 4: ConvNeXt-Diff V2 Specialist")
    gr.Markdown("This model focuses on **Diffusion Noise**. It is extremely sensitive.")
    
    with gr.Row():
        with gr.Column():
            in_img = gr.Image()
            btn = gr.Button("Analyze Face", variant="primary")
        with gr.Column():
            out_crop = gr.Image(label="What the model sees")
            out_text = gr.Text(label="Analysis Result")
            
    btn.click(predict, inputs=in_img, outputs=[out_crop, out_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
