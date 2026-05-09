import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gradio as gr
import pywt
from tqdm import tqdm

# --- WSL GPU FIX ---
cuda_path = "/usr/lib/wsl/lib"
if os.path.exists(cuda_path):
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from logic import extract_7ch

# --- CONFIG ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
HYBRID_DIR = os.path.join(MODELS_DIR, "hybrid_7ch")

MODELS_TO_LOAD = {
    'ConvNeXt_V2':    ('Swin_Diff_7Ch', 'convnext_diff_7ch_v2.keras', 15.0, "diffusion"),
    'Xception':       ('hybrid_7ch',    'Xcep_7ch_Calibrated.keras', 10.0, "legacy"),
    'EfficientNet':   ('hybrid_7ch',    'MobileNet_7ch_Calibrated.keras', 5.0, "diffusion"),
    'Generalist_V1':  ('hybrid_7ch',    'MobileNet_7ch_Mega.keras', 5.0, "legacy"),
    'ResNet50':       ('hybrid_7ch',    'Res_7ch_Mega.keras', 1.0, "legacy")
}

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🚀 Initializing Master Ensemble...")
models = {}
for name, (subdir, filename, weight, mtype) in MODELS_TO_LOAD.items():
    path = os.path.join(MODELS_DIR, subdir, filename)
    if os.path.exists(path):
        print(f"  Loading {name}...")
        try:
            m = keras.models.load_model(path, compile=False)
            models[name] = {'model': m, 'base_weight': weight, 'type': mtype}
        except Exception as e: print(f"  [ERR] {name}: {e}")

def process_core(img_rgb, is_webcam=False):
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
        face_crop = img_rgb[max(0,y1):y1+side, max(0,x1):x1+side]
    else: face_crop = img_rgb

    face_view = cv2.resize(face_crop, (256, 256))
    x_input = extract_7ch(face_view, 256)
    x_batch = np.expand_dims(x_input, 0)
    
    total_weighted_prob = 0
    total_weight = 0
    scores = {}
    
    for name, data in models.items():
        m, w, mtype = data['model'], data['base_weight'], data['type']
        try: pred = m.predict(x_batch, verbose=0)[0][0]
        except: pred = m({'hybrid_7ch_input': x_batch}, training=False)[0][0]
        
        prob = float(pred)
        if prob < 0 or prob > 1: prob = 1.0 / (1.0 + np.exp(-prob))
            
        # --- BALANCED WEIGHTING LOGIC ---
        eff_weight = w
        if is_webcam:
            # If webcam, suppress diffusion specialists and trust legacy models
            if mtype == "diffusion": eff_weight = 0.5 
            else: eff_weight = w * 2.0
        # No more 'Dictator' override here. Purely weighted consensus.
            
        total_weighted_prob += (prob * eff_weight)
        total_weight += eff_weight
        scores[name] = prob
        
    return total_weighted_prob / total_weight, scores, face_view

def predict_image(input_img, source_type):
    if input_img is None: return None, "No Image"
    is_webcam = (source_type == "Live Webcam")
    prob, scores, face = process_core(input_img, is_webcam)
    
    rep = f"🔍 SOURCE: {source_type}\n" + "-"*30 + "\n"
    for name, s in scores.items():
        rep += f"{name:<15}: {'REAL' if s>=0.5 else 'FAKE'} ({s:.4f})\n"
    
    verdict = "REAL HUMAN" if prob >= 0.5 else "AI GENERATED (FAKE)"
    rep += f"\n🏆 ENSEMBLE VERDICT: {verdict}\nConfidence: {abs(prob-0.5)*200:.2f}%"
    return face, rep

def predict_video(video_path, source_type):
    if video_path is None: return None, "No Video"
    is_webcam = (source_type == "Live Webcam")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, 10).astype(int) if total > 1 else range(10)
    
    v_probs = []
    m_avgs = {name: [] for name in models.keys()}
    last_face = None
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret: continue
        p, fs, face = process_core(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), is_webcam)
        v_probs.append(p); last_face = face
        for n, s in fs.items(): m_avgs[n].append(s)
    cap.release()
    
    avg = np.mean(v_probs)
    ver = "REAL VIDEO" if avg >= 0.5 else "DEEPFAKE DETECTED"
    summary = f"🎞️ VIDEO ANALYSIS ({source_type})\n" + "="*30
    summary += f"\nVerdict: {ver}\nGlobal Score: {avg:.4f}\nConfidence: {abs(avg-0.5)*200:.2f}%\n"
    summary += "\n📈 PER-MODEL AVG:\n"
    for n in models.keys():
        ma = np.mean(m_avgs[n])
        summary += f"  {n:<15}: {'REAL' if ma>=0.5 else 'FAKE'} ({ma:.4f})\n"
    summary += "\nTimeline: " + "".join(["R" if p>=0.5 else "F" for p in v_probs])
    return last_face, summary

# --- UI ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🛡️ PBL Master: Hybrid Ensemble (Adaptive Weighting)")
    source_sel = gr.Radio(["High-Res Upload", "Live Webcam"], label="Detection Context", value="High-Res Upload")
    
    with gr.Tab("Image Scan"):
        with gr.Row():
            in_img = gr.Image(); btn_img = gr.Button("Analyze Photo", variant="primary")
            out_img_crop = gr.Image(); out_img_text = gr.Text()
        btn_img.click(predict_image, inputs=[in_img, source_sel], outputs=[out_img_crop, out_img_text])
            
    with gr.Tab("Video Scan"):
        with gr.Row():
            in_vid = gr.Video(); btn_vid = gr.Button("Analyze Video", variant="primary")
            out_vid_crop = gr.Image(); out_vid_text = gr.Text()
        btn_vid.click(predict_video, inputs=[in_vid, source_sel], outputs=[out_vid_crop, out_vid_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
