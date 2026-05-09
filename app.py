import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # BUG FIX: prevent matplotlib GUI thread conflicts with Gradio
import matplotlib.pyplot as plt
from PIL import Image
from logic import DeepfakeDetector, crop_face

# Initialise detector (loads all 4 models once at startup)
detector = DeepfakeDetector()

# ------------------------------------------------------------------
#  IMAGE HANDLER
# ------------------------------------------------------------------
def analyse_image(pil_img):
    if pil_img is None:
        return "No image uploaded.", None, None

    img_rgb = np.array(pil_img.convert('RGB'))
    img_rgb = crop_face(img_rgb)

    # 1. Ensemble Prediction (returns real probs for all 4 models + Ensemble)
    probs = detector.predict_ensemble(img_rgb)
    ens   = probs['Ensemble']

    label = '🟢 REAL' if ens > 0.5 else '🔴 FAKE'
    conf  = ens if ens > 0.5 else 1 - ens

    # Model display names and their roles
    MODEL_META = {
        'Xception':        ('Diffusion Expert',           '94.61%'),
        'MobileNetV2':     ('Diffusion Specialist',        '90.08%'),
        'ResNet50':        ('Mega Generalist',             '83.75%'),
        'MNet_Generalist': ('GAN/FaceSwap Expert',         '83.63%'),
        'ConvNeXt_Diff':   ('Diffusion Ultra-Specialist',  '92.53%'),
        'Ensemble':        ('Final Verdict',               ''),
    }

    summary = f"### {label} | Ensemble Confidence: {conf*100:.1f}%\n\n"
    summary += "**Individual Model Verdicts:**\n"
    for name, p in probs.items():
        v_label = '🟢 REAL' if p > 0.5 else '🔴 FAKE'
        v_conf  = p if p > 0.5 else 1 - p
        role, acc = MODEL_META.get(name, ('', ''))
        acc_str = f" | acc={acc}" if acc else ""
        summary += f"- **{name}** `[{role}{acc_str}]` : {v_label}  ({v_conf*100:.1f}%)\n"

    # 2. Grad-CAM
    if ens > 0.5:
        cam_img = img_rgb
    else:
        cam_img = detector.generate_gradcam('Xception', img_rgb)
    plt.close('all')

    # 3. Confidence Bar Chart (exclude 'Ensemble' from bar chart, show it as dashed line)
    model_names = [n for n in probs if n != 'Ensemble']
    model_vals  = [probs[n] * 100 for n in model_names]
    colors = ['#2ecc71' if v > 50 else '#e74c3c' for v in model_vals]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(model_names, model_vals, color=colors, edgecolor='white', height=0.55)
    ax.axvline(50, color='yellow', linestyle='--', linewidth=1.5, label='Decision Threshold')
    ax.axvline(ens * 100, color='cyan', linestyle='-', linewidth=2, alpha=0.7, label=f'Ensemble ({ens*100:.1f}%)')

    # Value labels on bars
    for bar, val in zip(bars, model_vals):
        ax.text(min(val + 1, 97), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color='white', fontsize=8)

    ax.set_title('Neural Ensemble Evidence', color='white', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability of being REAL (%)', color='#cccccc')
    ax.set_facecolor('#1e1e2d')
    fig.patch.set_facecolor('#1e1e2d')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('#cccccc')
    ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=8)
    plt.tight_layout()

    return summary, Image.fromarray(cam_img), fig


# ------------------------------------------------------------------
#  VIDEO HANDLER
# ------------------------------------------------------------------
def analyse_video(video_path, n_frames=30, gamma=0.7):
    if video_path is None:
        return "No video uploaded.", None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video.", None, None

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        return "No readable frames found.", None, None

    n = min(int(n_frames), len(all_frames))
    sample_indices = np.linspace(0, len(all_frames) - 1, n).astype(int)

    raw_scores, smoothed_scores = [], []
    xcep_scores, mnet_scores, res_scores, mnet_mega_scores = [], [], [], []
    S = 0.5
    worst_frame_rgb, worst_score = None, 1.0

    for idx in sample_indices:
        frame = all_frames[idx]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = crop_face(img_rgb)

        all_probs = detector.predict_ensemble(img_rgb)
        ens_score = all_probs['Ensemble']

        raw_scores.append(ens_score)
        xcep_scores.append(all_probs.get('Xception', 0.5))
        mnet_scores.append(all_probs.get('MobileNetV2', 0.5))
        res_scores.append(all_probs.get('ResNet50', 0.5))
        mnet_mega_scores.append(all_probs.get('MNet_Generalist', 0.5))

        S = float(gamma) * ens_score + (1 - float(gamma)) * S
        smoothed_scores.append(S)

        if S < worst_score:
            worst_score = S
            worst_frame_rgb = img_rgb.copy()

    if not smoothed_scores:
        return "No frames could be read from the video.", None, None

    def verdict(p):
        lbl = '🟢 REAL' if p > 0.5 else '🔴 FAKE'
        c   = p if p > 0.5 else 1 - p
        return f"{lbl} ({c*100:.1f}%)"

    final_score = float(np.mean(smoothed_scores))
    label = '🟢 REAL' if final_score > 0.5 else '🔴 FAKE'
    conf  = final_score if final_score > 0.5 else 1 - final_score

    summary  = f"**{label} | Ensemble Confidence: {conf*100:.1f}%**\n"
    summary += f"Frames Analysed: {len(smoothed_scores)} / {int(n_frames)}\n\n"
    summary += f"**Individual Model Verdicts (avg over frames):**\n"
    summary += f"- Xception        `[Diffusion Expert,   94.61%]` : {verdict(np.mean(xcep_scores))}\n"
    summary += f"- MobileNetV2     `[Diffusion Spec.,    90.08%]` : {verdict(np.mean(mnet_scores))}\n"
    summary += f"- ResNet50        `[Mega Generalist,    83.75%]` : {verdict(np.mean(res_scores))}\n"
    summary += f"- MNet_Generalist `[GAN/FaceSwap Expert,83.63%]` : {verdict(np.mean(mnet_mega_scores))}\n"
    summary += f"- **Ensemble**                                   : {verdict(final_score)}\n"

    # Temporal Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    x_axis = list(range(len(raw_scores)))
    ax.plot(x_axis, raw_scores, 'o--', color='#3498db', alpha=0.5, label='Per-Frame')
    ax.plot(x_axis, smoothed_scores, '-', color='#2ecc71', linewidth=2, label='EMA Smoothed')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Decision Boundary')
    ax.fill_between(x_axis, 0, 0.5, alpha=0.05, color='red')
    ax.fill_between(x_axis, 0.5, 1, alpha=0.05, color='green')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Sampled Frame Index')
    ax.set_ylabel('Real Probability')
    ax.set_title('Temporal Detection Timeline')
    ax.legend()
    plt.tight_layout()

    if worst_frame_rgb is not None:
        cam_img = worst_frame_rgb if final_score > 0.5 else detector.generate_gradcam('Xception', worst_frame_rgb)
        worst_cam_pil = Image.fromarray(cam_img)
    else:
        worst_cam_pil = None

    return summary, worst_cam_pil, fig


# ------------------------------------------------------------------
#  GRADIO UI
# ------------------------------------------------------------------
with gr.Blocks(title="Hybrid Deepfake Detector") as demo:
    gr.Markdown("# 🛡️ 7-Channel Wavelet Ensemble | Universal Deepfake Detection")
    gr.Markdown(
        "**5-Model Ensemble: Xception (94.6%) + MobileNetV2 (90.1%) + ResNet50 (83.8%) + MNet_Generalist (83.6%) + ConvNeXt-Diff (92.5%) | Grad-CAM Explainability**"
    )

    with gr.Tabs():
        # --- IMAGE TAB ---
        with gr.Tab("🖼️ Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type='pil', label='Upload Face Image')
                    btn_img   = gr.Button('🔍 Analyse Image', variant='primary')
                with gr.Column(scale=2):
                    res_img  = gr.Textbox(label='Analysis Summary', lines=10)
                    with gr.Row():
                        img_cam  = gr.Image(label='Grad-CAM Heatmap')
                        img_plot = gr.Plot(label='Confidence Breakdown')
            btn_img.click(fn=analyse_image,
                          inputs=[img_input],
                          outputs=[res_img, img_cam, img_plot])

        # --- VIDEO TAB ---
        with gr.Tab("🎬 Video Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label='Upload Video')
                    n_fr      = gr.Slider(5, 30, value=15, step=1,
                                          label='Frames to Sample')
                    gam_sl    = gr.Slider(0.3, 0.95, value=0.7, step=0.05,
                                          label='EMA Smoothing (gamma)')
                    btn_vid   = gr.Button('🔍 Analyse Video', variant='primary')
                with gr.Column(scale=2):
                    res_vid   = gr.Textbox(label='Video Report', lines=8)
                    vid_worst = gr.Image(label='Grad-CAM: Most Suspicious Frame')
                    vid_plot  = gr.Plot(label='Temporal Timeline')
            btn_vid.click(fn=analyse_video,
                          inputs=[vid_input, n_fr, gam_sl],
                          outputs=[res_vid, vid_worst, vid_plot])

if __name__ == "__main__":
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    demo.launch(share=False, show_error=True, strict_cors=False)
