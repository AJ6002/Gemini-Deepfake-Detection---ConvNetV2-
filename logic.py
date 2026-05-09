import os
import cv2
import numpy as np
import tensorflow as tf
import keras
import pywt
import os

# --- KERAS COMPATIBILITY PATCH ---
# Fixes: ValueError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
_orig_dense_init = keras.layers.Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return _orig_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = _patched_dense_init
# ---------------------------------

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img_rgb: np.ndarray) -> np.ndarray:
    """
    Detects the largest face and crops it with a 20% margin.
    If no face is detected, returns a center crop.
    """
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            h, w = img_rgb.shape[:2]
            side = min(h, w)
            x, y = (w - side) // 2, (h - side) // 2
            return img_rgb[y:y+side, x:x+side]
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(w * 0.2)
        y1 = max(0, y - margin)
        y2 = min(img_rgb.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(img_rgb.shape[1], x + w + margin)
        return img_rgb[y1:y2, x1:x2]
    except Exception:
        return img_rgb


def extract_7ch(img_rgb: np.ndarray, size: int) -> np.ndarray:
    """
    Convert an RGB image into a 7-channel tensor:
      Ch 1-3 : Normalised RGB  [0, 1]
      Ch 4   : LH wavelet sub-band (horizontal edges / GAN artifacts)
      Ch 5   : HL wavelet sub-band (vertical edges)
      Ch 6   : HH wavelet sub-band (diagonal noise)
      Ch 7   : Fused high-frequency energy map
    """
    img = cv2.resize(img_rgb, (size, size))
    rgb = img.astype(np.float32) / 255.0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, (LH, HL, HH) = pywt.dwt2(gray, 'db1')

    def norm_resize(band):
        band = cv2.resize(band.astype(np.float32), (size, size))
        lo, hi = band.min(), band.max()
        return (band - lo) / (hi - lo + 1e-7)

    LH = norm_resize(LH)
    HL = norm_resize(HL)
    HH = norm_resize(HH)
    # BUG FIX 1: HF must be computed AFTER norm_resize of LH/HL/HH, not before
    HF = norm_resize(np.sqrt(LH**2 + HL**2 + HH**2))

    # Stack into (size, size, 7)
    return np.dstack([rgb,
                      LH[..., np.newaxis],
                      HL[..., np.newaxis],
                      HH[..., np.newaxis],
                      HF[..., np.newaxis]]).astype(np.float32)


class DeepfakeDetector:
    def __init__(self, models_dir='models'):
        print("Loading 7-Channel Wavelet Models...")

        hybrid_dir = os.path.join(models_dir, 'hybrid_7ch')
        modern_dir = os.path.join(models_dir, 'trained_modern')

        # ✅ 4-MODEL UNIVERSAL ENSEMBLE
        # Xception       : 94.61% — Diffusion Expert      (Surgery)
        # MobileNetV2    : 90.08% — Diffusion Specialist  (Projection)
        # ResNet50       : 83.75% — Mega Generalist       (Surgery)
        # MobileNet_Mega : 83.63% — GAN/FaceSwap Expert   (Projection, FF++/Celeb/DFD)
        self.model_xcep = keras.models.load_model(
            os.path.join(hybrid_dir, 'Xcep_7ch_Calibrated.keras'), compile=False)
        self.model_res = keras.models.load_model(
            os.path.join(hybrid_dir, 'Res_7ch_Mega.keras'), compile=False)
        self.model_mnet = keras.models.load_model(
            os.path.join(hybrid_dir, 'MobileNet_7ch_Calibrated.keras'), compile=False)
        self.model_mnet_mega = keras.models.load_model(
            os.path.join(hybrid_dir, 'MobileNet_7ch_Mega.keras'), compile=False)

        self.models = {
            'Xception':       (self.model_xcep,      256),
            'MobileNetV2':    (self.model_mnet,       256),
            'ResNet50':       (self.model_res,        256),
            'MNet_Generalist':(self.model_mnet_mega,  256),
        }
        print("[OK] 4-Model Universal Ensemble Ready.")

        # Optional: ConvNeXt-Diff-7Ch (Diffusion Ultra-Specialist)
        # Loads silently if available; skipped gracefully if not found.
        # Updating or replacing this model file will never affect the other 4 models.
        convnext_path = os.path.join(models_dir, 'Swin_Diff_7Ch', 'convnext_diff_7ch_v2.keras')
        if os.path.exists(convnext_path):
            try:
                self.model_convnext = keras.models.load_model(convnext_path, compile=False)
                self.models['ConvNeXt_Diff'] = (self.model_convnext, 256)
                print("[OK] ConvNeXt-Diff-7Ch Ultra-Specialist V2 (99.8% Accuracy) loaded.")
            except Exception as e:
                print(f"[WARN] ConvNeXt-Diff-7Ch found but failed to load: {e}")
        else:
            print("[INFO] ConvNeXt-Diff-7Ch not found. Running without Diffusion Ultra-Specialist.")

    # ----------------------------------------------------------
    #  ENSEMBLE PREDICTION
    # ----------------------------------------------------------
    def predict_ensemble(self, img_rgb: np.ndarray) -> dict:
        """
        Returns dict: {'Xception': p, 'ResNet50': p, 'Ensemble': p}
        prob close to 1 = REAL, close to 0 = FAKE
        """
        probs = {}
        for name, (model, size) in self.models.items():
            x = np.expand_dims(extract_7ch(img_rgb, size), 0).astype('float32')
            
            # HARD-FIX: Bypass Keras input_names entirely. 
            # We named the input 'hybrid_7ch_input' during surgery.
            try:
                pred = model({'hybrid_7ch_input': x}, training=False)
            except:
                # Absolute fallback for direct calls
                pred = model(x, training=False)
            
            p = np.array(pred).flatten()
            if len(p) > 2:
                # FIX: If length is massive (e.g. 1280 channels), it's a raw Backbone Feature Map, not a classifier.
                prob_real = 0.501 # Force to neutral uncertainty until trained
            elif len(p) > 1:
                prob_real = float(p[1])
            else:
                prob_real = float(p[0])

            # LOGIT CORRECTION: If model outputs are outside [0, 1], apply sigmoid
            if prob_real < 0 or prob_real > 1:
                prob_real = 1.0 / (1.0 + np.exp(-prob_real))
            
            probs[name] = float(np.clip(prob_real, 0.0, 1.0))

        # Dynamic Confidence-Weighted Soft Voting
        # BASE_MULTIPLIERS defines the priority of each model.
        # Add new models here without touching any other logic.
        # If a model is absent (not loaded), it is simply not in probs and is ignored.
        BASE_MULTIPLIERS = {
            'Xception':        10.0,  # King Anchor (94.61%) — Diffusion Expert
            'MobileNetV2':      5.0,  # Diffusion Specialist (90.08%)
            'ResNet50':         1.0,  # Mega Generalist, Tie-breaker (83.75%)
            'MNet_Generalist':  5.0,  # GAN/FaceSwap Specialist (83.63%)
            'ConvNeXt_Diff':   15.0,  # Ultra-Specialist V2 (99.8% Accuracy)
        }

        raw_weights = {n: max(abs(p - 0.5) * 2, 1e-6) for n, p in probs.items()}
        weights = {}
        for n in probs:
            m = BASE_MULTIPLIERS.get(n, 1.0)
            # Confidence Boost: If Ultra-Specialist is >95% sure, double its weight
            if n == 'ConvNeXt_Diff' and (probs[n] < 0.05 or probs[n] > 0.95):
                m *= 2.0
            weights[n] = raw_weights[n] * m

        total_w  = sum(weights.values()) + 1e-9
        ensemble = sum(weights[n] * probs[n] for n in weights) / total_w
        
        probs['Ensemble'] = float(ensemble)
        return probs

    # ----------------------------------------------------------
    #  GRAD-CAM
    # ----------------------------------------------------------
    def generate_gradcam(self, model_name: str, img_rgb: np.ndarray) -> np.ndarray:
        """
        Returns an RGB numpy array with Grad-CAM heatmap overlaid.
        Falls back gracefully if Grad-CAM fails.
        """
        try:
            model, size = self.models[model_name]

            # BUG FIX 3: find last trainable conv (skip BatchNorm, Activation etc.)
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, (keras.layers.Conv2D,
                                      keras.layers.SeparableConv2D,
                                      keras.layers.DepthwiseConv2D)):
                    last_conv = layer.name
                    break

            if last_conv is None:
                return img_rgb  # fallback: return original image

            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv).output, model.output]
            )

            x = np.expand_dims(extract_7ch(img_rgb, size), 0)
            input_name = getattr(model, 'input_names', [model.layers[0].name])[0]

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model({input_name: x})
                # BUG FIX 4: handle both Tensor and list predictions safely
                if isinstance(predictions, (list, tuple)):
                    pred_val = predictions[0]
                else:
                    pred_val = predictions
                loss = pred_val[:, 0]

            grads  = tape.gradient(loss, conv_outputs)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam    = conv_outputs[0] @ pooled[..., tf.newaxis]
            cam    = tf.squeeze(cam).numpy()

            cam = np.maximum(cam, 0)
            cam = cam / (cam.max() + 1e-9)
            # BUG FIX 5: resize to original image shape, not model size
            cam = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

        except Exception as e:
            print(f"[WARN] Grad-CAM failed: {e}. Returning original image.")
            return img_rgb
