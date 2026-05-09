import cv2
import numpy as np
import pywt
from PIL import Image

class Preprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        # Load Haar Cascade for face detection (fast and lightweight)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.ipynb')
        # If the file path above fails, it's safer to use the standard name
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face_and_align(self, frame):
        """
        Detects the largest face and crops it with a margin.
        Future Enhancement: Landmark-based alignment.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            # Fallback: Just return a center crop
            h, w = frame.shape[:2]
            side = min(h, w)
            x, y = (w - side) // 2, (h - side) // 2
            return frame[y:y+side, x:x+side]
        
        # Select largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add margin (e.g., 20%)
        margin = int(w * 0.2)
        y1 = max(0, y - margin)
        y2 = min(frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame.shape[1], x + w + margin)
        
        face_crop = frame[y1:y2, x1:x2]
        return cv2.resize(face_crop, self.target_size)

    def extract_wavelet_features(self, img_array, alpha=0.5):
        """
        Wavelet Frequency Enhancement:
        Highlight high-frequency noise/artifacts by performing 1-level DWT
        and re-injecting detail coefficients back into the spatial image.
        """
        # img_array is (H, W, 3) in [0, 255]
        # Convert to YCbCr to work on Luminance for extraction
        ycrcb = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)
        
        # 1-level 2D Discrete Wavelet Transform (Haar)
        coeffs = pywt.dwt2(y_channel, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # High-frequency detail reconstruct (Set LL to 0)
        coeffs_hf = (np.zeros_like(LL), (LH, HL, HH))
        hf_reconstruct = pywt.idwt2(coeffs_hf, 'haar')
        
        # Normalize HF map for fusion
        hf_map = cv2.normalize(hf_reconstruct, None, 0, 255, cv2.NORM_MINMAX)
        
        # Spatial-Frequency Fusion: Blend HF artifacts back into RGB channels
        # This highlights GAN-specific artifacts like checkerboard noise or lack thereof.
        enhanced = img_array.astype(np.float32) + alpha * np.expand_dims(hf_map, axis=2)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced

    def prepare_for_inference(self, frame, model_name):
        """
        Unified pipeline for model-specific inference preparation.
        Includes face detection, alignment, and frequency enhancement.
        """
        # 1. Face alignment
        face = self.detect_face_and_align(frame)
        
        # 2. Wavelet Enhancement (Advanced Technique)
        face_enhanced = self.extract_wavelet_features(face)
        
        # 3. Model-specific sizing and normalization
        if model_name == 'xception':
            img = cv2.resize(face_enhanced, (256, 256))
            img = img.astype(np.float32) / 255.0
        elif model_name == 'resnet50':
            img = cv2.resize(face_enhanced, (256, 256))
            img = img.astype(np.float32) / 255.0
        elif model_name == 'efficientnet':
            img = cv2.resize(face_enhanced, (224, 224))
            # EfficientNet usually expects [0, 255] for Keras internal preprocessing
            img = img.astype(np.float32)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return np.expand_dims(img, axis=0)
