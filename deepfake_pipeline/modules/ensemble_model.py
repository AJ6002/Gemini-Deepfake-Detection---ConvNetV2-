import tensorflow as tf
import numpy as np

class EnsembleDeepfakeModel:
    def __init__(self, model_paths):
        """
        Loads the ensemble of models: Xception, ResNet50, EfficientNetB0.
        model_paths: dict with model names as keys and file paths as values.
        """
        self.models = {}
        for name, path in model_paths.items():
            print(f"Loading {name} from {path}...")
            # Using compiled=False if they are just weights or use custom layers
            self.models[name] = tf.keras.models.load_model(path, compile=False)
            
    def predict_weighted(self, preprocessed_images):
        """
        Hybrid Confidence-Weighted Voting:
        Weights each model based on its 'certainty' (distance from 0.5).
        preprocessed_images: dict of model-specific preprocessed inputs.
        """
        probs = {}
        confidences = {}
        
        for name, img in preprocessed_images.items():
            model = self.models[name]
            pred = model.predict(img, verbose=0)[0]
            
            # Standardize: (Real probability is always at index [1] or [0] depending on architecture)
            # In the user's notebook, Xception/ResNet were binary (index 0), EfficientNet was categorical (index 1)
            # Let's handle this based on output shape
            prob_real = pred[1] if len(pred) > 1 else pred[0]
            
            probs[name] = float(prob_real)
            
            # Hybrid Weighting Logic: Distance from 0.5 Decision Boundary
            # Near 0 or 1 is high confidence, near 0.5 is low confidence.
            confidence = abs(prob_real - 0.5) * 2.0
            confidences[name] = max(confidence, 1e-6) # Avoid division by zero
            
        # 1. Total confidence for normalization
        total_confidence = sum(confidences.values())
        
        # 2. Weighted Sum (Soft Voting)
        weighted_prob = 0.0
        for name in self.models.keys():
            weight = confidences[name] / total_confidence
            weighted_prob += weight * probs[name]
            
        # 3. Label conversion
        label = "Real" if weighted_prob >= 0.5 else "Fake"
        
        return {
            "label": label,
            "weighted_prob": weighted_prob,
            "individual_scores": probs,
            "confidence_weights": {k: v/total_confidence for k, v in confidences.items()}
        }
