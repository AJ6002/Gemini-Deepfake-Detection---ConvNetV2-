import cv2
import numpy as np

class VideoDetectionPipeline:
    def __init__(self, ensemble_model, preprocessor, smoothing_gamma=0.7):
        """
        ensemble_model: instance of EnsembleDeepfakeModel.
        preprocessor: instance of Preprocessor.
        smoothing_gamma: EMA weight (0-1). Higher means more weight on the current frame.
        """
        self.ensemble = ensemble_model
        self.preprocessor = preprocessor
        self.gamma = smoothing_gamma

    def process_video(self, video_path, target_frames=10):
        """
        Video detection with Frame Sampling and Temporal Smoothing.
        video_path: Path to the .mp4/.avi file.
        target_frames: Number of frames to sample.
        """
        cap = cv2.VideoCapture(video_path)
        total_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_fps == 0:
            return {"error": "Video capture failed. Check file format or path."}

        # Sampling logic
        sample_indices = np.linspace(0, total_fps - 1, target_frames).astype(int)
        
        current_smoothed_prob = 0.5 # Neutral start
        frame_predictions = []
        
        for idx in range(total_fps):
            ret, frame = cap.read()
            if not ret: break
            
            if idx in sample_indices:
                # 1. Preprocess specific to each model
                preprocessed = {}
                for model_name in self.ensemble.models.keys():
                    preprocessed[model_name] = self.preprocessor.prepare_for_inference(frame, model_name)
                
                # 2. Ensemble prediction (Hybrid Weighted Voting)
                result = self.ensemble.predict_weighted(preprocessed)
                
                # 3. Temporal Smoothing (EMA)
                # First frame initializes the probability to avoid bias from the neutral 0.5.
                if len(frame_predictions) == 0:
                    current_smoothed_prob = result['weighted_prob']
                else:
                    current_smoothed_prob = self.gamma * result['weighted_prob'] + (1 - self.gamma) * current_smoothed_prob
                
                frame_predictions.append({
                    "frame_idx": idx,
                    "instant_prob": result['weighted_prob'],
                    "smoothed_prob": current_smoothed_prob,
                    "model_scores": result['individual_scores']
                })
        
        cap.release()
        
        # Video-level aggregation
        # If mean smoothed prob >= 0.5, label is Real.
        final_video_prob = np.mean([f['smoothed_prob'] for f in frame_predictions])
        final_label = "Real" if final_video_prob >= 0.5 else "Fake"
        
        return {
            "final_label": final_label,
            "video_prob": final_video_prob,
            "frames": frame_predictions
        }
