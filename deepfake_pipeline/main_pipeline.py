import os
import argparse
from modules.preprocessing import Preprocessor
from modules.ensemble_model import EnsembleDeepfakeModel
from modules.video_pipeline import VideoDetectionPipeline
from modules.explainability import MultiResolutionGradCAM
import cv2

def main(video_path, model_paths):
    # 1. Initialize Components
    preprocessor = Preprocessor(target_size=(256, 256))
    ensemble = EnsembleDeepfakeModel(model_paths)
    pipeline = VideoDetectionPipeline(ensemble, preprocessor)
    
    # 2. Process Video
    print(f"Starting detection for: {video_path}")
    results = pipeline.process_video(video_path, target_frames=15)
    
    # 3. Handle Results
    if "error" in results:
        print(results["error"])
        return

    print(f"\nDetection Complete!")
    print(f"Final Label: {results['final_label']}")
    print(f"Composite confidence: {results['video_prob']:.4f}")
    
    # 4. Generate Explainability for a suspicious frame (if found)
    # Let's pick the frame with highest 'Fake' probability
    suspicious_frame_info = min(results['frames'], key=lambda x: x['smoothed_prob'])
    
    if results['final_label'] == "Fake":
        print(f"\nGenerating Explainability Report for Frame {suspicious_frame_info['frame_idx']}...")
        
        # Reload that specific frame for visualization
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, suspicious_frame_info['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Using Xception for Grad-CAM as it's typically the most detailed
            explainer = MultiResolutionGradCAM(ensemble.models['xception'])
            img_input = preprocessor.prepare_for_inference(frame, 'xception')
            
            heatmap = explainer.generate_multi_res_cam(img_input, 'xception')
            if heatmap is not None:
                overlay = explainer.overlay_on_image(img_input, heatmap)
                output_path = "explainability_report.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"Explainability report saved to: {output_path}")

if __name__ == "__main__":
    # Example paths - User should update these to their actual weight files
    DEFAULT_MODELS = {
        'xception': 'models/xception_latest.keras',
        'resnet50': 'models/resnet50_latest.keras',
        'efficientnet': 'models/efficientnet_latest.keras'
    }
    
    parser = argparse.ArgumentParser(description="Advanced Hybrid Deepfake Detection Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    
    main(args.video, DEFAULT_MODELS)
