# 🛡️ Advanced Hybrid Deepfake Detection Pipeline (IEEE Standard)

This project implements a lightweight, robust, and explainable deepfake detection system designed for high-stakes verification (KYC, Dating Apps, Social Media). The architecture follows current IEEE-standard best practices by combining spatial and frequency domain analysis.

## 🚀 Key Advanced Features (Publication-Ready)

1.  **🌊 Wavelet Frequency Features**: Incorporates Discrete Wavelet Transform (DWT) to highlight GAN-specific artifacts and high-frequency noise that are often invisible in the spatial domain.
2.  **⚖️ Hybrid Confidence-Weighted Voting**: Replaces simple soft voting with weight-normalized predictions. Each model (Xception, ResNet, EfficientNet) is weighted by its predictive certainty, reducing the impact of low-confidence outliers.
3.  **⏳ Temporal Confidence Smoothing (EMA)**: Implements Exponential Moving Average across video frames to eliminate flickering predictions and focus on persistent abnormalities.
4.  **🔍 Multi-Resolution Grad-CAM**: Fuses heatmaps from multiple convolutional levels (Mid vs. Top) to visualize both structural inconsistencies and subtle texture anomalies.
5.  **🏗️ Multi-Stage Disjoint Training**: Splits validation sets into meta-training and meta-validation stages to ensure the ensemble weights are robust and not overfitted to a specific dataset split.

## 📂 Project Structure

- `modules/preprocessing.py`: Frequency enhancement & face alignment.
- `modules/ensemble_model.py`: Weighted ensemble logic.
- `modules/video_pipeline.py`: Frame sampling & EMA smoothing.
- `modules/explainability.py`: Grad-CAM fusion.
- `modules/trainer.py`: Multi-stage training scripts.
- `main_pipeline.py`: Entry point for end-to-end detection.

## 🛠️ Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## 🎯 Usage
```bash
python main_pipeline.py --video path/to/your/video.mp4
```
The system will output a final classification label, a composite confidence score, and save an **Explainability Report** for any suspicious frames.
