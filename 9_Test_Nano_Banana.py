import os
import cv2
import numpy as np
from logic import DeepfakeDetector

# Initialise the 4-model ensemble
print("⏳ Loading Ensemble (Xception, MobileNetV2, ResNet50, MNet_Generalist)...")
detector = DeepfakeDetector()
print("✅ Ready.\n")

# Path to the Nano-Banana Awesome Cases
NANO_ROOT = "/mnt/x/College_Docs/PBL/dataset/Awesome-Nano-Banana-images-main/Awesome-Nano-Banana-images-main/images"

def test_case(case_num):
    case_path = os.path.join(NANO_ROOT, f"case{case_num}")
    if not os.path.exists(case_path):
        print(f"❌ Case {case_num} not found.")
        return

    print(f"--- 🍌 TESTING CASE {case_num} ---")
    
    files = {
        "REAL (input.jpg) ": os.path.join(case_path, "input.jpg"),
        "FAKE (output.jpg)": os.path.join(case_path, "output.jpg")
    }

    for label, path in files.items():
        img = cv2.imread(path)
        if img is None:
            print(f"   ⚠️ Could not read {label}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run Ensemble
        probs = detector.predict_ensemble(img_rgb)
        ens = probs['Ensemble']
        verdict = "🟢 REAL" if ens > 0.5 else "🔴 FAKE"
        conf = ens if ens > 0.5 else 1.0 - ens
        
        print(f"\n   📄 {label}")
        print(f"      FINAL VERDICT: {verdict} ({conf*100:.1f}% confidence)")
        print(f"      Breakdown:")
        for name, p in probs.items():
            if name == 'Ensemble': continue
            v = "REAL" if p > 0.5 else "FAKE"
            c = p if p > 0.5 else 1.0 - p
            print(f"         - {name:15}: {v:4} ({c*100:.1f}%)")

if __name__ == "__main__":
    test_case(1)
    print("\n" + "="*40 + "\n")
    test_case(2)
