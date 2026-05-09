import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_prep

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Target the 53k pool
SRC_DIR = os.path.join(BASE_DIR, "data", "images", "fake")
OUTPUT_ROOT = os.path.join(BASE_DIR, "dataset", "ULTRA_SWIN_DATA")
LIMIT = 10000 # Take a huge chunk of Flux/SDXL goldmine

def run():
    if not os.path.exists(SRC_DIR):
        print(f"❌ Error: {SRC_DIR} not found.")
        return

    print(f"🔍 Scanning Modern Goldmine (FLUX/SDXL)...")
    # Set max_per_folder to 5000 because these folders contain unique people, not identity variations
    count = core_prep.scan_directory(SRC_DIR, OUTPUT_ROOT, "fake", LIMIT, max_per_folder=5000)
    
    print(f"\n✅ SUCCESS: Added {count} images from the Modern Goldmine.")

if __name__ == "__main__":
    run()
