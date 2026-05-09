import os
import zipfile
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_prep

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_NAME = "Midjourney.zip"
OUTPUT_ROOT = os.path.join(BASE_DIR, "dataset", "ULTRA_SWIN_DATA")
TEMP_EXTRACT = os.path.join(BASE_DIR, "data", "temp_Midjourney")
LIMIT = 5000 

def run():
    # Check root first, then dataset/ folder
    zip_path = os.path.join(BASE_DIR, ZIP_NAME)
    if not os.path.exists(zip_path):
        zip_path = os.path.join(BASE_DIR, "dataset", ZIP_NAME)

    if not os.path.exists(zip_path):
        print(f"❌ Error: {ZIP_NAME} not found in root or dataset/ folder.")
        return

    print(f"📦 Extracting {zip_path}...")
    if os.path.exists(TEMP_EXTRACT): shutil.rmtree(TEMP_EXTRACT)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_EXTRACT)

    print(f"🚀 Processing Faces (Target: {LIMIT})...")
    count = core_prep.scan_directory(TEMP_EXTRACT, OUTPUT_ROOT, "fake", LIMIT, max_per_folder=3)
    
    shutil.rmtree(TEMP_EXTRACT)
    print(f"\n✅ SUCCESS: Added {count} Midjourney images.")
    print(f"📢 You can now safely delete '{ZIP_NAME}'.")

if __name__ == "__main__":
    run()
