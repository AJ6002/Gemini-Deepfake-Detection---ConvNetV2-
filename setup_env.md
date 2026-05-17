# Environment Setup and Replication Guide

This document describes how to clean up existing environments and replicate the Hybrid 7-Channel Deepfake Detection System end-to-end on any standard Windows or WSL2 setup.

---

## Step 1: Clean Up Old Virtual Environments
Before recreating the virtual environment, safely delete any legacy, corrupted, or bloated virtual environments.

### On WSL2 / Linux / Git Bash
Run this command to completely remove the old virtual environment folder:
rm -rf venv-pbl-wsl

### On Windows PowerShell (Administrator)
Run this command:
Remove-Item -Recurse -Force venv-pbl-wsl

---

## Step 2: Create a Clean Virtual Environment
Initialize a fresh environment using Python 3.10 or 3.11.

### On WSL2 / Linux
1. Create the virtual environment:
   python3 -m venv venv-pbl-wsl
2. Activate the virtual environment:
   source venv-pbl-wsl/bin/activate

### On Windows Command Prompt (CMD)
1. Create the virtual environment:
   python -m venv venv-pbl-wsl
2. Activate the virtual environment:
   venv-pbl-wsl\Scripts\activate.bat

### On Windows PowerShell
1. Create the virtual environment:
   python -m venv venv-pbl-wsl
2. Activate the virtual environment:
   .\venv-pbl-wsl\Scripts\Activate.ps1

---

## Step 3: Upgrade Pip and Install Core Python Libraries
Once the virtual environment is active, upgrade pip and install the complete, unified dependency tree:
pip install --upgrade pip
pip install -r requirements.txt

---

## Step 4: WSL2 GPU Setup (RTX 3050 Compatibility)
To ensure TensorFlow and CUDA recognize your GPU within WSL2:
1. Ensure the Windows NVIDIA host drivers are up to date.
2. In WSL2, register the CUDA library location by running:
   export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
   (Note: This path is automatically managed by the system's python scripts on execution).

---

## Step 5: React Frontend Integration Setup
The frontend is built using React + Vite + TypeScript.

1. Navigate to the frontend directory:
   cd Deepfake_Security_Layer/Deepfake_Security_Layer/KYC_Deepfake_Verification
2. Install all node packages:
   npm install
3. Start the Vite hot-reloading development server:
   npm run dev

---

## Step 6: Deploying the Services

### Option A: Launching the Gradio Unified GUI
To scan images and videos using an interactive web dashboard:
python master_app.py

The GUI will be hosted locally at: http://localhost:7860

### Option B: Launching the KYC Flask API Backend
To start the production API service that authenticates React frontend requests:
python Deepfake_Security_Layer/Deepfake_Security_Layer/app_v2.py

The server will initialize on: http://localhost:5000

---

## Step 7: Model Weight Maintenance
Note that model binary files (*.keras, *.h5) are excluded from Git to prevent repository bloat. Before deploying, ensure the following trained weights are present in your workspace:

- Swin Specialist:
  models/Swin_Diff_7Ch/convnext_diff_7ch_v2.keras
- Legacy Generalist Ensemble:
  models/hybrid_7ch/Xcep_7ch_Calibrated.keras
  models/hybrid_7ch/MobileNet_7ch_Calibrated.keras
  models/hybrid_7ch/MobileNet_7ch_Mega.keras
  models/hybrid_7ch/Res_7ch_Mega.keras
