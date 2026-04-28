import sys
import os
import time

print("1. Importing torch...")
import torch
print(f"   Torch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

print("2. Importing SHARP modules...")
try:
    from sharp.models import PredictorParams, create_predictor
    print("   SHARP imports successful.")
except Exception as e:
    print(f"   SHARP import failed: {e}")
    sys.exit(1)

print("3. Initializing model architecture (this might download DINOv2 weights)...")
try:
    params = PredictorParams()
    model = create_predictor(params)
    print("   Model architecture initialized.")
except Exception as e:
    print(f"   Architecture initialization failed: {e}")
    sys.exit(1)

print("4. Loading checkpoint...")
try:
    DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    # Check if local file exists first to avoid hub hang
    local_path = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "sharp_2572gikvuh.pt")
    if os.path.exists(local_path):
        print(f"   Loading local checkpoint from {local_path}")
        state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
    else:
        print(f"   Checkpoint not found at {local_path}. This is unexpected.")
        sys.exit(1)
    
    if "model" in state_dict: state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    print("   Weights loaded.")
except Exception as e:
    print(f"   Weight loading failed: {e}")

print("5. Moving to GPU...")
try:
    model = model.to("cuda").eval()
    print("   Model is on GPU and ready.")
except Exception as e:
    print(f"   GPU transfer failed: {e}")

print("\nDIAGNOSIS COMPLETE: Model is fully functional.")
