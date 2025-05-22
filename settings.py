import os
from pathlib import Path

# Paths
BASE_DIR = Path("/nas-ctm01/homes/dpmoura/xai-post-hoc")
DATA_DIR = BASE_DIR / "cifar-10-batches-py"

# Training
LR = 1e-3
MAX_EPOCHS = 25
BATCH_SIZE = 128  

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent  # This ensures we are inside scripts_lightning

# Define the models directory inside the scripts_lightning directory
MODEL_DIR = SCRIPT_DIR / "models"

# Create the models directory if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Define checkpoint path
CHECKPOINT_PATH = MODEL_DIR / "checkpoints"
