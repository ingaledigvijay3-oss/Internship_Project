# lead_setup_structure.py
import os

PROJECT_ROOT = "Image_capioning_teamE"

DIRS = [
    "data/raw_images",
    "data/annotations",
    "data/preprocessed",
    "data/features",
    "data/tokenizer_artifacts",
    "models",
    "checkpoints",
    "logs",
    "results"
]

def setup_project_structure():
    for d in DIRS:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
    print(f"✅ Project structure created under '{PROJECT_ROOT}'")

if __name__ == "__main__":
    setup_project_structure()
