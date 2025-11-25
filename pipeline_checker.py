# lead_pipeline_checker.py
import os
import json
import numpy as np

CONFIG_PATH = "image_captioning_project/config.json"

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

def check_tokenizer():
    print("🔍 Checking tokenizer...")
    if os.path.exists(cfg["TOKENIZER_PATH"]):
        with open(cfg["TOKENIZER_PATH"], "r") as f:
            tok = json.load(f)
        print(f"✅ Tokenizer found with vocab size: {tok['vocab_size']}")
    else:
        print("⚠️ Tokenizer file missing!")

def check_features():
    print("🔍 Checking image feature files...")
    feature_dir = cfg["FEATURES_DIR"]
    if os.path.exists(feature_dir) and len(os.listdir(feature_dir)) > 0:
        print(f"✅ Found {len(os.listdir(feature_dir))} feature files")
    else:
        print("⚠️ No extracted features found in", feature_dir)

def check_annotations():
    ann_dir = os.path.join(cfg["DATA_DIR"], "annotations")
    if os.path.exists(ann_dir) and len(os.listdir(ann_dir)) > 0:
        print(f"✅ Annotations directory contains {len(os.listdir(ann_dir))} files")
    else:
        print("⚠️ Annotations missing!")

def main():
    print("\n🚀 Running pipeline verification...\n")
    check_annotations()
    check_tokenizer()
    check_features()
    print("\n✅ Verification completed.")

if __name__ == "__main__":
    main()