# config_generator.py
import json
import os

CONFIG = {
    "DATA_DIR": "image_captioning_project/data",
    "TRAIN_IMAGES": "image_captioning_project/data/raw_images/train",
    "VAL_IMAGES": "image_captioning_project/data/raw_images/val",
    "FEATURES_DIR": "image_captioning_project/data/features",
    "TOKENIZER_PATH": "image_captioning_project/data/tokenizer_artifacts/tokenizer.json",
    "VOCAB_PATH": "image_captioning_project/data/tokenizer_artifacts/vocab.txt",
    "MODEL_DIR": "image_captioning_project/models",
    "CHECKPOINT_DIR": "image_captioning_project/checkpoints",
    "MAX_LEN": 30,
    "VOCAB_SIZE": 10000,
    "EMBED_DIM": 256,
    "LSTM_UNITS": 512,
    "BATCH_SIZE": 64,
    "EPOCHS": 15,
    "LEARNING_RATE": 0.001
}

os.makedirs("image_captioning_project", exist_ok=True)
with open("image_captioning_project/config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=4)
print("✅ Configuration file saved at image_captioning_project/config.json")
