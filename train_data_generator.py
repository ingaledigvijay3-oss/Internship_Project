#!/usr/bin/env python3
"""
day 2
suyash gupta
train_data_generator.py

Builds caption↔image alignment for MS COCO, encodes captions using an existing tokenizer,
downloads a small subset of images (optional), extracts CNN features (InceptionV3),
and provides a Keras-compatible generator that yields:
  ((image_features, caption_in), caption_out, sample_weights)

Also writes a summary log to data/alignment_log.txt.

Usage (quick smoke test):
  python src/training/train_data_generator.py --year 2014 --split train2014 --subset_images 50

Requirements:
  - tokenizer.json from your previous "Tokenizer & Vocabulary" step.
  - pip install tensorflow
"""

import os
import re
import json
import math
import time
import random
import argparse
import urllib.request
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clean_text(s: str) -> str:
    # Keep consistent with tokenizer creation step
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def add_tokens(s: str) -> str:
    return f"<start> {s} <end>"

def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    word_index = cfg["word_index"]             # token -> id
    reserved = cfg["reserved"]                 # dict: <pad>, <unk>, <start>, <end>
    max_len = int(cfg["max_len"])              # includes start/end
    index_word = {int(v): k for k, v in word_index.items()}
    return word_index, index_word, reserved, max_len

def encode_caption(text: str, word_index: dict, unk_id: int):
    return [word_index.get(tok, unk_id) for tok in text.split()]

def pad_to_len(ids, max_len, pad_id):
    arr = np.full((max_len,), pad_id, dtype=np.int32)
    n = min(len(ids), max_len)
    arr[:n] = ids[:n]
    return arr

def log_write(path, msg):
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


# -----------------------------
# COCO helpers
# -----------------------------
def load_coco_captions_json(annotations_path: str):
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data["annotations"]                 # list of {image_id, caption}
    images = data["images"]                    # list of {id, file_name}
    id2file = {img["id"]: img["file_name"] for img in images}
    return anns, id2file

def build_alignment(anns, id2file, subset_images=None, seed=42):
    # Build (image_id, file_name, raw_caption) tuples
    img_to_caps = defaultdict(list)
    for a in anns:
        img_to_caps[a["image_id"]].append(a["caption"])

    img_ids = list(img_to_caps.keys())
    if subset_images:
        random.seed(seed)
        img_ids = random.sample(img_ids, min(subset_images, len(img_ids)))

    rows = []
    skipped = 0
    for img_id in img_ids:
        fname = id2file.get(img_id)
        if not fname:
            skipped += 1
            continue
        for cap in img_to_caps[img_id]:
            rows.append((img_id, fname, cap))
    return rows, skipped

def coco_image_url(split: str, file_name: str):
    # split: "train2014" | "val2014"
    return f"http://images.cocodataset.org/{split}/{file_name}"

def maybe_download_images(rows, split, dest_dir, limit=None):
    ensure_dir(dest_dir)
    seen = set()
    count = 0
    for _, fname, _ in rows:
        if fname in seen:
            continue
        seen.add(fname)
        if limit and count >= limit:
            break
        out_path = os.path.join(dest_dir, fname)
        if not os.path.exists(out_path):
            try:
                urllib.request.urlretrieve(coco_image_url(split, fname), out_path)
            except Exception as e:
                print(f"Warning: failed to download {fname}: {e}")
                continue
        count += 1
    return count


# -----------------------------
# Feature extractor
# -----------------------------
def build_inception_pool():
    base = InceptionV3(include_top=False, weights="imagenet", pooling="avg")
    base.trainable = False
    return base

def load_and_preprocess_image(path, img_size=(299, 299)):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = preprocess_input(img)  # scales to [-1, 1] for InceptionV3
    return img


# -----------------------------
# Keras Sequence Generator
# -----------------------------
class CaptionDataGenerator(Sequence):
    """
    Yields:
      ([image_features, caption_in], caption_out, sample_weights)
    Shapes:
      image_features: (B, 2048)
      caption_in:     (B, T)   where T = max_len - 1
      caption_out:    (B, T)
      sample_weights: (B, T)   1 for non-pad, 0 for pad
    """
    def __init__(self, image_paths, caption_in, caption_out, pad_id, batch_size=32, shuffle=True, model=None):
        assert len(image_paths) == len(caption_in) == len(caption_out)
        self.image_paths = np.array(image_paths)
        self.caption_in = np.array(caption_in, dtype=np.int32)
        self.caption_out = np.array(caption_out, dtype=np.int32)
        self.pad_id = pad_id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = model or build_inception_pool()
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.image_paths[inds]
        cap_in = self.caption_in[inds]
        cap_out = self.caption_out[inds]

        # Load images and extract features
        imgs = tf.stack([load_and_preprocess_image(p) for p in paths], axis=0)
        feats = self.model(imgs, training=False).numpy().astype(np.float32)  # (B, 2048)

        # Mask for loss (ignore pads in labels)
        sample_weights = (cap_out != self.pad_id).astype(np.float32)  # (B, T)
        return [feats, cap_in], cap_out, sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--year", type=int, default=2014, choices=[2014, 2017])
    parser.add_argument("--split", type=str, default="train2014", choices=["train2014", "val2014"])
    parser.add_argument("--subset_images", type=int, default=50, help="Number of unique images to use for a quick test")
    parser.add_argument("--download_limit", type=int, default=None, help="Limit total image downloads (None=all subset images)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args([])

    random.seed(args.seed)
    np.random.seed(args.seed)

    coco_dir = os.path.join(args.base_dir, f"coco{args.year}")
    annot_dir = os.path.join(coco_dir, "annotations")
    tok_path = os.path.join(coco_dir, "tokenizer_artifacts", "tokenizer.json")
    images_dir = os.path.join(coco_dir, "images", args.split)
    log_path = os.path.join("data", "alignment_log.txt")
    ensure_dir(os.path.dirname(log_path))
    ensure_dir(images_dir)

    # 1) Load tokenizer
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Missing tokenizer.json at {tok_path}. Run the tokenizer step first.")
    word_index, index_word, reserved, MAX_LEN = load_tokenizer(tok_path)
    PAD_ID = reserved.get("<pad>", 0)
    UNK_ID = reserved.get("<unk>", 1)
    START_ID = reserved.get("<start>", 2)
    END_ID = reserved.get("<end>", 3)

    # 2) Load captions annotations
    capt_json = os.path.join(annot_dir, f"captions_{'train' if args.split=='train2014' else 'val'}{args.year}.json")

    if not os.path.exists(capt_json):
        raise FileNotFoundError(f"Missing {capt_json}. Download COCO annotations first.")
    anns, id2file = load_coco_captions_json(capt_json)

    # 3) Build alignment (subset for quick test)
    rows, skipped = build_alignment(anns, id2file, subset_images=args.subset_images, seed=args.seed)

    # 4) Verify tokenizer outputs and encode
    #    Clean and add <start>/<end>, then encode, then build shifted pairs and pad
    raw_caps = []
    encoded = []
    for _, _, c in rows:
        s = add_tokens(clean_text(c))
        raw_caps.append(s)
        ids = encode_caption(s, word_index, UNK_ID)
        encoded.append(ids)

    T = MAX_LEN - 1  # because we'll build (input=[:-1], target=[1:])
    cap_in = []
    cap_out = []
    pad_counts = []
    oov_count = 0
    tok_count = 0

    for ids in encoded:
        inp = ids[:-1]
        out = ids[1:]
        inp_p = pad_to_len(inp, T, PAD_ID)
        out_p = pad_to_len(out, T, PAD_ID)
        cap_in.append(inp_p)
        cap_out.append(out_p)

        pad_counts.append(int(np.sum(out_p == PAD_ID)))
        # OOV: count UNK tokens in original ids
        oov_count += sum(1 for t in ids if t == UNK_ID)
        tok_count += len(ids)

    cap_in = np.stack(cap_in)
    cap_out = np.stack(cap_out)

    # 5) Map image paths
    image_paths = [os.path.join(images_dir, fname) for _, fname, _ in rows]
    unique_files = len(set(image_paths))

    # 6) Optionally download a small set of images
    start_dl = time.time()
    downloaded = maybe_download_images(rows, args.split, images_dir, limit=args.download_limit)
    dl_time = time.time() - start_dl

    # 7) Build generator and test one batch
    gen = CaptionDataGenerator(
        image_paths=image_paths,
        caption_in=cap_in,
        caption_out=cap_out,
        pad_id=PAD_ID,
        batch_size=args.batch_size,
        shuffle=True,
        model=build_inception_pool()
    )
    [feats, X_in], Y_out, W = gen[0]  # first batch

    # 8) Write alignment log
    with open(log_path, "w", encoding="utf-8") as f:
        pass  # truncate

    log_write(log_path, "=== Caption Preparation + Data Alignment (COCO) ===")
    log_write(log_path, f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_write(log_path, f"Year: {args.year}, Split: {args.split}")
    log_write(log_path, f"Subset images requested: {args.subset_images}")
    log_write(log_path, f"Unique files referenced: {unique_files}, Downloaded now: {downloaded} in {dl_time:.2f}s")
    log_write(log_path, f"Total aligned caption rows: {len(rows)}, Skipped images (no filename): {skipped}")
    log_write(log_path, f"Tokenizer MAX_LEN: {MAX_LEN} (T_in_out={T})")
    lengths = [len(x) for x in encoded]
    p95 = int(np.percentile(lengths, 95)) if lengths else 0
    log_write(log_path, f"Caption length stats (incl. <start>/<end>): min={min(lengths) if lengths else 0}, "
                        f"p95={p95}, max={max(lengths) if lengths else 0}")
    log_write(log_path, f"Pad counts per sample (target): min={min(pad_counts) if pad_counts else 0}, "
                        f"max={max(pad_counts) if pad_counts else 0}")
    oov_rate = (oov_count / max(1, tok_count)) * 100.0
    log_write(log_path, f"OOV tokens in subset: {oov_count}/{tok_count} = {oov_rate:.2f}%")
    log_write(log_path, f"First batch shapes: feats={feats.shape}, X_in={X_in.shape}, Y_out={Y_out.shape}, W={W.shape}")
    log_write(log_path, f"Feature vector example (first row, first 5 vals): {feats[0, :5].tolist()}")

    # Show a few sample alignments
    for i in range(min(5, len(rows))):
        img_id, fname, _ = rows[i]
        s = raw_caps[i]
        ids = encoded[i]
        preview = " ".join(s.split()[:12])
        log_write(log_path, f"[sample {i}] image_id={img_id}, file={fname}")
        log_write(log_path, f"          caption='{preview} ...'")
        log_write(log_path, f"          ids(len={len(ids)}): {ids[:12]} ...")

    print(f"\nWrote log to: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
output:-
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
87910968/87910968 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step
=== Caption Preparation + Data Alignment (COCO) ===
Date/Time: 2025-11-13 14:53:20
Year: 2014, Split: train2014
Subset images requested: 50
Unique files referenced: 50, Downloaded now: 50 in 52.35s
Total aligned caption rows: 251, Skipped images (no filename): 0
Tokenizer MAX_LEN: 17 (T_in_out=16)
Caption length stats (incl. <start>/<end>): min=9, p95=16, max=28
Pad counts per sample (target): min=0, max=8
OOV tokens in subset: 11/3093 = 0.36%
First batch shapes: feats=(16, 2048), X_in=(16, 16), Y_out=(16, 16), W=(16, 16)
Feature vector example (first row, first 5 vals): [0.053123388439416885, 0.038219623267650604, 0.5897897481918335, 0.31992998719215393, 0.7930019497871399]
[sample 0] image_id=472938, file=COCO_train2014_000000472938.jpg
          caption='<start> a vine covered home with a garden terrace and wooden bench ...'
          ids(len=13): [2, 4, 4398, 131, 350, 9, 4, 802, 6350, 10, 99, 105] ...
[sample 1] image_id=472938, file=COCO_train2014_000000472938.jpg
          caption='<start> a courtyard area features a climbing tree climbing up a brick ...'
          ids(len=20): [2, 4, 1691, 106, 957, 4, 1308, 134, 1308, 35, 4, 348] ...
[sample 2] image_id=472938, file=COCO_train2014_000000472938.jpg
          caption='<start> a wooden bench sitting in front of a window <end> ...'
          ids(len=11): [2, 4, 99, 105, 14, 8, 41, 6, 4, 126, 3] ...
[sample 3] image_id=472938, file=COCO_train2014_000000472938.jpg
          caption='<start> a tan building facade with a bench out front <end> ...'
          ids(len=11): [2, 4, 875, 72, 4730, 9, 4, 105, 87, 41, 3] ...
[sample 4] image_id=472938, file=COCO_train2014_000000472938.jpg
          caption='<start> a window surrounded by ivy plants and a bench <end> ...'
          ids(len=11): [2, 4, 126, 402, 49, 4358, 632, 10, 4, 105, 3] ...

Wrote log to: data/alignment_log.txt
Done."""    