"""
File: coco_tokenizer_suyash.py
Author: Suyash Gupta
Project: Image Captioning - Tokenizer & Vocabulary Planning
Description:
    This script builds a tokenizer for the MS COCO dataset.
    It cleans captions, builds vocabulary, adds special tokens,
    computes max caption length, and saves tokenizer artifacts.

Output (as observed in Jupyter Notebook):
------------------------------------------
Downloading captions annotations (~241 MB)...
Done.
Extracting...
Done.
Loaded 414,113 train captions and 202,654 val captions
{'p95_len': 17, 'MAX_LEN_used': 17}
{'vocab_size': 10000, 'val_tokens': 2528021, 'val_oov_tokens': 11728, 'val_oov_rate_percent': 0.46}
Saved files: ['tokenizer.json', 'vocab.txt']
"""

import os, json, zipfile, urllib.request, re, math
from collections import Counter

# ========================== SETTINGS ==========================
YEAR = 2014  # Choose dataset year (2014 or 2017)
VOCAB_SIZE = 10000
RESERVED = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
BASE_DIR = "data/coco"
DATA_DIR = os.path.join(BASE_DIR, f"coco{YEAR}")
os.makedirs(DATA_DIR, exist_ok=True)

# URLs for COCO caption annotations
ANNOT_URLS = {
    2014: "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    2017: "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
ANNOT_ZIP = os.path.join(DATA_DIR, f"annotations_trainval{YEAR}.zip")
ANNOT_DIR = os.path.join(DATA_DIR, "annotations")

# ========================== 1) DOWNLOAD CAPTIONS ==========================
if not os.path.exists(ANNOT_ZIP):
    print(f"Downloading COCO {YEAR} captions (~240MB)...")
    urllib.request.urlretrieve(ANNOT_URLS[YEAR], ANNOT_ZIP)
    print("Download done.")

if not os.path.exists(ANNOT_DIR):
    print("Extracting annotations...")
    with zipfile.ZipFile(ANNOT_ZIP, "r") as z:
        z.extractall(DATA_DIR)
    print("Extraction done.")

train_json = os.path.join(ANNOT_DIR, f"captions_train{YEAR}.json")
val_json   = os.path.join(ANNOT_DIR, f"captions_val{YEAR}.json")

# Function to load captions from JSON file
def load_captions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ann["caption"] for ann in data["annotations"]]

train_raw = load_captions(train_json)
val_raw   = load_captions(val_json)
print(f"Loaded {len(train_raw):,} train and {len(val_raw):,} val captions")

# ========================== 2) CLEAN & ADD TOKENS ==========================
punct_regex = re.compile(r"[^a-z0-9\s]")   # Keep lowercase letters, digits, and spaces
space_regex = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s.lower().strip()
    s = punct_regex.sub(" ", s)
    s = space_regex.sub(" ", s).strip()
    return s

def add_tokens(s: str) -> str:
    return f"<start> {s} <end>"

train_clean = [add_tokens(clean_text(c)) for c in train_raw]
val_clean   = [add_tokens(clean_text(c)) for c in val_raw]

# ========================== 3) MAX CAPTION LENGTH ==========================
lengths = [len(s.split()) for s in train_clean]

def percentile(values, p=0.95):
    xs = sorted(values)
    if not xs: return 0
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c: return xs[int(k)]
    return int(round(xs[f] * (c - k) + xs[c] * (k - f)))

p95 = percentile(lengths, 0.95)
MAX_LEN = max(15, min(30, p95))  # Clamp between 15..30 words
print({"p95_len": p95, "MAX_LEN_used": MAX_LEN})

# ========================== 4) BUILD VOCABULARY ==========================
counter = Counter()
for cap in train_clean:
    for tok in cap.split():
        if tok not in RESERVED:
            counter[tok] += 1

kept = [w for w, _ in counter.most_common(VOCAB_SIZE - len(RESERVED))]

word_index = dict(RESERVED)  # token -> id
idx = len(RESERVED)
for w in kept:
    word_index[w] = idx
    idx += 1

index_word = {i: w for w, i in word_index.items()}
UNK_ID = RESERVED["<unk>"]

def to_ids(text: str):
    return [word_index.get(tok, UNK_ID) for tok in text.split()]

# ========================== 5) OOV RATE (VALIDATION) ==========================
val_ids = [to_ids(s) for s in val_clean]
val_token_count = sum(len(x) for x in val_ids)
val_oov_count = sum(1 for x in val_ids for t in x if t == UNK_ID)
oov_rate = (val_oov_count / max(1, val_token_count)) * 100.0

print({
    "vocab_size": len(word_index),
    "val_tokens": val_token_count,
    "val_oov_tokens": val_oov_count,
    "val_oov_rate_percent": round(oov_rate, 2)
})

# ========================== 6) SAVE TOKENIZER FILES ==========================
ARTI_DIR = os.path.join(DATA_DIR, "tokenizer_artifacts")
os.makedirs(ARTI_DIR, exist_ok=True)

# Save vocabulary text file
with open(os.path.join(ARTI_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
    for i in range(len(index_word)):
        f.write(index_word.get(i, "<unk>") + "\n")

# Save tokenizer config and stats
config = {
    "year": YEAR,
    "vocab_size": len(word_index),
    "max_len": MAX_LEN,
    "reserved": RESERVED,
    "preprocessing": {
        "lowercase": True,
        "strip_punctuation": True,
        "keep_digits": True,
        "note": "Applied clean_text(); added <start>/<end>"
    },
    "stats": {
        "train_captions": len(train_clean),
        "val_captions": len(val_clean),
        "p95_len": p95,
        "val_oov_rate_percent": round(oov_rate, 2)
    },
    "word_index": word_index,
}

with open(os.path.join(ARTI_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("Saved files in:", ARTI_DIR, "\n", os.listdir(ARTI_DIR))
