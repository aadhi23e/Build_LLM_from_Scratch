# download_datasets.py
import os
import re
import json
from datasets import load_dataset

# Target directories
PRETRAIN_JSONL = "./data/mini_gpt/pretrain/bookcorpusopen.jsonl"
PRETRAIN_BOOKS_DIR = "./data/mini_gpt/pretrain/books/"
POSTTRAIN_JSONL = "./data/mini_gpt/posttrain/nemotron_sft.jsonl"
POSTTRAIN_BOOKS_DIR = "./data/mini_gpt/posttrain/books/"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """Make safe filenames by removing non-ASCII and special chars."""
    safe = re.sub(r'[^a-zA-Z0-9_\- ]', '', name)
    return safe.strip().replace(" ", "_") + ".txt"

def save_subset_jsonl(dataset, out_path, sample_size=500):
    """Save a small subset of dataset to JSONL for prototyping."""
    ensure_dir(os.path.dirname(out_path))
    subset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
    with open(out_path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(subset)} samples to {out_path}")

def save_books(dataset, out_dir, sample_size=500):
    """Save each book/text as <title>.txt in UTF-8."""
    ensure_dir(out_dir)
    subset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
    for item in subset:
        title = item.get("title", "untitled")
        text = item.get("text", "")
        filename = sanitize_filename(title)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"Saved {len(subset)} books to {out_dir}")

def main():
    # --- Pretrain dataset: BookCorpusOpen ---
    print("Downloading BookCorpusOpen...")
    book_ds = load_dataset("lucadiliello/bookcorpusopen", split="train")
    # save_subset_jsonl(book_ds, PRETRAIN_JSONL, sample_size=500)
    save_books(book_ds, PRETRAIN_BOOKS_DIR, sample_size=500)

    # --- Posttrain dataset: Nemotron SFT ---
    # print("Downloading Nemotron-Cascade-2-SFT-Data...")
    # nemotron_ds = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data", split="train")
    # save_subset_jsonl(nemotron_ds, POSTTRAIN_JSONL, sample_size=1000)

if __name__ == "__main__":
    main()
