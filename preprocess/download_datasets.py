"""
Download AG News and Yelp Review Full (5-class) datasets from HuggingFace
and save them as cleaned CSVs matching the existing cleanedData.csv schema:
    ID, Content, Category
"""

import os
import argparse

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

AGNEWS_LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

YELP_LABEL_MAP = {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}


def download_agnews(output_dir: str) -> str:
    """Download AG News train split and save as CSV."""
    print("Downloading AG News from HuggingFace...")
    ds = load_dataset("fancyzhx/ag_news", split="train")

    df = pd.DataFrame({
        "ID": range(len(ds)),
        "Content": ds["text"],
        "Category": [AGNEWS_LABEL_MAP[label] for label in ds["label"]],
    })

    # Drop rows where Content is missing or not a string
    df = df[df["Content"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    df = df.reset_index(drop=True)
    df["ID"] = range(len(df))

    out_path = os.path.join(output_dir, "agnews_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved AG News ({len(df)} samples) -> {out_path}")
    return out_path


def download_yelp(output_dir: str, subset_size: int = 40000, seed: int = 42) -> str:
    """Download Yelp Review Full (5-class) train split, subsample, and save as CSV."""
    print("Downloading Yelp Review Full from HuggingFace...")
    ds = load_dataset("Yelp/yelp_review_full", split="train")

    if subset_size and subset_size < len(ds):
        print(f"Subsampling {subset_size} from {len(ds)} samples (seed={seed})...")
        ds = ds.shuffle(seed=seed).select(range(subset_size))

    df = pd.DataFrame({
        "ID": range(len(ds)),
        "Content": ds["text"],
        "Category": [YELP_LABEL_MAP[label] for label in ds["label"]],
    })

    # Drop rows where Content is missing or not a string
    df = df[df["Content"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    df = df.reset_index(drop=True)
    df["ID"] = range(len(df))

    out_path = os.path.join(output_dir, "yelp_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved Yelp ({len(df)} samples) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download AG News and Yelp datasets from HuggingFace."
    )
    parser.add_argument(
        "--dataset",
        choices=["agnews", "yelp", "all"],
        default="all",
        help="Which dataset(s) to download (default: all)",
    )
    parser.add_argument(
        "--yelp-subset-size",
        type=int,
        default=40000,
        help="Number of Yelp samples to keep (default: 40000)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to save cleaned CSVs (default: preprocess/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Yelp subsampling (default: 42)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("agnews", "all"):
        download_agnews(args.output_dir)

    if args.dataset in ("yelp", "all"):
        download_yelp(args.output_dir, subset_size=args.yelp_subset_size, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
