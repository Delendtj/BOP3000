from ultralytics.data.split import autosplit
import argparse
import random
import shutil
from pathlib import Path

def splitter(source, train: float, val: float, test: float):
    source = Path(source)
    images_dir = source / "images"
    labels_dir = source / "labels"

    assert images_dir.exists(), f"Missing images dir: {images_dir}"
    assert labels_dir.exists(), f"Missing labels dir: {labels_dir}"

    images = list(images_dir.glob("*"))
    random.shuffle(images)

    n = len(images)
    n_train = int(train * n)
    n_val = int(val * n)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        out_images = source / split / "images"
        out_labels = source / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for img in files:
            label = labels_dir / f"{img.stem}.txt"

            shutil.copy2(img, out_images / img.name)

            if label.exists():
                shutil.copy2(label, out_labels / label.name)
            else:
                # Create empty label file if missing (YOLO expects this)
                (out_labels / f"{img.stem}.txt").touch()

    print("Dataset split complete!")
    for k, v in splits.items():
        print(f"{k}: {len(v)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test folders")
    parser.add_argument("--source", required=True, help="Dataset root containing images/ and labels/")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)

    args = parser.parse_args()

    assert abs((args.train + args.val + args.test) - 1.0) < 1e-6, "Splits must sum to 1.0"

    splitter(args.source, args.train, args.val, args.test)




