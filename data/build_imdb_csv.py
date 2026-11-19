import csv
import os
from pathlib import Path


def read_reviews_from_folder(folder: Path, label: int, root: Path) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []

    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        review_id = file_path.relative_to(root).as_posix()
        rows.append((text, label, review_id))

    return rows


def build_split_csv(imdb_root: str, split: str, output_csv: str) -> None:
    """
    Build a CSV file from the Stanford IMDb dataset folders.

    Args:
        imdb_root: path to aclImdb root folder
        split: "train" or "test"
        output_csv: output CSV path
    """
    root = Path(imdb_root)
    pos_dir = root / split / "pos"
    neg_dir = root / split / "neg"

    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(
            f"Could not find expected folders: {pos_dir} and {neg_dir}"
        )

    rows = []
    rows.extend(read_reviews_from_folder(pos_dir, label=1, root=root))
    rows.extend(read_reviews_from_folder(neg_dir, label=0, root=root))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "review_id"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    imdb_root = data_dir / "aclImdb"

    build_split_csv(str(imdb_root), split="train", output_csv=str(data_dir / "imdb_train.csv"))
    build_split_csv(str(imdb_root), split="test", output_csv=str(data_dir / "imdb_test.csv"))