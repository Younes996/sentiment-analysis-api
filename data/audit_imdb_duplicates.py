import csv
from hashlib import sha256
from pathlib import Path

from sklearn.model_selection import train_test_split


def read_reviews(csv_path: Path, partition: str) -> list[dict[str, str]]:
    """Read reviews from a CSV and attach their partition."""
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        row["partition"] = partition

    return rows


def group_by_fingerprint(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Group reviews by the SHA-256 fingerprint of their text."""
    groups: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        fingerprint = sha256(row["text"].encode("utf-8")).hexdigest()
        groups.setdefault(fingerprint, []).append(row)

    return groups


def find_cross_partition_groups(
    groups: dict[str, list[dict[str, str]]],
) -> list[list[dict[str, str]]]:
    """Return groups whose reviews appear in more than one partition."""
    cross_partition_groups: list[list[dict[str, str]]] = []

    for rows in groups.values():
        partitions = {row["partition"] for row in rows}
        if len(partitions) > 1:
            cross_partition_groups.append(rows)

    return cross_partition_groups


def audit_duplicates(
    train_csv: Path,
    test_csv: Path,
) -> list[list[dict[str, str]]]:
    """Find exact duplicate groups shared by the official train and test sets."""
    train_rows = read_reviews(train_csv, partition="train")
    test_rows = read_reviews(test_csv, partition="test")

    groups = group_by_fingerprint(train_rows + test_rows)
    return find_cross_partition_groups(groups)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent

    # Audit exact matches between the official IMDb train and test sets.
    cross_partition_groups = audit_duplicates(
        train_csv=data_dir / "imdb_train.csv",
        test_csv=data_dir / "imdb_test.csv",
    )
    print(
        "Exact duplicate groups shared by the official train and test sets: "
        f"{len(cross_partition_groups)}"
    )

    affected_reviews = {"train": 0, "test": 0}

    for group in cross_partition_groups:
        for row in group:
            partition = row["partition"]
            affected_reviews[partition] += 1

    print(f"Affected train reviews: {affected_reviews['train']}")
    print(f"Affected test reviews: {affected_reviews['test']}")

    conflicting_groups = 0

    for group in cross_partition_groups:
        labels = {row["label"] for row in group}
        if len(labels) > 1:
            conflicting_groups += 1

    print(f"Shared duplicate groups with conflicting labels: {conflicting_groups}")

    # Exclude matching train reviews to preserve the official test set.
    train_ids_to_exclude = {
        row["review_id"]
        for group in cross_partition_groups
        for row in group
        if row["partition"] == "train"
    }
    print(f"Unique train review IDs to exclude: {len(train_ids_to_exclude)}")

    train_rows = read_reviews(data_dir / "imdb_train.csv", partition="train")
    eligible_train_rows = [
        row for row in train_rows
        if row["review_id"] not in train_ids_to_exclude
    ]
    eligible_ids = {row["review_id"] for row in eligible_train_rows}
    assert eligible_ids.isdisjoint(train_ids_to_exclude)
    print(f"Eligible train reviews: {len(eligible_train_rows)}")

    train_groups = group_by_fingerprint(eligible_train_rows)
    train_duplicates = 0
    extra_reviews = 0
    conflicting_train_groups = 0
    for group in train_groups.values():
        if len(group) > 1:
            train_duplicates += 1
            extra_reviews += len(group) - 1
            labels = {row["label"] for row in group}
            if len(labels) > 1:
                conflicting_train_groups += 1
    print(f"Exact duplicate groups within eligible train reviews: {train_duplicates}")
    print(f"Extra reviews within eligible train duplicate groups: {extra_reviews}")
    print(
        "Eligible train duplicate groups with conflicting labels: "
        f"{conflicting_train_groups}"
    )

    # Pick a stable representative for each exact text.
    deduplicated_train_rows = [
        min(group, key=lambda row: row["review_id"])
        for group in train_groups.values()
    ]

    assert len(deduplicated_train_rows) == len(eligible_train_rows) - extra_reviews
    print(f"Eligible train reviews after deduplication: {len(deduplicated_train_rows)}")

    # Sort by ID so the seeded split is independent of CSV row order.
    ordered_train_rows = sorted(
        deduplicated_train_rows, key=lambda row: row["review_id"]
    )

    final_train_rows, validation_rows = train_test_split(
        ordered_train_rows,
        test_size=0.2,
        random_state=42,
        stratify=[row["label"] for row in ordered_train_rows],
    )
    for row in validation_rows:
        row["partition"] = "validation"

    print(f"Final train reviews: {len(final_train_rows)}")
    print(f"Validation reviews: {len(validation_rows)}")

    # Audit the final partitions while keeping all official test reviews.
    test_rows = read_reviews(data_dir / "imdb_test.csv", partition="test")
    final_groups = group_by_fingerprint(final_train_rows + validation_rows + test_rows)
    final_overlaps = find_cross_partition_groups(final_groups)
    print(f"Exact duplicate groups across final partitions: {len(final_overlaps)}")

    internal_duplicate_counts = {"train": 0, "validation": 0}
    for group in final_groups.values():
        for partition in internal_duplicate_counts:
            if sum(row["partition"] == partition for row in group) > 1:
                internal_duplicate_counts[partition] += 1

    for partition, count in internal_duplicate_counts.items():
        print(f"Exact duplicate groups within final {partition}: {count}")

    if final_overlaps or any(internal_duplicate_counts.values()):
        raise RuntimeError("Final partitions failed the exact duplicate audit.")

    # Save the audited split so both models use the same partitions.
    output_path = data_dir / "imdb_partitions.csv"

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["review_id", "partition"])

        for row in final_train_rows + validation_rows + test_rows:
            writer.writerow([row["review_id"], row["partition"]])