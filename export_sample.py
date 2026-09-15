"""
Exports a small sample of the dataset (one abstract per class, 8 rows total)
for inclusion in the submission zip, instead of the full dataset.

Usage:
    python export_sample.py
"""

import csv
from pathlib import Path

from config import TOP_CATEGORIES, get_config
from dataset import load_data


def export_sample(config, output_path) -> None:
    """
    Picks one example row per class from the test split (unseen during
    training, so it also doubles as a fair demo set for predict.py) and
    writes them to a CSV file.

    Args:
        config: A config file.
        output_path: CSV file to write to.
    """
    dataset = load_data(config)['test']

    rows_by_category = {category: None for category in TOP_CATEGORIES}
    for row in dataset:
        category = TOP_CATEGORIES[row['label']]
        if rows_by_category[category] is None:
            rows_by_category[category] = row
        if all(value is not None for value in rows_by_category.values()):
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    with open(output_path, 'w', newline = '', encoding = 'utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['id', 'title', 'abstract', 'category'])
        for category, row in rows_by_category.items():
            if row is not None:
                writer.writerow([row['id'], row['title'], row['abstract'], category])

    print(f"Saved {sum(v is not None for v in rows_by_category.values())} sample rows to {output_path}")


if __name__ == "__main__":
    export_sample(get_config(), 'sample_data/sample_abstracts.csv')
