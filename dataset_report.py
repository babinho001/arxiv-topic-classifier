"""
Generates dataset-level report assets: class distribution across splits and
the abstract length distribution. Needs only the dataset download -- no
tokenizer or trained model required, so this can be run before train.py.

Usage:
    python dataset_report.py
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # headless: we only ever save figures, never display them interactively
import matplotlib.pyplot as plt
import numpy as np

from config import TOP_CATEGORIES, get_config
from dataset import load_data
from evaluate import save_json


def plot_class_distribution(dataset, output_path):
    """
    Saves a grouped bar chart of per-class row counts, one group per split.

    Args:
        dataset: DatasetDict as returned by dataset.py::load_data.
        output_path: PNG file to write to.

    Returns:
        Dict[str, Counter]: split name -> Counter of label index -> count.
    """
    splits = list(dataset.keys())
    counts_by_split = {split: Counter(dataset[split]['label']) for split in splits}

    x = np.arange(len(TOP_CATEGORIES))
    width = 0.8 / len(splits)
    offsets = (np.arange(len(splits)) - (len(splits) - 1) / 2) * width

    fig, ax = plt.subplots(figsize = (10, 5))
    for offset, split in zip(offsets, splits):
        values = [counts_by_split[split].get(i, 0) for i in range(len(TOP_CATEGORIES))]
        ax.bar(x + offset, values, width = width, label = split)

    ax.set_xticks(x)
    ax.set_xticklabels(TOP_CATEGORIES, rotation = 45, ha = 'right')
    ax.set_ylabel('Number of abstracts')
    ax.set_title('Class distribution across splits')
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)

    return counts_by_split


def plot_abstract_length_histogram(dataset, config, output_path):
    """
    Saves a histogram of abstract lengths (train split), measured in
    whitespace-split words -- a fast proxy for tokenizer output length, used
    here because this script can run before any tokenizer is trained. A
    vertical line marks the context_size truncation budget.

    Args:
        dataset: DatasetDict as returned by dataset.py::load_data.
        config: A config file.
        output_path: PNG file to write to.

    Returns:
        np.ndarray: whitespace-word length of every train-split abstract.
    """
    lengths = np.array([len(text.split()) for text in dataset['train']['abstract']])
    truncation_budget = config['context_size'] - 2 # room reserved for [SOS]/[EOS]

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.hist(lengths, bins = 40, color = '#4C72B0')
    ax.axvline(
        truncation_budget, color = 'red', linestyle = '--',
        label = f"context_size budget ({truncation_budget} tokens)"
    )
    ax.set_xlabel('Abstract length (whitespace-split words, train split)')
    ax.set_ylabel('Number of abstracts')
    ax.set_title('Abstract length distribution')
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)

    return lengths


def main():
    config = get_config()
    output_dir = Path(config['reports_folder'])

    dataset = load_data(config)

    counts_by_split = plot_class_distribution(dataset, output_dir / 'dataset_class_distribution.png')
    lengths = plot_abstract_length_histogram(dataset, config, output_dir / 'dataset_abstract_length_hist.png')

    truncation_budget = config['context_size'] - 2
    stats = {
        'rows_per_split': {split: len(dataset[split]) for split in dataset.keys()},
        'class_counts_per_split': {
            split: {TOP_CATEGORIES[i]: counts_by_split[split].get(i, 0) for i in range(len(TOP_CATEGORIES))}
            for split in dataset.keys()
        },
        'abstract_whitespace_word_length_train': {
            'mean': float(lengths.mean()),
            'median': float(np.median(lengths)),
            'p90': float(np.percentile(lengths, 90)),
            'p95': float(np.percentile(lengths, 95)),
            'max': int(lengths.max()),
            'pct_truncated_at_context_size': float((lengths > truncation_budget).mean() * 100),
        },
    }
    save_json(stats, output_dir / 'dataset_stats.json')

    print(f"Saved dataset report assets to {output_dir}/")


if __name__ == "__main__":
    main()
