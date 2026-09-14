from pathlib import Path
from typing import Any, Dict

# The 8 most frequent primary (first-listed) arXiv leaf categories, fixed order.
# See PROJECT_HANDOFF.md for how this list and the frequency numbers were derived.
TOP_CATEGORIES = [
    "hep-ph",
    "astro-ph",
    "cs.CV",
    "cond-mat.mes-hall",
    "quant-ph",
    "cond-mat.mtrl-sci",
    "hep-th",
    "gr-qc",
]

CATEGORY_TO_ID = {category: index for index, category in enumerate(TOP_CATEGORIES)}
ID_TO_CATEGORY = {index: category for category, index in CATEGORY_TO_ID.items()}


def get_config() -> Dict[str, Any]:
    """
    Returns:
        A static dictionary of model configuration variables:
            dataset_name (str): Hugging Face dataset identifier.
            batch_size (int): batch size of the model
            num_epochs (int): number of epochs of the model
            learning_rate (float): learning rate of the model
            context_size (int): maximum allowed abstract length (in tokens)
            model_dimension (int): dimension of the embedding vector space
            number_of_blocks (int): number of encoder blocks
            heads (int): number of attention heads
            feed_forward_dimension (int): hidden dimension of the feed forward blocks
            dropout (float): dropout rate
            num_classes (int): number of target classes
            model_folder (str): folder in which the weights will be saved
            model_basename (str): name of the model
            preload (str | None): epoch from which to load the weights
            tokenizer_file (str): file where the tokenizer is stored
            tokenizer_vocab_size (int): maximum vocabulary size for the WordPiece tokenizer
            tokenizer_min_frequency (int): minimum frequency of a subword to add it to the vocabulary
            experiment_name (str): tensorboard experiment name
            seed (int): seed of the model
            reports_folder (str): folder in which report plots/JSON summaries are saved
    """
    return {
        "dataset_name": "TimSchopf/arxiv_categories",
        "batch_size": 64,
        "num_epochs": 12,
        "learning_rate": 3 * 10**-4,
        "context_size": 256,
        "model_dimension": 256,
        "number_of_blocks": 4,
        "heads": 8,
        "feed_forward_dimension": 1024,
        "dropout": 0.1,
        "num_classes": len(TOP_CATEGORIES),
        "model_folder": "weights",
        "model_basename": "arxiv_classifier_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "tokenizer_vocab_size": 16000,
        "tokenizer_min_frequency": 3,
        "experiment_name": "runs/arxiv_classifier",
        "seed": 561,
        "reports_folder": "reports",
    }


def get_weights_file_path(
        config,
        epoch: str
    ) -> str:
    """
    Get the saved model weights file path for a given epoch.

    Args:
        config: Config file.
        epoch (str): Epoch from which to load the weights.

    Returns:
        str: Path to the saved weights of the model.
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)


def get_latest_weights(config) -> str:
    """
    Get the latest saved (numbered-epoch) model weights from a folder. Ignores
    non-numeric checkpoints such as the 'best' one from get_best_weights.

    Args:
        config: Config file.

    Returns:
        str: Path to the latest saved weights of the model, or None if none exist.
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filenames = list(Path(model_folder).glob(f"{model_basename}*"))

    numbered = [f for f in model_filenames if f.stem.split('_')[-1].isdigit()]
    if len(numbered) == 0:
        return None

    numbered.sort(key = lambda filename: int(filename.stem.split('_')[-1]))

    return str(numbered[-1])


def get_best_weights(config) -> str:
    """
    Get the checkpoint saved whenever validation macro-F1 improved during
    training (see train.py::train_model), i.e. the best-performing epoch
    regardless of how many epochs ran after it.

    Args:
        config: Config file.

    Returns:
        str: Path to the best-validation-macro-F1 checkpoint, or None if it doesn't exist.
    """
    path = Path(get_weights_file_path(config, 'best'))
    return str(path) if path.exists() else None
