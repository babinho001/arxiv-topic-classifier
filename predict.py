"""
Runs inference with an already-trained model -- no retraining, no dataset
download. Needs only two saved artifacts on disk: the tokenizer
(config['tokenizer_file'], default tokenizer.json) and a model checkpoint
(config['model_folder']/*.pt, default weights/). Both are produced by a
train.py run.

Usage:
    python predict.py -t "Abstract text to classify."
    python predict.py -t "First abstract." -t "Second abstract."
    python predict.py -f sample_data/sample_abstracts.csv
    python predict.py --checkpoint weights/arxiv_classifier_07.pt -t "..."
"""

import argparse
import csv
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ID_TO_CATEGORY, get_best_weights, get_config, get_latest_weights
from model import get_model


def load_for_inference(
        config,
        checkpoint_path = None,
        device = None
    ):
    """
    Loads a trained tokenizer and model checkpoint, ready for inference.

    Args:
        config: A config file.
        checkpoint_path: Specific checkpoint file to load, or None to auto-select
            (prefers the best-validation-macro-F1 checkpoint, falling back to
            the latest numbered epoch if no 'best' checkpoint exists).
        device: Torch device to load onto, or None to auto-select.

    Returns:
        (EncoderClassifier, Tokenizer, torch.device): the model in eval mode,
            the tokenizer, and the device they live on.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at {tokenizer_path}. Run train.py first, "
            "or copy tokenizer.json next to this script."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    checkpoint_path = checkpoint_path or get_best_weights(config) or get_latest_weights(config)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {config['model_folder']}/. Run train.py first, "
            "or pass --checkpoint pointing at a saved .pt file."
        )

    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    state = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint {checkpoint_path} (epoch {state['epoch']}) on {device}.")
    return model, tokenizer, device


def predict(
        abstract_text: str,
        model,
        tokenizer: Tokenizer,
        config,
        device
    ):
    """
    Classifies one abstract, mirroring dataset.py::AbstractDataset's
    tokenize/pad/truncate logic exactly so inference matches training.

    Args:
        abstract_text (str): Abstract to classify.
        model: A loaded EncoderClassifier in eval mode.
        tokenizer (Tokenizer): The trained WordPiece tokenizer.
        config: A config file.
        device: Torch device the model lives on.

    Returns:
        List[Tuple[str, float]]: (category, probability) pairs, sorted by
            probability descending.
    """
    sos_id = tokenizer.token_to_id('[SOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    pad_id = tokenizer.token_to_id('[PAD]')

    context_size = config['context_size']
    max_body_tokens = context_size - 2
    tokens = tokenizer.encode(abstract_text).ids[:max_body_tokens]
    num_padding_tokens = context_size - len(tokens) - 2

    ids = [sos_id] + tokens + [eos_id] + [pad_id] * num_padding_tokens
    encoder_input = torch.tensor(ids, dtype = torch.int64, device = device).unsqueeze(0) # (1, context_size)
    encoder_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(1).int() # (1, 1, 1, context_size)

    with torch.no_grad():
        logits = model(encoder_input, encoder_mask)
        probabilities = torch.softmax(logits, dim = -1).squeeze(0)

    ranked = sorted(
        ((ID_TO_CATEGORY[i], probabilities[i].item()) for i in range(probabilities.size(0))),
        key = lambda pair: pair[1],
        reverse = True
    )
    return ranked


def read_input_texts(args) -> list:
    """
    Collects abstract texts from --text and/or --file arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List[str]: abstract texts to classify.
    """
    texts = list(args.text or [])

    if args.file:
        file_path = Path(args.file)
        if file_path.suffix.lower() == '.csv':
            with open(file_path, newline = '', encoding = 'utf-8') as fh:
                reader = csv.DictReader(fh)
                texts.extend(row['abstract'] for row in reader if row.get('abstract'))
        else:
            with open(file_path, encoding = 'utf-8') as fh:
                texts.extend(line.strip() for line in fh if line.strip())

    return texts


def main():
    parser = argparse.ArgumentParser(description = "Classify arXiv abstracts with a trained model (no retraining).")
    parser.add_argument('-t', '--text', action = 'append', help = "Abstract text to classify (repeatable).")
    parser.add_argument('-f', '--file', help = "Path to a .csv (with an 'abstract' column) or plain-text file (one abstract per line).")
    parser.add_argument('--checkpoint', help = "Specific checkpoint .pt file to load (defaults to the latest in weights/).")
    args = parser.parse_args()

    config = get_config()
    model, tokenizer, device = load_for_inference(config, args.checkpoint)

    texts = read_input_texts(args)
    if not texts:
        parser.error("Provide at least one abstract via --text/-t or --file/-f.")

    for text in texts:
        ranked = predict(text, model, tokenizer, config, device)
        top_category, top_probability = ranked[0]
        preview = text if len(text) <= 80 else text[:77] + '...'

        print(f"\n[{top_category}] (p={top_probability:.3f})  {preview}")
        print("  " + ", ".join(f"{name}={probability:.3f}" for name, probability in ranked))


if __name__ == "__main__":
    main()
