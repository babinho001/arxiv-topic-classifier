# Torch stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Other files stuff
from dataset import AbstractDataset, load_data
from model import get_model
from config import get_weights_file_path, get_latest_weights, get_config
from evaluate import run_validation, run_test

# HuggingFace stuff
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.normalizers import BertNormalizer
from datasets import Dataset as HFDataset

# Easy access stuff
import warnings
from pathlib import Path
from tqdm import tqdm

# Set the random seed for this project, for reproducibility.
import random
SEED = get_config()["seed"]
torch.manual_seed(SEED)
random.seed(SEED)


def get_all_abstracts(dataset: HFDataset):
    """
    Yields abstracts from the provided dataset.

    Args:
        dataset (HFDataset): Dataset to iterate through.

    Yields:
        str: Abstract text from the dataset.
    """
    for item in dataset:
        yield item['abstract']


def get_or_build_tokenizer(
        config,
        dataset: HFDataset,
        force_rewrite: bool = False
    ) -> Tokenizer:
    """
    If the tokenizer file specified in the config doesn't exist, or if we force
    rewrite, then build a WordPiece tokenizer from scratch on the given dataset.
    Else, load the tokenizer from the specified file.

    Args:
        config: A config file.
        dataset (HFDataset): HuggingFace dataset of abstracts to build the tokenizer from.
        force_rewrite (bool): If the function should disregard the cached tokenizer file.

    Returns:
        Tokenizer: A WordPiece tokenizer trained on the abstracts.
    """
    tokenizer_path = Path(config['tokenizer_file'])

    if not Path.exists(tokenizer_path) or force_rewrite:

        # Subword tokenizer: rare/technical words split into meaningful pieces
        # instead of collapsing to [UNK] (see PROJECT_HANDOFF.md for why WordLevel
        # doesn't fit this corpus). Case is preserved (lowercase=False): acronyms
        # like "CNN" carry topical signal that lowercasing would destroy.
        tokenizer = Tokenizer(WordPiece(unk_token = '[UNK]'))
        tokenizer.normalizer = BertNormalizer(lowercase = False)
        tokenizer.pre_tokenizer = BertPreTokenizer()

        trainer = WordPieceTrainer(
            special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            vocab_size = config['tokenizer_vocab_size'],
            min_frequency = config['tokenizer_min_frequency']
        )

        tokenizer.train_from_iterator(get_all_abstracts(dataset), trainer = trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}.")
    return tokenizer


def get_dataset(config):
    """
    Initializes the training, validation and test datasets and the tokenizer.

    Args:
        config: A config file.

    Returns:
        DataLoader: Training dataset dataloader.
        DataLoader: Validation dataset dataloader.
        DataLoader: Test dataset dataloader.
        Tokenizer: Abstract tokenizer.
    """
    dataset = load_data(config)

    # Tokenizer is trained on the training split only, to avoid leaking
    # validation/test vocabulary into the model.
    tokenizer = get_or_build_tokenizer(config, dataset['train'], force_rewrite = True)

    training_dataset = AbstractDataset(dataset['train'], tokenizer, config['context_size'])
    validation_dataset = AbstractDataset(dataset['validation'], tokenizer, config['context_size'])
    test_dataset = AbstractDataset(dataset['test'], tokenizer, config['context_size'])

    training_dataloader = DataLoader(training_dataset, batch_size = config['batch_size'], shuffle = True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = config['batch_size'], shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False)

    return training_dataloader, validation_dataloader, test_dataloader, tokenizer


def train_model(config):
    """
    Train the encoder-only classifier with the given parameters.

    Args:
        config: A config file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    training_dataloader, validation_dataloader, test_dataloader, tokenizer = get_dataset(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_latest_weights(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from the beginning.")

    # Single-label classification: plain cross-entropy over class logits. Per
    # PROJECT_HANDOFF.md, class-weighted variants are a follow-up experiment,
    # not the baseline.
    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):

        model.train()
        batch_iterator = tqdm(training_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)

            logits = model(encoder_input, encoder_mask)
            loss = loss_function(logits, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run the validation at the end of every epoch.
        run_validation(model, validation_dataloader, loss_function, device, writer, global_step, lambda msg: batch_iterator.write(msg))

        # Save weights at certain 'milestone' epochs.
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        if epoch % 10 == 9 or epoch == 0 or epoch == config['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

    # Run the test evaluation at the end of training.
    run_test(model, test_dataloader, device, print)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
