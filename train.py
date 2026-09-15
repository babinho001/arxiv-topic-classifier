# Torch stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Other files stuff
from dataset import AbstractDataset, load_data
from model import get_model
from config import get_weights_file_path, get_latest_weights, get_best_weights, get_config
from evaluate import run_validation, run_test, plot_training_curves, save_json

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
    reports_dir = Path(config['reports_folder'])
    reports_dir.mkdir(parents = True, exist_ok = True)
    metrics_history = []

    training_dataloader, validation_dataloader, test_dataloader, tokenizer = get_dataset(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])
    # AdamW (Adam + decoupled weight decay) instead of plain Adam: the first
    # 12-epoch run overfit almost immediately (best epoch was epoch 0, train
    # loss collapsed to ~0 while validation loss climbed every epoch after).
    # Weight decay penalizes the large weights that let that happen.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr = config['learning_rate'], eps = 1e-9, weight_decay = config['weight_decay']
    )

    initial_epoch = 0
    global_step = 0
    best_val_macro_f1 = -1.0
    preload = config['preload']
    model_filename = get_latest_weights(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        best_val_macro_f1 = state.get('val_macro_f1', -1.0)
    else:
        print("No model to preload, starting from the beginning.")

    # Single-label classification: plain cross-entropy over class logits. Per
    # PROJECT_HANDOFF.md, class-weighted variants are a follow-up experiment,
    # not the baseline.
    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):

        model.train()
        epoch_loss_total = 0.0
        epoch_examples = 0
        batch_iterator = tqdm(training_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)

            logits = model(encoder_input, encoder_mask)
            loss = loss_function(logits, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            epoch_loss_total += loss.item() * label.size(0)
            epoch_examples += label.size(0)

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        train_loss = epoch_loss_total / epoch_examples

        # Run the validation at the end of every epoch.
        val_loss, val_accuracy, val_macro_f1 = run_validation(
            model, validation_dataloader, loss_function, device, writer, global_step, lambda msg: batch_iterator.write(msg)
        )

        # Record per-epoch metrics for the training-curve report plot, saving
        # after every epoch so progress survives a Colab disconnect mid-run.
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_macro_f1': val_macro_f1,
        })
        save_json(metrics_history, reports_dir / 'metrics_history.json')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'val_macro_f1': val_macro_f1,
        }

        # Save weights at certain 'milestone' epochs.
        if epoch % 10 == 9 or epoch == 0 or epoch == config['num_epochs'] - 1:
            torch.save(checkpoint, get_weights_file_path(config, f'{epoch:02d}'))

        # Also save whenever validation macro-F1 improves, independent of the
        # milestone schedule above -- this is the checkpoint predict.py prefers,
        # so the best-performing epoch is never lost even if later epochs
        # overfit or num_epochs turns out to be more than needed.
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(checkpoint, get_weights_file_path(config, 'best'))
            print(f"New best validation macro-F1: {val_macro_f1:.4f} -- saved checkpoint.")

    plot_training_curves(metrics_history, reports_dir / 'training_curves.png')

    # Evaluate on the best checkpoint (by validation macro-F1), not whatever
    # the model's in-memory weights happen to be at the end of the loop --
    # those can differ if later epochs overfit past the best one, and
    # predict.py also prefers the best checkpoint, so this keeps the report's
    # test numbers consistent with what predict.py will actually produce.
    best_checkpoint_path = get_best_weights(config)
    if best_checkpoint_path is not None:
        best_state = torch.load(best_checkpoint_path, map_location = device)
        model.load_state_dict(best_state['model_state_dict'])
        print(
            f"Loaded best checkpoint (epoch {best_state['epoch']}, "
            f"val macro-F1 {best_state['val_macro_f1']:.4f}) for final test evaluation."
        )

    run_test(model, test_dataloader, device, print, output_dir = reports_dir)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
