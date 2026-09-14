import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # headless: we only ever save figures, never display them interactively
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import ID_TO_CATEGORY


def save_json(data, output_path) -> None:
    """
    Writes a dict/list to a JSON file, creating parent folders as needed.

    Args:
        data: JSON-serializable data.
        output_path: File to write to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    with open(output_path, 'w', encoding = 'utf-8') as fh:
        json.dump(data, fh, indent = 2, ensure_ascii = False)


def plot_confusion_matrix(
        matrix: np.ndarray,
        category_names,
        output_path
    ) -> None:
    """
    Saves a confusion matrix heatmap (with raw counts annotated per cell).

    Args:
        matrix (np.ndarray): Confusion matrix, shape (num_classes, num_classes).
        category_names: Class names, in the same order as the matrix rows/columns.
        output_path: PNG file to write to.
    """
    fig, ax = plt.subplots(figsize = (7, 6))
    im = ax.imshow(matrix, cmap = 'Blues')

    ax.set_xticks(range(len(category_names)))
    ax.set_yticks(range(len(category_names)))
    ax.set_xticklabels(category_names, rotation = 45, ha = 'right')
    ax.set_yticklabels(category_names)
    ax.set_xlabel('Predicted category')
    ax.set_ylabel('True category')
    ax.set_title('Confusion matrix (test set)')

    max_value = matrix.max() if matrix.size > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = 'white' if value > max_value / 2 else 'black'
            ax.text(j, i, str(value), ha = 'center', va = 'center', color = color)

    fig.colorbar(im, ax = ax)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)


def plot_per_class_metrics(
        all_labels,
        all_predictions,
        category_names,
        output_path
    ):
    """
    Saves a grouped bar chart of precision/recall/F1 per class, and returns
    the underlying numbers.

    Args:
        all_labels: True class indices.
        all_predictions: Predicted class indices.
        category_names: Class names, indexed the same way as the label indices.
        output_path: PNG file to write to.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): precision, recall, F1, support per class.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, labels = list(range(len(category_names))), zero_division = 0
    )

    x = np.arange(len(category_names))
    width = 0.25

    fig, ax = plt.subplots(figsize = (9, 5))
    ax.bar(x - width, precision, width = width, label = 'Precision')
    ax.bar(x, recall, width = width, label = 'Recall')
    ax.bar(x + width, f1, width = width, label = 'F1')

    ax.set_xticks(x)
    ax.set_xticklabels(category_names, rotation = 45, ha = 'right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Per-class precision / recall / F1 (test set)')
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)

    return precision, recall, f1, support


def plot_training_curves(
        history,
        output_path
    ) -> None:
    """
    Saves a two-panel figure: training/validation loss per epoch, and
    validation accuracy/macro-F1 per epoch.

    Args:
        history: List of per-epoch dicts with keys
            'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_macro_f1'
            (see train.py::train_model, which builds this).
        output_path: PNG file to write to.
    """
    epochs = [record['epoch'] for record in history]

    fig, axes = plt.subplots(1, 2, figsize = (11, 4))

    axes[0].plot(epochs, [record['train_loss'] for record in history], marker = 'o', label = 'Train loss')
    axes[0].plot(epochs, [record['val_loss'] for record in history], marker = 'o', label = 'Validation loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per epoch')
    axes[0].legend()

    axes[1].plot(epochs, [record['val_accuracy'] for record in history], marker = 'o', label = 'Validation accuracy')
    axes[1].plot(epochs, [record['val_macro_f1'] for record in history], marker = 'o', label = 'Validation macro-F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Validation metrics per epoch')
    axes[1].legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)


@torch.no_grad()
def run_validation(
        model,
        dataloader,
        loss_function,
        device,
        writer,
        global_step: int,
        print_message
    ):
    """
    Runs the model over the validation set and reports loss, accuracy and macro-F1.

    Args:
        model: The encoder classifier being trained.
        dataloader: Validation DataLoader.
        loss_function: Loss function used for training (for a comparable validation loss).
        device: Torch device the model lives on.
        writer: TensorBoard SummaryWriter, or None to skip logging.
        global_step (int): Current training step, for TensorBoard logging.
        print_message: Callable used to print progress (so it can play nicely with tqdm).

    Returns:
        (float, float, float): average loss, accuracy, macro-F1.
    """
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    for batch in dataloader:
        encoder_input = batch['encoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        label = batch['label'].to(device)

        logits = model(encoder_input, encoder_mask)
        loss = loss_function(logits, label)
        total_loss += loss.item() * label.size(0)

        predictions = torch.argmax(logits, dim = -1)
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(label.cpu().tolist())

    average_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average = 'macro')

    print_message(f"Validation loss: {average_loss:.4f}, accuracy: {accuracy:.4f}, macro-F1: {macro_f1:.4f}")

    if writer is not None:
        writer.add_scalar('validation_loss', average_loss, global_step)
        writer.add_scalar('validation_accuracy', accuracy, global_step)
        writer.add_scalar('validation_macro_f1', macro_f1, global_step)
        writer.flush()

    model.train()
    return average_loss, accuracy, macro_f1


@torch.no_grad()
def run_test(
        model,
        dataloader,
        device,
        print_message,
        output_dir = None
    ):
    """
    Runs the model over the test set and reports accuracy, macro-F1 and a confusion matrix.
    If output_dir is given, also saves report assets there: a confusion matrix
    plot, a per-class precision/recall/F1 plot, and a test_results.json summary
    (including the full sklearn classification report text).

    Args:
        model: The trained encoder classifier.
        dataloader: Test DataLoader.
        device: Torch device the model lives on.
        print_message: Callable used to print the results.
        output_dir: Folder to save report assets into, or None to skip saving.

    Returns:
        (float, float, np.ndarray): accuracy, macro-F1, confusion matrix.
    """
    model.eval()

    all_predictions = []
    all_labels = []

    for batch in dataloader:
        encoder_input = batch['encoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        label = batch['label'].to(device)

        logits = model(encoder_input, encoder_mask)
        predictions = torch.argmax(logits, dim = -1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(label.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average = 'macro')
    matrix = confusion_matrix(all_labels, all_predictions)
    category_names = [ID_TO_CATEGORY[i] for i in range(len(ID_TO_CATEGORY))]

    print_message(f"Test accuracy: {accuracy:.4f}, macro-F1: {macro_f1:.4f}")
    print_message(f"Categories (confusion matrix row/column order): {category_names}")
    print_message(f"Confusion matrix:\n{matrix}")

    if output_dir is not None:
        output_dir = Path(output_dir)

        plot_confusion_matrix(matrix, category_names, output_dir / 'confusion_matrix.png')
        precision, recall, f1, support = plot_per_class_metrics(
            all_labels, all_predictions, category_names, output_dir / 'per_class_metrics.png'
        )

        save_json({
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': matrix.tolist(),
            'category_order': category_names,
            'per_class': {
                category_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i]),
                }
                for i in range(len(category_names))
            },
            'classification_report': classification_report(
                all_labels, all_predictions, target_names = category_names, zero_division = 0
            ),
        }, output_dir / 'test_results.json')

        print_message(f"Saved test report assets to {output_dir}/")

    return accuracy, macro_f1, matrix
