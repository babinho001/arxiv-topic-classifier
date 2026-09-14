import torch

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import ID_TO_CATEGORY


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
        print_message
    ):
    """
    Runs the model over the test set and reports accuracy, macro-F1 and a confusion matrix.

    Args:
        model: The trained encoder classifier.
        dataloader: Test DataLoader.
        device: Torch device the model lives on.
        print_message: Callable used to print the results.

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

    return accuracy, macro_f1, matrix
