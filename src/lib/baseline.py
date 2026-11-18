"""Replicate the WAMM CNN-BiLSTM baseline: Table VI of Osama et al., arXiv:2512.23610.

The model follows the setup the paper describes (§III-D, §IV, Tables IV and V):
  - stratified 80/20 split, class weights inversely proportional to class frequency;
  - input: ASCII codes of the first 100 characters of the request, zero-padded;
  - character embedding of size 100, a 1D convolution block, then 2 bidirectional
    LSTM layers, trained with Adam (default learning rate), batch 64, 5 epochs.
The paper leaves some settings out (LSTM width, conv filters and kernel); those are
named constants below and marked "Not in the paper".

Each notebook in src/notebooks/baseline runs this for one dataset.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from lib.datasets import ROOT, fetch_dataset

RESULTS_DIR = ROOT / "results"
PROGRESS_LOG_FILE = RESULTS_DIR / "progress.log"
RANDOM_SEED = 42
TEST_FRACTION = 0.2
ALLOWED_GAP_PERCENTAGE_POINTS = 5.0  # Largest accepted difference from the paper.

# Table VI of the paper, CNN-BiLSTM rows: dataset -> (accuracy %, macro F1 %).
PAPER_CNN_BILSTM_RESULTS = {
    "DS_Base.csv": (97.11, 81.58),
    "DS_Augmented_v1.csv": (87.03, 51.49),
    "DS_Augmented_v2.csv": (97.42, 57.61),
}

# Settings from Tables IV and V.
SEQUENCE_MAX_CHARACTERS = 100
CHARACTER_VOCABULARY_SIZE = 129  # ASCII 0-127 shifted up by one, so 0 means padding.
CHARACTER_EMBEDDING_SIZE = 100
LSTM_LAYER_COUNT = 2
TRAINING_EPOCHS = 5
TRAINING_BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 64  # Not in the paper.
CONVOLUTION_FILTER_COUNT = 128  # Not in the paper.
CONVOLUTION_KERNEL_WIDTH = 5  # Not in the paper.

PREDICTION_BATCH_SIZE = 4096
SECONDS_BETWEEN_PROGRESS_LOGS = 60


def log_progress(message: str) -> None:
    """Print, and append to results/progress.log so long notebook runs can be followed."""
    timestamped_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(timestamped_line, flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    with PROGRESS_LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(timestamped_line + "\n")


# ---------------------------------------------------------- Data loading

def load_dataset(dataset_name: str) -> tuple[pd.Series, np.ndarray, list[str]]:
    """Return (requests, integer labels, class names) for one dataset, dropping unlabeled rows."""
    dataset_frame = pd.read_csv(fetch_dataset(dataset_name), dtype=str, keep_default_na=False)
    if dataset_frame.shape[1] != 2:
        raise ValueError(
            f"{dataset_name}: expected 2 columns (request, label), got {dataset_frame.shape[1]}"
        )
    request_texts = dataset_frame.iloc[:, 0]
    label_texts = dataset_frame.iloc[:, 1].str.strip()
    is_labeled_request = (label_texts != "") & (request_texts != "")
    class_names, label_ids = np.unique(
        label_texts[is_labeled_request].to_numpy(), return_inverse=True
    )
    return (
        request_texts[is_labeled_request].reset_index(drop=True),
        label_ids.astype(np.int64),
        [str(class_name) for class_name in class_names],
    )


def split_train_test(request_texts: pd.Series, label_ids: np.ndarray):
    """Stratified 80/20 split; returns train_texts, test_texts, train_labels, test_labels."""
    train_row_indices, test_row_indices = train_test_split(
        np.arange(len(label_ids)),
        test_size=TEST_FRACTION,
        stratify=label_ids,
        random_state=RANDOM_SEED,
    )
    return (
        request_texts.iloc[train_row_indices].reset_index(drop=True),
        request_texts.iloc[test_row_indices].reset_index(drop=True),
        label_ids[train_row_indices],
        label_ids[test_row_indices],
    )


def balanced_class_weights(train_labels: np.ndarray, class_count: int) -> np.ndarray:
    """Weights inversely proportional to class frequency (sklearn's "balanced")."""
    return compute_class_weight("balanced", classes=np.arange(class_count), y=train_labels)


def encode_requests_as_ascii_codes(
    request_texts: pd.Series, max_characters: int = SEQUENCE_MAX_CHARACTERS
) -> np.ndarray:
    """ASCII code + 1 of the first `max_characters` characters, zero-padded; non-ASCII becomes '?'."""
    encoded_requests = np.zeros((len(request_texts), max_characters), np.uint8)
    for row_index, request_text in enumerate(request_texts):
        ascii_codes = np.frombuffer(
            request_text[:max_characters].encode("ascii", "replace"), np.uint8
        )
        encoded_requests[row_index, : len(ascii_codes)] = ascii_codes + 1
    return encoded_requests


# ---------------------------------------------------------------- Model

class CnnBiLstmClassifier(nn.Module):
    """Character embedding -> Conv1d + ReLU + max-pool -> 2-layer BiLSTM -> linear classifier."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.character_embedding = nn.Embedding(
            CHARACTER_VOCABULARY_SIZE, CHARACTER_EMBEDDING_SIZE, padding_idx=0
        )
        self.convolution_block = nn.Sequential(
            nn.Conv1d(
                CHARACTER_EMBEDDING_SIZE,
                CONVOLUTION_FILTER_COUNT,
                CONVOLUTION_KERNEL_WIDTH,
                padding=CONVOLUTION_KERNEL_WIDTH // 2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.bidirectional_lstm = nn.LSTM(
            CONVOLUTION_FILTER_COUNT,
            LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYER_COUNT,
            bidirectional=True,
            batch_first=True,
        )
        self.classification_head = nn.Linear(2 * LSTM_HIDDEN_SIZE, class_count)

    def forward(self, character_codes: torch.Tensor) -> torch.Tensor:
        embedded_sequence = self.character_embedding(character_codes.long())
        # Conv1d wants (batch, channels, length); the LSTM wants (batch, length, channels).
        convolved_sequence = self.convolution_block(embedded_sequence.transpose(1, 2)).transpose(1, 2)
        _, (final_hidden_states, _) = self.bidirectional_lstm(convolved_sequence)
        # The last layer's forward and backward final states.
        return self.classification_head(
            torch.cat([final_hidden_states[-2], final_hidden_states[-1]], dim=1)
        )


def _training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate_cnn_bilstm(
    train_texts, test_texts, train_labels, test_labels, class_names
) -> dict:
    """Train CNN-BiLSTM with a class-weighted loss; return metrics, timings and test predictions."""
    if len(train_texts) != len(train_labels) or len(test_texts) != len(test_labels):
        raise ValueError("texts and labels must have the same length")
    if len(train_labels) == 0 or len(test_labels) == 0:
        raise ValueError("train and test sets must not be empty")
    torch.manual_seed(RANDOM_SEED)
    device = _training_device()
    log_progress(f"CNN-BiLSTM training on {device} with {len(train_labels)} requests")

    train_character_codes = torch.from_numpy(encode_requests_as_ascii_codes(train_texts)).to(device)
    test_character_codes = torch.from_numpy(encode_requests_as_ascii_codes(test_texts)).to(device)
    train_label_tensor = torch.from_numpy(train_labels).to(device)
    cnn_bilstm_classifier = CnnBiLstmClassifier(len(class_names)).to(device)
    class_weight_tensor = torch.tensor(
        balanced_class_weights(train_labels, len(class_names)), dtype=torch.float32, device=device
    )
    weighted_loss_function = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(cnn_bilstm_classifier.parameters())

    training_start_time = time.perf_counter()
    _run_training_loop(
        cnn_bilstm_classifier,
        optimizer,
        weighted_loss_function,
        train_character_codes,
        train_label_tensor,
    )
    training_seconds = time.perf_counter() - training_start_time
    predicted_labels, inference_ms_per_request = _predict_in_batches(
        cnn_bilstm_classifier, test_character_codes
    )
    model_result = {
        "model": "CNN-BiLSTM",
        "training_seconds": training_seconds,
        "inference_ms_per_request": inference_ms_per_request,
        **score_predictions(test_labels, predicted_labels, class_names),
        "predicted_labels": predicted_labels,
    }
    log_progress(
        f"CNN-BiLSTM: accuracy {model_result['accuracy']:.2f}%, "
        f"macro F1 {model_result['macro_f1']:.2f}%, "
        f"trained in {training_seconds / 60:.1f} min, {inference_ms_per_request:.4f} ms/request"
    )
    return model_result


def _run_training_loop(
    model, optimizer, loss_function, train_character_codes, train_label_tensor
) -> None:
    """Shuffled mini-batch training, logging progress about once a minute."""
    model.train()
    train_sample_count = len(train_label_tensor)
    shuffle_generator = torch.Generator().manual_seed(RANDOM_SEED)
    steps_per_epoch = math.ceil(train_sample_count / TRAINING_BATCH_SIZE)
    total_step_count = TRAINING_EPOCHS * steps_per_epoch
    training_start_time = last_log_time = time.monotonic()
    for epoch_index in range(TRAINING_EPOCHS):
        shuffled_indices = torch.randperm(train_sample_count, generator=shuffle_generator).to(
            train_label_tensor.device
        )
        for step_index in range(steps_per_epoch):
            batch_indices = shuffled_indices[
                step_index * TRAINING_BATCH_SIZE : (step_index + 1) * TRAINING_BATCH_SIZE
            ]
            logits = model(train_character_codes[batch_indices])
            batch_loss = loss_function(logits, train_label_tensor[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            if time.monotonic() - last_log_time < SECONDS_BETWEEN_PROGRESS_LOGS:
                continue
            completed_step_count = epoch_index * steps_per_epoch + step_index + 1
            seconds_remaining = (
                (time.monotonic() - training_start_time) / completed_step_count
                * (total_step_count - completed_step_count)
            )
            log_progress(
                f"CNN-BiLSTM epoch {epoch_index + 1}/{TRAINING_EPOCHS} "
                f"step {step_index + 1}/{steps_per_epoch} loss {batch_loss.item():.4f}, "
                f"~{seconds_remaining / 60:.0f} min left"
            )
            last_log_time = time.monotonic()
        log_progress(f"CNN-BiLSTM finished epoch {epoch_index + 1}/{TRAINING_EPOCHS}")


@torch.no_grad()
def _predict_in_batches(model, character_codes: torch.Tensor) -> tuple[np.ndarray, float]:
    """Batched argmax predictions and the average milliseconds per request."""
    model.eval()
    sample_count = len(character_codes)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start_time = time.perf_counter()
    predicted_label_batches = [
        model(character_codes[batch_start : batch_start + PREDICTION_BATCH_SIZE]).argmax(dim=1).cpu()
        for batch_start in range(0, sample_count, PREDICTION_BATCH_SIZE)
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms_per_request = 1000 * (time.perf_counter() - inference_start_time) / max(sample_count, 1)
    return torch.cat(predicted_label_batches).numpy(), inference_ms_per_request


# ------------------------------------------------------------- Reporting

def score_predictions(true_labels, predicted_labels, class_names: list[str]) -> dict:
    """Accuracy and macro F1 in percent, plus the per-class report."""
    return {
        "accuracy": 100 * accuracy_score(true_labels, predicted_labels),
        "macro_f1": 100 * f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "per_class_report": classification_report(
            true_labels,
            predicted_labels,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


def compare_with_paper(dataset_name: str, model_result: dict) -> pd.DataFrame:
    """Our accuracy and macro F1 next to Table VI, with the gap in percentage points."""
    if dataset_name not in PAPER_CNN_BILSTM_RESULTS:
        raise KeyError(f"no paper result for {dataset_name!r}")
    paper_accuracy, paper_macro_f1 = PAPER_CNN_BILSTM_RESULTS[dataset_name]
    comparison_rows = []
    for metric_name, our_value, paper_value in (
        ("accuracy", model_result["accuracy"], paper_accuracy),
        ("macro_f1", model_result["macro_f1"], paper_macro_f1),
    ):
        gap_in_points = our_value - paper_value
        comparison_rows.append({
            "metric": metric_name,
            "ours": round(our_value, 2),
            "paper": paper_value,
            "gap_points": round(gap_in_points, 2),
            "within_5_points": abs(gap_in_points) <= ALLOWED_GAP_PERCENTAGE_POINTS,
        })
    return pd.DataFrame(comparison_rows).set_index("metric")


def save_results(dataset_name: str, model_result: dict, class_names: list[str]) -> Path:
    """Write metrics (no predictions or request text) to results/<dataset>.json atomically."""
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / f"{Path(dataset_name).stem}.json"
    serializable_result = {
        "dataset": dataset_name,
        "classes": class_names,
        "paper_accuracy_and_macro_f1": PAPER_CNN_BILSTM_RESULTS.get(dataset_name),
        **{key: value for key, value in model_result.items() if key != "predicted_labels"},
    }
    temporary_file = results_file.with_suffix(".json.tmp")
    temporary_file.write_text(json.dumps(serializable_result, indent=2), encoding="utf-8")
    temporary_file.replace(results_file)
    return results_file
