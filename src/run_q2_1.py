"""Run Question 2.1: dataset summary and a linear classification baseline.

Usage from the repository root::

    .venv\\Scripts\\python.exe src\\run_q2_1.py

The dataset cache is written under ``src/data/`` and the reproducible results are written to
``results/q2_1/``. The cache is ignored by Git; the summary and plots are not.
"""

import json
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import channel_stats, image_loaders
from load_data import CLASSES, load_coco_dominant10
from metrics import accuracy
from train import get_device, set_seed, train


SEEDS = (0, 1, 2)
BATCH_SIZE = 128
EPOCHS = 25
DEFAULT_LEARNING_RATE = 0.001


class LinearClassifier(nn.Module):
    """Multinomial logistic regression over the flattened 3x64x64 image."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.linear = nn.Linear(3 * 64 * 64, num_classes)

    def forward(self, images):
        return self.linear(images.flatten(start_dim=1))


def predictions(model, loader, device):
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_predictions.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)
    return torch.cat(all_predictions), torch.cat(all_labels)


def run_one_seed(data, mean, std, seed, device, learning_rate):
    set_seed(seed)
    train_loader, validation_loader, test_loader = image_loaders(
        data, batch_size=BATCH_SIZE, mean=mean, std=std
    )
    model = LinearClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def loss_fn(current_model, batch):
        images, labels = batch
        return F.cross_entropy(current_model(images), labels)

    def validation_accuracy(current_model):
        preds, labels = predictions(current_model, validation_loader, device)
        return accuracy(preds, labels)

    history = train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        validation_accuracy,
        epochs=EPOCHS,
        device=device,
    )
    test_predictions, test_labels = predictions(model, test_loader, device)
    return history, accuracy(test_predictions, test_labels), model


def main():
    parser = argparse.ArgumentParser(description="Run the Q2.1 linear baseline.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--output-suffix", default="", help="Optional result folder suffix.")
    args = parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    data = load_coco_dominant10()
    mean, std = channel_stats(data["train"]["images"])
    device = get_device()
    runs = []
    for seed in SEEDS:
        history, test_accuracy, model = run_one_seed(
            data, mean, std, seed, device, args.learning_rate
        )
        runs.append({
            "seed": seed,
            "best_epoch": history["best_epoch"],
            "best_validation_accuracy": max(history["val_score"]),
            "test_accuracy": test_accuracy,
            "history": history,
        })

    counts = {
        split: torch.bincount(data[split]["labels"], minlength=len(CLASSES)).tolist()
        for split in ("train", "validation", "test")
    }
    summary = {
        "seeds": list(SEEDS),
        "device": str(device),
        "classes": list(CLASSES),
        "splits": {
            split: {
                "images": int(data[split]["images"].shape[0]),
                "image_shape": list(data[split]["images"].shape[1:]),
                "captions": len(data[split]["captions"]),
            }
            for split in ("train", "validation", "test")
        },
        "class_counts": counts,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "configuration": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "SGD",
            "learning_rate": args.learning_rate,
            "model": "linear classifier / multinomial logistic regression",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "best_validation_accuracy_mean": sum(run["best_validation_accuracy"] for run in runs) / len(runs),
        "best_validation_accuracy_std": torch.tensor([run["best_validation_accuracy"] for run in runs]).std(unbiased=True).item(),
        "test_accuracy_mean": sum(run["test_accuracy"] for run in runs) / len(runs),
        "test_accuracy_std": torch.tensor([run["test_accuracy"] for run in runs]).std(unbiased=True).item(),
        "runs": runs,
    }

    result_name = "q2_1" + (f"_{args.output_suffix}" if args.output_suffix else "")
    output_dir = ROOT / "results" / result_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in runs:
        history = run["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        loss_axis.plot(epochs, history["train_loss"], label=f"Seed {run['seed']}")
        accuracy_axis.plot(epochs, [100 * score for score in history["val_score"]],
                           label=f"Seed {run['seed']}")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training cross-entropy")
    loss_axis.set_title("Training loss")
    loss_axis.grid(alpha=0.25)
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Validation accuracy (%)")
    accuracy_axis.set_title("Validation accuracy")
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(title="Run")
    figure.suptitle("Q2.1 — Linear classification baseline")
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(figure)

    print(json.dumps({
        "device": str(device),
        "seeds": list(SEEDS),
        "test_accuracy_mean": round(summary["test_accuracy_mean"], 4),
        "test_accuracy_std": round(summary["test_accuracy_std"], 4),
        "results": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
