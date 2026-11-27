"""Train and select a configurable MLP for Question 2.2.

Run from the repository root with ``python src/run_q2_2.py``. Hyperparameters are selected
using validation accuracy only; the test split is evaluated after the final configuration is
selected. Results and plots are written to ``results/q2_2/``.
"""

import json
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
from train import count_parameters, get_device, set_seed, train

SELECTION_SEED = 17
FINAL_SEEDS = (0, 1, 2)
BATCH_SIZE = 128
SEARCH_EPOCHS = 18
FINAL_EPOCHS = 35
PATIENCE = 8


class ConfigurableMLP(nn.Module):
    """MLP with configurable hidden widths, activation, and dropout probability."""

    def __init__(self, hidden_widths=(512,), activation="relu", dropout=0.0):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        widths = [3 * 64 * 64, *hidden_widths, len(CLASSES)]
        layers = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(activations[activation]())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images.flatten(start_dim=1))


def predict(model, loader, device):
    predictions, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.append(model(images.to(device)).argmax(dim=1).cpu())
            labels_all.append(labels)
    return torch.cat(predictions), torch.cat(labels_all)


def fit(config, data, mean, std, seed, device, epochs, patience, verbose=False,
        evaluate_test=False):
    set_seed(seed)
    train_loader, val_loader, test_loader = image_loaders(
        data, batch_size=BATCH_SIZE, mean=mean, std=std
    )
    model = ConfigurableMLP(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn(current_model, batch):
        images, labels = batch
        return F.cross_entropy(current_model(images), labels)

    def validation_score(current_model):
        predictions, labels = predict(current_model, val_loader, device)
        return accuracy(predictions, labels)

    history = train(
        model, optimizer, loss_fn, train_loader, validation_score,
        epochs=epochs, patience=patience, device=device, verbose=verbose,
    )
    test_accuracy = None
    if evaluate_test:
        predictions, labels = predict(model, test_loader, device)
        test_accuracy = accuracy(predictions, labels)
    return model, history, max(history["val_score"]), test_accuracy


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), tensor.std(unbiased=True).item()


def main():
    data = load_coco_dominant10()
    mean, std = channel_stats(data["train"]["images"])
    device = get_device()

    # Small, explicit search: vary depth/width/activation/dropout, then select solely by
    # best validation accuracy from a fixed selection seed. Adam and its learning rate are
    # held fixed to keep the CPU search tractable and make the comparison interpretable.
    candidates = [
        {"hidden_widths": (256,), "activation": "relu", "dropout": 0.0},
        {"hidden_widths": (512,), "activation": "relu", "dropout": 0.0},
        {"hidden_widths": (512, 256), "activation": "relu", "dropout": 0.0},
        {"hidden_widths": (512,), "activation": "tanh", "dropout": 0.0},
        {"hidden_widths": (512,), "activation": "relu", "dropout": 0.3},
    ]
    search = []
    for index, config in enumerate(candidates, start=1):
        model, history, best_val, _ = fit(
            config, data, mean, std, SELECTION_SEED, device,
            epochs=SEARCH_EPOCHS, patience=5, verbose=False,
        )
        search.append({
            "candidate": index,
            "config": {**config, "hidden_widths": list(config["hidden_widths"])},
            "best_validation_accuracy": best_val,
            "best_epoch": history["best_epoch"],
            "parameters": count_parameters(model),
        })
        print(f"candidate {index}/{len(candidates)}: val={best_val:.4f}, best_epoch={history['best_epoch']}", flush=True)

    chosen_search = max(search, key=lambda item: item["best_validation_accuracy"])
    chosen = dict(chosen_search["config"])
    chosen["hidden_widths"] = tuple(chosen["hidden_widths"])

    final_runs = []
    final_models = []
    for seed in FINAL_SEEDS:
        model, history, best_val, test_score = fit(
            chosen, data, mean, std, seed, device,
            epochs=FINAL_EPOCHS, patience=PATIENCE, verbose=False, evaluate_test=True,
        )
        final_runs.append({
            "seed": seed,
            "best_epoch": history["best_epoch"],
            "best_validation_accuracy": best_val,
            "test_accuracy": test_score,
            "history": history,
        })
        final_models.append(model)
        print(f"final seed {seed}: val={best_val:.4f}, test={test_score:.4f}, epoch={history['best_epoch']}", flush=True)

    validation_mean, validation_std = mean_std([r["best_validation_accuracy"] for r in final_runs])
    test_mean, test_std = mean_std([r["test_accuracy"] for r in final_runs])
    summary = {
        "device": str(device),
        "search_seed": SELECTION_SEED,
        "final_seeds": list(FINAL_SEEDS),
        "batch_size": BATCH_SIZE,
        "search_max_epochs": SEARCH_EPOCHS,
        "final_max_epochs": FINAL_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "optimizer": {"name": "Adam", "learning_rate": 0.001},
        "selection_rule": "highest best validation accuracy on the fixed search seed",
        "search_candidates": search,
        "selected_config": {**chosen, "hidden_widths": list(chosen["hidden_widths"])},
        "parameters": count_parameters(final_models[0]),
        "validation_accuracy_mean": validation_mean,
        "validation_accuracy_std": validation_std,
        "test_accuracy_mean": test_mean,
        "test_accuracy_std": test_std,
        "runs": final_runs,
        "baseline_q2_1": {
            "learning_rate": 0.001,
            "test_accuracy_mean": 0.26733333865801495,
            "test_accuracy_std": 0.009073775261640549,
            "parameters": 122890,
        },
    }
    out = ROOT / "results" / "q2_2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in final_runs:
        history = run["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"Seed {run['seed']}")
        axes[1].plot(epochs, [100 * x for x in history["val_score"]], label=f"Seed {run['seed']}")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(title="Run")
    fig.suptitle("Q2.2 — Selected MLP")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "selected_config": summary["selected_config"],
        "parameters": summary["parameters"],
        "validation_accuracy": f"{validation_mean:.4f} ± {validation_std:.4f}",
        "test_accuracy": f"{test_mean:.4f} ± {test_std:.4f}",
        "results": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
