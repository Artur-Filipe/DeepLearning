"""Controlled one-factor-at-a-time MLP study for Question 2.3."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import ImageDataset, channel_stats, image_loaders
from load_data import CLASSES, load_coco_dominant10
from metrics import accuracy, confusion_matrix, per_class_accuracy
from run_q2_2 import ConfigurableMLP, predict
from train import count_parameters, get_device, set_seed, train

SEEDS = (0, 1, 2)
BASE_CONFIG = {"hidden_widths": (512, 256), "activation": "relu", "dropout": 0.0}
BATCH_SIZE = 128
MAX_EPOCHS = 18
PATIENCE = 5
LR = 1e-3

CONDITIONS = {
    "control": {"dropout": 0.0, "weight_decay": 0.0, "flip": False},
    "dropout_03": {"dropout": 0.3, "weight_decay": 0.0, "flip": False},
    "weight_decay_1e-4": {"dropout": 0.0, "weight_decay": 1e-4, "flip": False},
    "horizontal_flip": {"dropout": 0.0, "weight_decay": 0.0, "flip": True},
}


def fit_one(data, mean, std, condition, seed, device, evaluate_test=False, details=False):
    set_seed(seed)
    train_loader, val_loader, test_loader = image_loaders(
        data, batch_size=BATCH_SIZE, mean=mean, std=std, flip=condition["flip"]
    )
    train_eval_loader = DataLoader(
        ImageDataset(data["train"]["images"], data["train"]["labels"], mean, std),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    config = dict(BASE_CONFIG)
    config["dropout"] = condition["dropout"]
    model = ConfigurableMLP(**config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=condition["weight_decay"]
    )

    def loss_fn(current_model, batch):
        images, labels = batch
        return F.cross_entropy(current_model(images), labels)

    def val_score(current_model):
        pred, labels = predict(current_model, val_loader, device)
        return accuracy(pred, labels)

    history = train(
        model, optimizer, loss_fn, train_loader, val_score,
        epochs=MAX_EPOCHS, patience=PATIENCE, device=device, verbose=False,
    )
    result = {
        "best_epoch": history["best_epoch"],
        "best_validation_accuracy": max(history["val_score"]),
        "best_training_accuracy": accuracy(
            *predict(model, train_eval_loader, device)
        ),
        "history": history,
    }
    if evaluate_test:
        pred, labels = predict(model, test_loader, device)
        result["test_accuracy"] = accuracy(pred, labels)
        if details:
            result["per_class_accuracy"] = per_class_accuracy(pred, labels).tolist()
            result["confusion_matrix"] = confusion_matrix(pred, labels).tolist()
    return model, result


def stats(values):
    values = torch.tensor(values, dtype=torch.float64)
    return {"mean": values.mean().item(), "std": values.std(unbiased=True).item()}


def main():
    data = load_coco_dominant10()
    mean, std = channel_stats(data["train"]["images"])
    device = get_device()

    # First pass uses validation only across every condition. All conditions have the same
    # seed set, epoch cap, batch size, optimizer, LR, and patience.
    validation_results = {}
    for name, condition in CONDITIONS.items():
        runs = []
        for seed in SEEDS:
            _, result = fit_one(data, mean, std, condition, seed, device)
            runs.append({"seed": seed, **result})
            print(f"{name} seed {seed}: val={result['best_validation_accuracy']:.4f}, epoch={result['best_epoch']}", flush=True)
        validation_results[name] = {
            "condition": condition,
            "runs": runs,
            "validation": stats([r["best_validation_accuracy"] for r in runs]),
            "training_at_best_validation": stats(
                [r["best_training_accuracy"] for r in runs]
            ),
            "train_validation_accuracy_gap": stats(
                [r["best_training_accuracy"] - r["best_validation_accuracy"]
                 for r in runs]
            ),
            "mean_best_epoch": sum(r["best_epoch"] for r in runs) / len(runs),
        }

    # Pick solely by mean validation accuracy; evaluate test once per seed for the selected
    # condition, also collecting the required class-wise analysis and confusion matrix.
    selected_name = max(
        validation_results,
        key=lambda key: validation_results[key]["validation"]["mean"],
    )
    selected_condition = CONDITIONS[selected_name]
    final_runs = []
    for seed in SEEDS:
        _, result = fit_one(
            data, mean, std, selected_condition, seed, device,
            evaluate_test=True, details=True,
        )
        final_runs.append({"seed": seed, **result})
        print(f"selected {selected_name} seed {seed}: test={result['test_accuracy']:.4f}", flush=True)

    baseline_val_mean = validation_results["control"]["validation"]["mean"]
    for name, item in validation_results.items():
        paired_deltas = [
            run["best_validation_accuracy"] - control_run["best_validation_accuracy"]
            for run, control_run in zip(
                item["runs"], validation_results["control"]["runs"]
            )
        ]
        item["validation_delta_vs_control"] = stats(paired_deltas)

    summary = {
        "device": str(device),
        "seeds": list(SEEDS),
        "base_config": {**BASE_CONFIG, "hidden_widths": list(BASE_CONFIG["hidden_widths"])},
        "optimizer": "Adam",
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "equal_budget": "same maximum epochs, batch size, optimizer, LR, and seeds; early stopping uses validation accuracy",
        "conditions_validation_only": validation_results,
        "selected_condition_by_mean_validation": selected_name,
        "selected_model_parameters": count_parameters(
            ConfigurableMLP(**{**BASE_CONFIG, "dropout": selected_condition["dropout"]})
        ),
        "selected_final_runs": final_runs,
        "selected_test_accuracy": stats([r["test_accuracy"] for r in final_runs]),
        "selected_training_accuracy_at_best_validation": stats(
            [r["best_training_accuracy"] for r in final_runs]
        ),
        "selected_train_validation_accuracy_gap": stats(
            [r["best_training_accuracy"] - r["best_validation_accuracy"]
             for r in final_runs]
        ),
        "selected_per_class_accuracy_mean": torch.tensor(
            [r["per_class_accuracy"] for r in final_runs], dtype=torch.float64
        ).mean(dim=0).tolist(),
        "selected_confusion_matrix_sum": torch.tensor(
            [r["confusion_matrix"] for r in final_runs], dtype=torch.long
        ).sum(dim=0).tolist(),
        "classes": list(CLASSES),
    }

    out = ROOT / "results" / "q2_3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, item in validation_results.items():
        histories = [r["history"] for r in item["runs"]]
        epochs = min(len(h["train_loss"]) for h in histories)
        mean_loss = [sum(h["train_loss"][i] for h in histories) / len(histories) for i in range(epochs)]
        mean_val = [100 * sum(h["val_score"][i] for h in histories) / len(histories) for i in range(epochs)]
        axes[0].plot(range(1, epochs + 1), mean_loss, label=name)
        axes[1].plot(range(1, epochs + 1), mean_val, label=name)
    axes[0].set(title="Training loss (seed mean)", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation accuracy (seed mean)", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Q2.3 — One-factor-at-a-time study")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = torch.tensor(summary["selected_confusion_matrix_sum"])
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Test confusion matrix — {selected_name} (pooled over seeds)")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "selected_condition": selected_name,
        "selected_validation": summary["conditions_validation_only"][selected_name]["validation"],
        "selected_training": summary["selected_training_accuracy_at_best_validation"],
        "train_validation_gap": summary["selected_train_validation_accuracy_gap"],
        "selected_test": summary["selected_test_accuracy"],
        "results": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
