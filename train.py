import copy
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision.datasets import MNIST


RUNS = [("baseline", 85, None), ("onecycle", 12, 0.1), ("low_peak", 12, 0.02)]
SEEDS = [17, 42, 103]
BATCH_SIZE = 512


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(800, 500), nn.ReLU(), nn.Linear(500, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_data(device, root="data"):
    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(2026))

    def prepare(images, labels):
        return images[:, None].to(device).float() / 255, labels.to(device)

    return {
        "train": prepare(train.data[order[5000:]], train.targets[order[5000:]]),
        "validation": prepare(train.data[order[:5000]], train.targets[order[:5000]]),
        "test": prepare(test.data, test.targets),
    }


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, data):
    model.eval()
    images, labels = data
    correct = torch.zeros((), device=images.device)
    for x, y in zip(images.split(2048), labels.split(2048)):
        correct += (model(x).argmax(1) == y).sum()
    return 100 * correct.item() / len(labels)


def train(data, name="onecycle", epochs=12, max_lr=0.1, seed=17, output="results"):
    images, labels = data["train"]
    device = images.device
    folder = Path(output) / name / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    warmup = LeNet().to(device)
    optimizer = torch.optim.SGD(warmup.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(warmup(images[:BATCH_SIZE]), labels[:BATCH_SIZE])
        loss.backward()
        optimizer.step()
    synchronize(device)
    del warmup, optimizer

    set_seed(seed)
    model = LeNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    if max_lr is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1 + 0.0001 * step) ** -0.75,
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs,
            steps_per_epoch=math.ceil(len(labels) / BATCH_SIZE),
            pct_start=5 / 12, three_phase=True, anneal_strategy="linear",
            div_factor=max_lr / 0.01, final_div_factor=1000,
            base_momentum=0.8, max_momentum=0.95,
        )
    generator = torch.Generator().manual_seed(seed)
    history, results = [], []
    elapsed = 0
    checkpoint = None

    for epoch in range(1, epochs + 1):
        model.train()
        synchronize(device)
        started = time.perf_counter()
        order = torch.randperm(len(labels), generator=generator).to(device)
        loss_sum = torch.zeros((), device=device)
        for indices in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[indices])
            loss = nn.functional.cross_entropy(logits, labels[indices])
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * len(indices)
        synchronize(device)
        elapsed += time.perf_counter() - started
        train_loss = loss_sum.item() / len(labels)
        if not math.isfinite(train_loss):
            raise RuntimeError(f"Training diverged: {name}, seed={seed}, epoch={epoch}")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "validation_accuracy": evaluate(model, data["validation"]),
            "train_seconds": elapsed,
        })
        if epoch == 12 and epochs > 12:
            checkpoint = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{name}, seed={seed}, epoch={epoch}: val={history[-1]['validation_accuracy']:.2f}%")

    results.append({
        "method": name, "seed": seed, "epochs": epochs,
        "test_accuracy": evaluate(model, data["test"]), "train_seconds": elapsed,
    })
    torch.save(model.state_dict(), folder / "model.pt")
    if checkpoint is not None:
        model.load_state_dict(checkpoint)
        results.append({
            "method": "baseline_12", "seed": seed, "epochs": 12,
            "test_accuracy": evaluate(model, data["test"]),
            "train_seconds": history[11]["train_seconds"],
        })
    history = pd.DataFrame(history)
    results = pd.DataFrame(results)
    history.to_csv(folder / "history.csv", index=False)
    results.to_csv(folder / "results.csv", index=False)
    return history, results


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(device)
    results = []
    for index, seed in enumerate(SEEDS):
        for name, epochs, max_lr in RUNS[index:] + RUNS[:index]:
            _, scores = train(data, name, epochs, max_lr, seed)
            results.append(scores)
    results = pd.concat(results, ignore_index=True)
    results.to_csv("results/results.csv", index=False)
    print(results.groupby("method").agg(
        accuracy=("test_accuracy", "mean"), std=("test_accuracy", "std"),
        seconds=("train_seconds", "mean"),
    ).round(3))


if __name__ == "__main__":
    main()
