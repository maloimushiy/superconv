import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision.datasets import CIFAR10, MNIST

from models import LeNet, ResNet20


RUNS = [("baseline", 85, None), ("onecycle", 12, 0.1), ("low_peak", 12, 0.02)]
SEEDS = [17, 42, 103]


def load_data(device, root="data", dataset="mnist", train_size=None):
    dataset_class = {"mnist": MNIST, "cifar10": CIFAR10}[dataset]
    train = dataset_class(root, train=True, download=True)
    test = dataset_class(root, train=False, download=True)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(2026))

    def prepare(source):
        images = torch.as_tensor(source.data)
        images = images[:, None] if dataset == "mnist" else images.permute(0, 3, 1, 2)
        images = images.to(device).float() / 255
        if dataset == "cifar10":
            mean = images.new_tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]
            std = images.new_tensor([0.2470, 0.2435, 0.2616])[None, :, None, None]
            images = (images - mean) / std
        return images, torch.as_tensor(source.targets, device=device)

    images, labels = prepare(train)
    training_ids = order[5000:][:train_size].to(device)
    validation_ids = order[:5000].to(device)
    return {
        "train": (images[training_ids], labels[training_ids]),
        "validation": (images[validation_ids], labels[validation_ids]),
        "test": prepare(test),
    }


def augment(images):
    batch = len(images)
    padded = nn.functional.pad(images, (4, 4, 4, 4), mode="reflect")
    offsets = torch.randint(9, (2, batch), device=images.device)
    crops = padded.unfold(2, 32, 1).unfold(3, 32, 1)
    images = crops[torch.arange(batch, device=images.device), :, offsets[0], offsets[1]]
    flip = torch.rand(batch, device=images.device) < 0.5
    images[flip] = images[flip].flip(-1)
    return images


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
    loss, correct = torch.zeros(2, device=images.device)
    for x, y in zip(images.split(512), labels.split(512)):
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
    return loss.item() / len(labels), 100 * correct.item() / len(labels)


def train(data, name="onecycle", epochs=12, max_lr=0.1, seed=17, output="results",
          schedule=None, lr=0.01, weight_decay=0.0005, cycle_momentum=True,
          batch_size=None, diagnostics_every=20, live=None, evaluate_test=True,
          pct_start=5 / 12, three_phase=True):
    images, labels = data["train"]
    cifar = images.shape[1] == 3
    device = images.device
    batch_size = batch_size or (128 if cifar else 512)
    schedule = schedule or ("inverse" if max_lr is None else "onecycle")
    model_class = ResNet20 if cifar else LeNet
    folder = Path(output) / name / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(name=name, epochs=epochs, max_lr=max_lr, seed=seed, schedule=schedule,
                  lr=lr, weight_decay=weight_decay, cycle_momentum=cycle_momentum,
                  batch_size=batch_size, train_size=len(labels), dataset="cifar10" if cifar else "mnist",
                  evaluate_test=evaluate_test, pct_start=pct_start, three_phase=three_phase)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    set_seed(seed)
    warmup = model_class().to(device)
    for _ in range(5):
        warmup.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(warmup(images[:batch_size]), labels[:batch_size]).backward()
    synchronize(device)
    del warmup
    set_seed(seed)
    model = model_class().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    total_steps = epochs * math.ceil(len(labels) / batch_size)
    if schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, three_phase=three_phase, anneal_strategy="linear",
            div_factor=max_lr / lr, final_div_factor=1000,
            cycle_momentum=cycle_momentum, base_momentum=0.8, max_momentum=0.95,
        )
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=lr / 1000)
    else:
        if schedule != "inverse":
            raise ValueError(schedule)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1 + 0.0001 * step) ** -0.75)
    generator = torch.Generator().manual_seed(seed)
    weights = {key: p for key, p in model.named_parameters() if p.ndim >= 2}
    probe = tuple(tensor[:5000] for tensor in data["train"])
    history, diagnostics, results = [], [], []
    elapsed, step, checkpoint = 0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        synchronize(device)
        started = time.perf_counter()
        order = torch.randperm(len(labels), generator=generator).to(device)
        loss_sum = torch.zeros((), device=device)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            x = augment(images[indices]) if cifar else images[indices]
            loss = nn.functional.cross_entropy(model(x), labels[indices])
            loss.backward()
            sampled = diagnostics_every > 0 and step % diagnostics_every == 0
            if sampled:
                before = [p.detach().clone() for p in weights.values()]
                grad_l2 = sum(p.grad.square().sum() for p in weights.values()).sqrt().item()
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if sampled:
                weight_l2 = sum(p.square().sum() for p in before).sqrt()
                delta_l2 = sum((p.detach() - old).square().sum() for p, old in zip(weights.values(), before)).sqrt()
                diagnostics.append(dict(
                    step=step + 1, epoch=epoch, lr=current_lr, grad_l2=grad_l2,
                    weight_l2=weight_l2.item(), update_ratio=(delta_l2 / weight_l2.clamp_min(1e-12)).item(),
                    **{f"rms/{key}": p.detach().square().mean().sqrt().item() for key, p in weights.items()},
                ))
            scheduler.step()
            loss_sum += loss.detach() * len(indices)
            step += 1
        synchronize(device)
        elapsed += time.perf_counter() - started
        train_loss = loss_sum.item() / len(labels)
        if not math.isfinite(train_loss):
            raise RuntimeError(f"Training diverged: {name}, seed={seed}, epoch={epoch}")
        train_eval_loss, train_accuracy = evaluate(model, probe)
        val_loss, val_accuracy = evaluate(model, data["validation"])
        history.append(dict(epoch=epoch, train_loss=train_loss, train_eval_loss=train_eval_loss,
                            train_accuracy=train_accuracy, validation_loss=val_loss,
                            validation_accuracy=val_accuracy, gap=val_loss - train_eval_loss,
                            train_seconds=elapsed, lr=current_lr))
        if name == "baseline" and epoch == 12 and epochs > 12:
            checkpoint = copy.deepcopy(model.state_dict())
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(folder / "diagnostics.csv", index=False)
        if live is not None:
            live(history, diagnostics, f"{name} · seed {seed} · epoch {epoch}/{epochs}")
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{name}, seed={seed}, epoch={epoch}: val={val_accuracy:.2f}%, train={elapsed:.1f}s", flush=True)

    split = "test" if evaluate_test else "validation"
    score_loss, score_accuracy = evaluate(model, data[split])
    results.append(dict(method=name, seed=seed, epochs=epochs, train_seconds=elapsed,
                        **{f"{split}_accuracy": score_accuracy, f"{split}_loss": score_loss}))
    torch.save(model.state_dict(), folder / "model.pt")
    if checkpoint is not None and evaluate_test:
        model.load_state_dict(checkpoint)
        test_loss, test_accuracy = evaluate(model, data["test"])
        results.append(dict(method="baseline_12", seed=seed, epochs=12,
                            test_accuracy=test_accuracy, test_loss=test_loss,
                            train_seconds=history[11]["train_seconds"]))
    results = pd.DataFrame(results)
    results.to_csv(folder / "results.csv", index=False)
    return pd.DataFrame(history), results


def main():
    torch.set_num_threads(4)
    data = load_data(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    for index, seed in enumerate(SEEDS):
        for name, epochs, max_lr in RUNS[index:] + RUNS[:index]:
            _, scores = train(data, name, epochs, max_lr, seed)
            results.append(scores)
    pd.concat(results).to_csv("results/results.csv", index=False)


if __name__ == "__main__":
    main()
