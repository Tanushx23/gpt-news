import math
import os
import time

import matplotlib.pyplot as plt
import torch

from data.prepare import get_dataloader
from model.gpt import GPT

config = {
    "vocab_size"     : 8000,
    "d_model"        : 384,
    "num_heads"      : 6,
    "num_layers"     : 6,
    "context_len"    : 128,
    "dropout"        : 0.2,
    "batch_size"     : 128,
    "max_steps"      : 16000,
    "warmup_steps"   : 300,     # NEW — stabilizes early training
    "eval_every"     : 500,
    "eval_batches"   : 50,      # NEW — was hardcoded to 20, small vs. new held-out val set
    "lr"             : 3e-4,
    "weight_decay"   : 0.1,
    "val_split"      : 0.1,
    "text_file"      : "headlines.txt",
    "tokenizer_file" : "bpe_tokenizer.json",
    "checkpoint_dir" : "checkpoints",
}


def build_optimizer(model, lr, weight_decay):
    """
    Excludes LayerNorm and bias params from weight decay — standard
    GPT-2 practice. Only decays weight matrices (Linear + Embedding),
    which is where weight decay actually helps generalization.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:  # biases, LayerNorm gains/offsets
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def build_scheduler(optimizer, warmup_steps, max_steps):
    """Linear warmup, then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader = get_dataloader(
        config["text_file"],
        config["tokenizer_file"],
        config["context_len"],
        config["batch_size"],
        val_split=config["val_split"],
    )

    model = GPT(
        vocab_size  = config["vocab_size"],
        d_model     = config["d_model"],
        num_heads   = config["num_heads"],
        num_layers  = config["num_layers"],
        context_len = config["context_len"],
        dropout     = config["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
    scheduler = build_scheduler(optimizer, config["warmup_steps"], config["max_steps"])

    train_losses, val_losses, steps_log = [], [], []
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    start_time = time.time()

    print(f"\nStarting training for {config['max_steps']} steps "
          f"(warmup: {config['warmup_steps']})...")

    for step in range(config["max_steps"]):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        model.train()
        logits, loss, _ = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % config["eval_every"] == 0 or step == config["max_steps"] - 1:
            model.eval()
            val_loss, val_steps = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss, _ = model(vx, targets=vy)
                    val_loss += vloss.item()
                    val_steps += 1
                    if val_steps >= config["eval_batches"]:
                        break
            val_loss /= val_steps
            train_loss = loss.item()
            elapsed = (time.time() - start_time) / 60
            current_lr = scheduler.get_last_lr()[0]

            print(f"Step {step:5d} | train: {train_loss:.4f} | val: {val_loss:.4f} "
                  f"| lr: {current_lr:.2e} | time: {elapsed:.1f}m")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            steps_log.append(step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step"        : step,
                    "model_state" : model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "val_loss"    : val_loss,
                    "config"      : config,
                }, os.path.join(config["checkpoint_dir"], "best_model.pt"))
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

    torch.save({
        "step"        : config["max_steps"],
        "model_state" : model.state_dict(),
        "config"      : config,
    }, os.path.join(config["checkpoint_dir"], "final_model.pt"))

    plt.figure(figsize=(12, 5))
    plt.plot(steps_log, train_losses, label="Train Loss", color="blue")
    plt.plot(steps_log, val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("GPT Training — headline-level split, BOS/EOS, warmup")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")

    return model, config


if __name__ == "__main__":
    train()
