import torch
import torch.nn as nn
from model.gpt import GPT
from data.prepare import get_dataloader
import os
import time
import matplotlib.pyplot as plt

config = {
    "vocab_size"      : 8000,
    "d_model"         : 384,
    "num_heads"       : 6,
    "num_layers"      : 6,
    "context_len"     : 128,
    "dropout"         : 0.2,
    "batch_size"      : 64,
    "max_steps"       : 4000,
    "eval_every"      : 250,
    "lr"              : 3e-4,
    "text_file"       : "headlines.txt",
    "tokenizer_file"  : "bpe_tokenizer.json",
    "checkpoint_dir"  : "checkpoints",
}

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader = get_dataloader(
        config["text_file"],
        config["tokenizer_file"],
        config["context_len"],
        config["batch_size"]
    )

    model = GPT(
        vocab_size  = config["vocab_size"],
        d_model     = config["d_model"],
        num_heads   = config["num_heads"],
        num_layers  = config["num_layers"],
        context_len = config["context_len"],
        dropout     = config["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["max_steps"]
    )

    train_losses = []
    val_losses   = []
    steps        = []

    train_iter = iter(train_loader)
    best_val_loss = float("inf")

    print(f"\nStarting training for {config['max_steps']} steps...")
    start_time = time.time()

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
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss, _ = model(vx, targets=vy)
                    val_loss += vloss.item()
                    val_steps += 1
                    if val_steps >= 20:
                        break

            val_loss /= val_steps
            train_loss = loss.item()
            elapsed = (time.time() - start_time) / 60
            current_lr = scheduler.get_last_lr()[0]

            print(f"Step {step:4d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | lr: {current_lr:.2e} | time: {elapsed:.1f}m")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            steps.append(step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step"        : step,
                    "model_state" : model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "val_loss"    : val_loss,
                    "config"      : config
                }, os.path.join(config["checkpoint_dir"], "best_model.pt"))
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    torch.save({
        "step"        : config["max_steps"],
        "model_state" : model.state_dict(),
        "config"      : config
    }, os.path.join(config["checkpoint_dir"], "final_model.pt"))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label="Train Loss", color="blue")
    plt.plot(steps, val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("GPT Training on Indian News Headlines")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")

    return model, config

if __name__ == "__main__":
    model, config = train()
