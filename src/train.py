# src/train.py

import os
import argparse
import torch
import torch.nn.functional as F


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data_utils import read_text_file, create_token_mappings, create_dataloaders
from model import Transformer

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Read text
    text = read_text_file(args.data_file)

    # 2. Build mappings
    stoi, itos, encoded_data = create_token_mappings(text)
    vocab_size = len(stoi)

    # 3. Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        encoded_data,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        split=args.split
    )

    # 4. Instantiate model
    model = Transformer(
        vocab_size=vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_length,
        dropout=args.dropout
    ).to(device)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # (Optional) Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args))

    # 6. Training loop
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch+1})

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 7. Save the trained model
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer on text data.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length for training.")
    parser.add_argument("--model_dim", type=int, default=256, help="Dimension of the transformer model.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--split", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--save_path", type=str, default="", help="Where to save the model after training.")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm", help="W&B Project name.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
