# src/generate.py

import argparse
import torch
import torch.nn.functional as F

from data_utils import read_text_file, create_token_mappings
from model import Transformer

def generate_text(model, prompt, stoi, itos, device, seq_length=64, max_new_tokens=100):
    """
    Generates text from the given model starting with `prompt`.
    """
    model.eval()
    tokens = [stoi[ch] for ch in prompt]  # Convert each character in prompt to its token id

    for _ in range(max_new_tokens):
        # Take only the last `seq_length` tokens (model was trained on fixed context length)
        idx_cond = torch.tensor(tokens[-seq_length:], device=device).unsqueeze(0)
        logits = model(idx_cond)
        # Get the logits for the last timestep
        logits = logits[:, -1, :]
        # Sample from the distribution (argmax for simplicity/ to experiment with other sample methods)
        next_token = torch.argmax(logits, dim=-1).item()
        tokens.append(next_token)

    return ''.join([itos[t] for t in tokens])

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained Transformer.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the saved model .pt file.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to text file (used for vocab).")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt to start generation.")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length to use in generation.")
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension (must match trained model).")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads (must match trained model).")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers (must match trained model).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout (must match trained model).")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of new tokens to generate.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild vocab from data_file
    text = read_text_file(args.data_file)
    stoi, itos, encoded_data = create_token_mappings(text)
    vocab_size = len(stoi)

    # Rebuild model with same hyperparams
    model = Transformer(
        vocab_size=vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_length,
        dropout=args.dropout
    ).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))

    # Generate
    output = generate_text(model, args.prompt, stoi, itos, device, args.seq_length, args.max_new_tokens)
    print("Generated text:\n")
    print(output)

if __name__ == "__main__":
    main()
