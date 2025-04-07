# src/data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    """
    A PyTorch Dataset that takes a list of integer tokens (data) and
    returns sequences of length seq_length for language modeling.
    """
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # x: tokens at [idx : idx+seq_length]
        # y: tokens at [idx+1 : idx+seq_length+1]
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y

def read_text_file(path):
    """
    Reads a text file from `path` and returns the text as a string.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def create_token_mappings(text):
    """
    Given the text, create character-to-index (stoi) and index-to-char (itos) mappings.
    Returns:
        stoi (dict): mapping from character to integer index
        itos (dict): mapping from integer index to character
        encoded_data (list): the entire text encoded as list of integer token IDs
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded_data = [stoi[ch] for ch in text]
    return stoi, itos, encoded_data

def create_dataloaders(encoded_data, seq_length=64, batch_size=64, split=0.8):
    """
    Splits the encoded_data into train/val, then creates DataLoaders for each.
    Returns:
        train_loader, val_loader
    """
    # Train/val split
    train_size = int(len(encoded_data) * split)
    train_data = encoded_data[:train_size]
    val_data = encoded_data[train_size:]

    # Create Dataset objects
    train_dataset = CharDataset(train_data, seq_length)
    val_dataset = CharDataset(val_data, seq_length)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader
