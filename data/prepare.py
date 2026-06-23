import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class HeadlineDataset(Dataset):
    """
    Dataset that reads tokenized headlines and returns
    overlapping chunks of context_len tokens for training.
    """
    def __init__(self, text_file, tokenizer_file, context_len):
        self.context_len = context_len

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        # Load and tokenize entire text
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Encode full text into token IDs
        encoded = self.tokenizer.encode(text)
        self.data = torch.tensor(encoded.ids, dtype=torch.long)

        print(f"Total tokens: {len(self.data):,}")

    def __len__(self):
        # Each sample is context_len tokens + 1 target token
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        # Input: tokens at positions idx to idx+context_len
        x = self.data[idx : idx + self.context_len]
        # Target: tokens shifted by 1 (next token prediction)
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return x, y


def get_dataloader(text_file, tokenizer_file, context_len, 
                   batch_size, split=0.9):
    """
    Creates train and validation dataloaders.
    split=0.9 means 90% train, 10% validation.
    """
    dataset = HeadlineDataset(text_file, tokenizer_file, context_len)

    # Split into train and validation
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True  # faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    return train_loader, val_loader
