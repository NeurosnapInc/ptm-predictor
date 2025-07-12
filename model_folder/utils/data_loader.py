import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np

class PTMDataset(Dataset):
    def __init__(self, csv_file, alphabet, batch_converter):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            alphabet: ESM alphabet object
            batch_converter: ESM batch converter
        """
        self.data = pd.read_csv(csv_file)
        self.alphabet = alphabet
        self.batch_converter = batch_converter
        
        # Convert string representation of lists to actual lists
        self.data['positions'] = self.data['positions'].apply(ast.literal_eval)
        
        # Get number of PTM types from the first row
        self.num_ptm_types = len(self.data['positions'].iloc[0])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        positions_list = row['positions']
        
        # Create label matrix (seq_length, num_ptm_types)
        seq_length = len(sequence)
        labels = torch.zeros(seq_length, self.num_ptm_types)
        
        # Fill in the labels
        for ptm_idx, positions in enumerate(positions_list):
            if positions:
                for pos in positions:
                    # Convert 1-indexed to 0-indexed
                    labels[pos - 1, ptm_idx] = 1
        
        return {
            'sequence': sequence,
            'labels': labels,
            'seq_length': seq_length
        }

def collate_fn(batch, batch_converter):
    """Custom collate function to handle batch conversion"""
    sequences = [(f"protein_{i}", item['sequence']) for i, item in enumerate(batch)]
    labels = [item['labels'] for item in batch]
    seq_lengths = [item['seq_length'] for item in batch]
    
    # Use ESM batch converter
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    
    # Get max sequence length in batch (including special tokens)
    max_len_in_batch = batch_tokens.size(1)
    
    # Pad labels to match tokenized sequences
    # Note: ESM adds special tokens (<cls> at start, <eos> at end)
    padded_labels = []
    for label, seq_len in zip(labels, seq_lengths):
        # Create padded label tensor
        padded_label = torch.zeros(max_len_in_batch, label.size(1))
        # Copy original labels, accounting for <cls> token at position 0
        padded_label[1:seq_len+1] = label
        padded_labels.append(padded_label)
    
    # Stack all labels
    batch_labels_tensor = torch.stack(padded_labels)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    # ESM tokens: 1 = pad, 0 = <cls>, 2 = <eos>
    attention_mask = (batch_tokens >= 4).float()
    
    return {
        'tokens': batch_tokens,
        'labels': batch_labels_tensor,
        'attention_mask': attention_mask,
        'seq_lengths': torch.tensor(seq_lengths)
    }

def get_data_loaders(train_csv, val_csv, test_csv, alphabet, batch_converter, 
                     batch_size=4, num_workers=4):
    """Create data loaders for train, validation, and test sets"""
    
    # Create datasets
    train_dataset = PTMDataset(train_csv, alphabet, batch_converter)
    val_dataset = PTMDataset(val_csv, alphabet, batch_converter)
    test_dataset = PTMDataset(test_csv, alphabet, batch_converter)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, batch_converter),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, batch_converter),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, batch_converter),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
