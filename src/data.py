"""Data loading and preprocessing for AI4DSNI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


class SequenceEncoder:
    """Nucleotide to integer mapping for RNA/DNA sequences.
    
    Vocabulary:
        0: PAD (padding token)
        1: A (adenine)
        2: C (cytosine)
        3: G (guanine)
        4: T/U (thymine/uracil)
        5: N (unknown/ambiguous)
    """
    
    # Vocabulary mapping
    VOCAB = {
        "<PAD>": 0,
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "U": 4,  # Treat U same as T
        "N": 5,
    }
    VOCAB_SIZE = 6
    PAD_TOKEN = 0
    
    def __init__(self, max_len: int = 1500):
        """Initialize encoder.
        
        Args:
            max_len: Maximum sequence length. Sequences longer are truncated,
                    shorter are padded.
        """
        self.max_len = max_len
        self._vocab = self.VOCAB.copy()
        
    def encode(self, sequence: str) -> torch.Tensor:
        """Encode a nucleotide sequence to integer tensor.
        
        Args:
            sequence: Nucleotide sequence string (A, C, G, T/U, N).
            
        Returns:
            Integer tensor of shape (max_len,).
        """
        sequence = sequence.upper().strip()
        
        # Encode each nucleotide
        encoded = []
        for nt in sequence[:self.max_len]:
            encoded.append(self._vocab.get(nt, self._vocab["N"]))
        
        # Pad if necessary
        if len(encoded) < self.max_len:
            encoded.extend([self.PAD_TOKEN] * (self.max_len - len(encoded)))
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def decode(self, tensor: torch.Tensor) -> str:
        """Decode integer tensor back to sequence string.
        
        Args:
            tensor: Integer tensor of nucleotide indices.
            
        Returns:
            Nucleotide sequence string.
        """
        idx_to_nt = {v: k for k, v in self._vocab.items() if k != "U"}
        idx_to_nt[0] = ""  # Don't include PAD in decoded string
        
        sequence = "".join(idx_to_nt.get(int(idx), "N") for idx in tensor)
        return sequence.strip()
    
    def __call__(self, sequence: str) -> torch.Tensor:
        """Alias for encode()."""
        return self.encode(sequence)


class DSNIDataset(Dataset):
    """Deep-Sea Niche Identification dataset with multi-task labels.
    
    Supports lazy loading of sequences and provides labels for:
    - Temperature preference (cold/mesophilic/thermophilic)
    - pH tolerance (acidic/neutral/alkaline)
    - Oxygen requirement (aerobic/anaerobic)
    - Media preference (minimal/rich/defined/complex)
    """
    
    # Label mappings for each task
    LABEL_MAPS = {
        "temperature": {"cold": 0, "mesophilic": 1, "thermophilic": 2},
        "ph": {"acidic": 0, "neutral": 1, "alkaline": 2},
        "oxygen": {"aerobic": 0, "anaerobic": 1},
        "media": {"minimal": 0, "rich": 1, "defined": 2, "complex": 3},
    }
    
    def __init__(
        self,
        sequences: List[str],
        metadata: pd.DataFrame,
        encoder: Optional[SequenceEncoder] = None,
        max_seq_len: int = 1500,
        tasks: Optional[List[str]] = None,
    ):
        """Initialize dataset.
        
        Args:
            sequences: List of nucleotide sequences.
            metadata: DataFrame with columns for each task label.
            encoder: SequenceEncoder instance. Created if None.
            max_seq_len: Maximum sequence length for encoder.
            tasks: List of task names to include. Uses all if None.
        """
        self.sequences = sequences
        self.metadata = metadata.reset_index(drop=True)
        self.encoder = encoder or SequenceEncoder(max_len=max_seq_len)
        self.tasks = tasks or list(self.LABEL_MAPS.keys())
        
        # Validate
        assert len(sequences) == len(metadata), (
            f"Mismatch: {len(sequences)} sequences vs {len(metadata)} metadata rows"
        )
        
        # Cache for encoded sequences (lazy loading)
        self._cache: Dict[int, torch.Tensor] = {}
        self._use_cache = True
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get encoded sequence and multi-task labels.
        
        Returns:
            Dictionary with:
                - 'sequence': encoded sequence tensor (max_len,)
                - 'labels': dict of task_name -> label tensor
                - 'mask': attention mask (1 for real tokens, 0 for padding)
        """
        # Encode sequence (with caching)
        if self._use_cache and idx in self._cache:
            seq_tensor = self._cache[idx]
        else:
            seq_tensor = self.encoder(self.sequences[idx])
            if self._use_cache:
                self._cache[idx] = seq_tensor
        
        # Create attention mask
        mask = (seq_tensor != self.encoder.PAD_TOKEN).long()
        
        # Get labels for each task
        labels = {}
        for task in self.tasks:
            if task in self.metadata.columns:
                label_val = self.metadata.loc[idx, task]
                if pd.isna(label_val):
                    # Use -100 for missing labels (ignored in loss)
                    labels[task] = torch.tensor(-100, dtype=torch.long)
                elif isinstance(label_val, str):
                    label_map = self.LABEL_MAPS.get(task, {})
                    labels[task] = torch.tensor(
                        label_map.get(label_val.lower(), -100), 
                        dtype=torch.long
                    )
                else:
                    labels[task] = torch.tensor(int(label_val), dtype=torch.long)
            else:
                labels[task] = torch.tensor(-100, dtype=torch.long)
        
        return {
            "sequence": seq_tensor,
            "labels": labels,
            "mask": mask,
        }
    
    def disable_cache(self):
        """Disable sequence caching (useful for memory constraints)."""
        self._use_cache = False
        self._cache.clear()
        
    def enable_cache(self):
        """Enable sequence caching."""
        self._use_cache = True


def load_fasta(fasta_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Load sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file.
        
    Returns:
        Tuple of (sequence_ids, sequences).
    """
    fasta_path = Path(fasta_path)
    
    ids = []
    sequences = []
    current_id = None
    current_seq = []
    
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous sequence
                if current_id is not None:
                    ids.append(current_id)
                    sequences.append("".join(current_seq))
                # Start new sequence
                current_id = line[1:].split()[0]  # Get ID (first word after >)
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id is not None:
            ids.append(current_id)
            sequences.append("".join(current_seq))
    
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return ids, sequences


def get_splits(
    sequences: List[str],
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_by: str = "genus",
    seed: int = 42,
    encoder: Optional[SequenceEncoder] = None,
    max_seq_len: int = 1500,
    tasks: Optional[List[str]] = None,
) -> Tuple[DSNIDataset, DSNIDataset, DSNIDataset]:
    """Create train/val/test splits with genus-level stratification.
    
    Ensures that sequences from the same genus are not split across
    train/val/test to prevent data leakage.
    
    Args:
        sequences: List of nucleotide sequences.
        metadata: DataFrame with 'genus' column and task labels.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        split_by: Column name for group-based splitting (default: 'genus').
        seed: Random seed for reproducibility.
        encoder: Shared SequenceEncoder instance.
        max_seq_len: Maximum sequence length.
        tasks: List of task names.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )
    
    n_samples = len(sequences)
    metadata = metadata.reset_index(drop=True)
    
    # Get groups for stratified splitting
    if split_by in metadata.columns:
        groups = metadata[split_by].fillna("unknown").values
    else:
        logger.warning(f"Column '{split_by}' not found, using random split")
        groups = np.arange(n_samples)  # Each sample is its own group
    
    # First split: train vs (val + test)
    splitter1 = GroupShuffleSplit(
        n_splits=1, 
        test_size=val_ratio + test_ratio, 
        random_state=seed
    )
    train_idx, temp_idx = next(splitter1.split(np.arange(n_samples), groups=groups))
    
    # Second split: val vs test (from the temp set)
    temp_groups = groups[temp_idx]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    
    splitter2 = GroupShuffleSplit(
        n_splits=1, 
        test_size=relative_test_ratio, 
        random_state=seed
    )
    val_idx_rel, test_idx_rel = next(splitter2.split(np.arange(len(temp_idx)), groups=temp_groups))
    
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]
    
    logger.info(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create shared encoder
    encoder = encoder or SequenceEncoder(max_len=max_seq_len)
    
    # Create datasets
    def make_dataset(indices):
        return DSNIDataset(
            sequences=[sequences[i] for i in indices],
            metadata=metadata.iloc[indices].reset_index(drop=True),
            encoder=encoder,
            max_seq_len=max_seq_len,
            tasks=tasks,
        )
    
    train_dataset = make_dataset(train_idx)
    val_dataset = make_dataset(val_idx)
    test_dataset = make_dataset(test_idx)
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: DSNIDataset,
    val_dataset: DSNIDataset,
    test_dataset: Optional[DSNIDataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset (optional).
        batch_size: Batch size.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for GPU transfer.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    
    def collate_fn(batch):
        """Custom collate function for multi-task labels."""
        sequences = torch.stack([item["sequence"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        
        # Collate labels for each task
        labels = {}
        tasks = batch[0]["labels"].keys()
        for task in tasks:
            labels[task] = torch.stack([item["labels"][task] for item in batch])
        
        return {
            "sequence": sequences,
            "labels": labels,
            "mask": masks,
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    
    return train_loader, val_loader, test_loader
