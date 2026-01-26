"""Core tests for DSNI (DSMZ and NIH Integrated) components."""

from __future__ import annotations

import pytest
import torch
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SequenceEncoder, DSNIDataset, get_splits
from src.models import (
    FlatEncoder,
    VariabilityGatedEncoder,
    MultiTaskDecoder,
    create_encoder,
    create_decoder,
)
from src.train import DSNIModule


class TestSequenceEncoder:
    """Tests for SequenceEncoder."""
    
    def test_encode_basic(self):
        """Test basic encoding functionality."""
        encoder = SequenceEncoder(max_len=10)
        seq = "ACGT"
        encoded = encoder.encode(seq)
        
        assert encoded.shape == (10,)
        assert encoded.dtype == torch.long
        # A=1, C=2, G=3, T=4, PAD=0
        assert encoded[:4].tolist() == [1, 2, 3, 4]
        assert encoded[4:].tolist() == [0, 0, 0, 0, 0, 0]  # Padding
    
    def test_encode_truncation(self):
        """Test that long sequences are truncated."""
        encoder = SequenceEncoder(max_len=5)
        seq = "ACGTACGT"
        encoded = encoder.encode(seq)
        
        assert encoded.shape == (5,)
        assert encoded.tolist() == [1, 2, 3, 4, 1]  # ACGTA
    
    def test_encode_unknown_nucleotide(self):
        """Test handling of unknown nucleotides."""
        encoder = SequenceEncoder(max_len=5)
        seq = "ANXYZ"  # X, Y, Z should map to N
        encoded = encoder.encode(seq)
        
        assert encoded[0] == 1  # A
        assert encoded[1] == 5  # N
        assert encoded[2] == 5  # Unknown -> N
    
    def test_encode_case_insensitive(self):
        """Test case insensitivity."""
        encoder = SequenceEncoder(max_len=4)
        upper = encoder.encode("ACGT")
        lower = encoder.encode("acgt")
        
        assert torch.equal(upper, lower)
    
    def test_decode(self):
        """Test decoding functionality."""
        encoder = SequenceEncoder(max_len=10)
        seq = "ACGT"
        encoded = encoder.encode(seq)
        decoded = encoder.decode(encoded)
        
        assert decoded[:4] == "ACGT"
    
    def test_callable(self):
        """Test that encoder is callable."""
        encoder = SequenceEncoder(max_len=10)
        result = encoder("ACGT")
        assert result.shape == (10,)


class TestDSNIDataset:
    """Tests for DSNIDataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing with paper's label names."""
        sequences = ["ACGT" * 10, "GCTA" * 10, "AAAA" * 10]
        metadata = pd.DataFrame({
            "temperature": ["psychrophile", "mesophile", "thermophile"],
            "ph": ["acidophile", "neutrophile", "alkaliphile"],
            "oxygen": ["aerobe", "facultative", "microaerophile"],
            "media": [0, 1, 2],  # Integer labels for 42 classes
        })
        return sequences, metadata

    def test_dataset_length(self, sample_data):
        """Test dataset length."""
        sequences, metadata = sample_data
        dataset = DSNIDataset(sequences, metadata, max_seq_len=50)

        assert len(dataset) == 3

    def test_dataset_getitem(self, sample_data):
        """Test getting items from dataset."""
        sequences, metadata = sample_data
        dataset = DSNIDataset(sequences, metadata, max_seq_len=50)

        item = dataset[0]

        assert "sequence" in item
        assert "labels" in item
        assert "mask" in item
        assert item["sequence"].shape == (50,)
        assert item["mask"].shape == (50,)

        # Check labels (using paper's terminology)
        assert item["labels"]["temperature"] == 0  # psychrophile
        assert item["labels"]["ph"] == 0  # acidophile
        assert item["labels"]["oxygen"] == 0  # aerobe
        assert item["labels"]["media"] == 0  # media class 0

    def test_dataset_missing_labels(self):
        """Test handling of missing labels."""
        sequences = ["ACGT" * 10]
        metadata = pd.DataFrame({
            "temperature": [None],
            "ph": ["acidophile"],
        })
        dataset = DSNIDataset(sequences, metadata, max_seq_len=50)

        item = dataset[0]

        # Missing temperature should be -100
        assert item["labels"]["temperature"] == -100
        assert item["labels"]["ph"] == 0


class TestGetSplits:
    """Tests for get_splits function."""

    @pytest.fixture
    def sample_data(self):
        """Create larger sample data for split testing with paper's labels."""
        n_samples = 100
        sequences = ["ACGT" * 25 for _ in range(n_samples)]
        metadata = pd.DataFrame({
            "genus": [f"genus_{i % 10}" for i in range(n_samples)],
            "temperature": ["mesophile"] * n_samples,
            "ph": ["neutrophile"] * n_samples,
            "oxygen": ["facultative"] * n_samples,
            "media": [0] * n_samples,  # Integer label
        })
        return sequences, metadata
    
    def test_split_ratios(self, sample_data):
        """Test that split ratios are approximately correct."""
        sequences, metadata = sample_data
        train, val, test = get_splits(
            sequences, metadata,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        
        total = len(train) + len(val) + len(test)
        assert total == 100
        
        # Allow some tolerance due to group-based splitting
        assert 50 <= len(train) <= 80
        assert 5 <= len(val) <= 25
        assert 5 <= len(test) <= 25
    
    def test_shared_encoder(self, sample_data):
        """Test that splits share the same encoder."""
        sequences, metadata = sample_data
        encoder = SequenceEncoder(max_len=100)
        train, val, test = get_splits(
            sequences, metadata,
            encoder=encoder,
            seed=42,
        )
        
        assert train.encoder is encoder
        assert val.encoder is encoder
        assert test.encoder is encoder


class TestFlatEncoder:
    """Tests for FlatEncoder."""
    
    def test_forward_shape(self):
        """Test output shape of FlatEncoder."""
        encoder = FlatEncoder(
            vocab_size=6,
            embedding_dim=32,
            hidden_dim=64,
            num_conv_layers=2,
        )
        
        x = torch.randint(0, 6, (4, 100))  # batch=4, seq_len=100
        output = encoder(x)
        
        assert output.shape == (4, 64)
    
    def test_output_dim_property(self):
        """Test output_dim property."""
        encoder = FlatEncoder(hidden_dim=128)
        assert encoder.output_dim == 128


class TestVariabilityGatedEncoder:
    """Tests for VariabilityGatedEncoder (Transformer + biological gating)."""

    def test_forward_shape(self):
        """Test output shape."""
        encoder = VariabilityGatedEncoder(
            vocab_size=6,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            max_seq_len=200,
        )

        x = torch.randint(0, 6, (4, 100))
        mask = torch.ones(4, 100)
        output = encoder(x, mask)

        assert output.shape == (4, 64)

    def test_with_padding_mask(self):
        """Test handling of padding mask."""
        encoder = VariabilityGatedEncoder(
            vocab_size=6,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=100,
        )

        x = torch.randint(0, 6, (2, 50))
        mask = torch.ones(2, 50)
        mask[0, 25:] = 0  # Pad second half of first sequence

        output = encoder(x, mask)
        assert output.shape == (2, 64)

    def test_v_region_mask(self):
        """Test V-region mask creation."""
        encoder = VariabilityGatedEncoder(
            vocab_size=6,
            hidden_dim=64,
            max_seq_len=1500,
        )

        # V-regions should be marked as 1
        assert encoder.v_region_mask.shape == (1, 1500)
        # V1 region (69-99) should be marked
        assert encoder.v_region_mask[0, 80] == 1.0
        # Conserved region (before V1) should be 0
        assert encoder.v_region_mask[0, 50] == 0.0

    def test_biological_features(self):
        """Test that biological features (GC, entropy) are computed."""
        encoder = VariabilityGatedEncoder(
            vocab_size=6,
            hidden_dim=64,
            num_layers=1,
            max_seq_len=100,
        )

        # Create sequence with known GC content
        x = torch.tensor([[2, 3, 2, 3, 2, 3, 2, 3, 2, 3]])  # All GC (C=2, G=3)
        gc = encoder._compute_local_gc_content(x)
        # Should have high GC content
        assert gc[0, 5] > 0.8  # Center of window should be ~100% GC


class TestMultiTaskDecoder:
    """Tests for MultiTaskDecoder."""

    def test_forward_all_tasks(self):
        """Test forward with all tasks."""
        decoder = MultiTaskDecoder(
            input_dim=64,
            hidden_dims=[32],
            task_configs={
                "temperature": {"num_classes": 3},
                "ph": {"num_classes": 3},
                "oxygen": {"num_classes": 4},  # 4 classes per paper
            },
        )

        x = torch.randn(4, 64)
        outputs = decoder(x)

        assert "temperature" in outputs
        assert "ph" in outputs
        assert "oxygen" in outputs
        assert outputs["temperature"].shape == (4, 3)
        assert outputs["ph"].shape == (4, 3)
        assert outputs["oxygen"].shape == (4, 4)
    
    def test_forward_specific_tasks(self):
        """Test forward with specific tasks."""
        decoder = MultiTaskDecoder(
            input_dim=64,
            task_configs={
                "temperature": {"num_classes": 3},
                "ph": {"num_classes": 3},
            },
        )
        
        x = torch.randn(4, 64)
        outputs = decoder(x, tasks=["temperature"])
        
        assert "temperature" in outputs
        assert "ph" not in outputs
    
    def test_compute_loss(self):
        """Test loss computation."""
        decoder = MultiTaskDecoder(
            input_dim=64,
            task_configs={
                "temperature": {"num_classes": 3},
                "ph": {"num_classes": 3},
            },
        )
        
        x = torch.randn(4, 64)
        logits = decoder(x)
        
        labels = {
            "temperature": torch.randint(0, 3, (4,)),
            "ph": torch.randint(0, 3, (4,)),
        }
        
        total_loss, task_losses = decoder.compute_loss(logits, labels)
        
        assert total_loss.ndim == 0  # Scalar
        assert "temperature" in task_losses
        assert "ph" in task_losses
    
    def test_compute_loss_with_ignore_index(self):
        """Test loss computation with ignored labels."""
        decoder = MultiTaskDecoder(
            input_dim=64,
            task_configs={
                "temperature": {"num_classes": 3},
            },
        )
        
        x = torch.randn(4, 64)
        logits = decoder(x)
        
        labels = {
            "temperature": torch.tensor([0, 1, -100, 2]),  # One ignored
        }
        
        total_loss, _ = decoder.compute_loss(logits, labels)
        assert total_loss.isfinite()


class TestCreateEncoder:
    """Tests for create_encoder factory function."""
    
    def test_create_flat_encoder(self):
        """Test creating FlatEncoder."""
        config = {
            "embedding_dim": 32,
            "hidden_dim": 64,
            "flat": {"num_conv_layers": 2},
        }
        encoder = create_encoder("flat", config)
        
        assert isinstance(encoder, FlatEncoder)
        assert encoder.output_dim == 64
    
    def test_create_variability_gated_encoder(self):
        """Test creating VariabilityGatedEncoder."""
        config = {
            "embedding_dim": 32,
            "hidden_dim": 64,
            "max_seq_len": 200,
            "variability_gated": {
                "num_layers": 2,
                "num_heads": 4,
                "ff_dim": 128,
            },
        }
        encoder = create_encoder("variability_gated", config)

        assert isinstance(encoder, VariabilityGatedEncoder)
    
    def test_invalid_encoder_type(self):
        """Test error for invalid encoder type."""
        with pytest.raises(ValueError):
            create_encoder("invalid", {})


class TestDSNIModule:
    """Tests for DSNIModule Lightning module."""
    
    @pytest.fixture
    def module(self):
        """Create a test module."""
        encoder = FlatEncoder(
            vocab_size=6,
            embedding_dim=32,
            hidden_dim=64,
            num_conv_layers=2,
        )
        decoder = MultiTaskDecoder(
            input_dim=64,
            hidden_dims=[32],
            task_configs={
                "temperature": {"num_classes": 3},
                "ph": {"num_classes": 3},
            },
        )
        return DSNIModule(
            encoder=encoder,
            decoder=decoder,
            learning_rate=1e-3,
        )
    
    def test_forward(self, module):
        """Test forward pass."""
        x = torch.randint(0, 6, (4, 100))
        mask = torch.ones(4, 100)
        
        outputs = module(x, mask)
        
        assert "temperature" in outputs
        assert "ph" in outputs
    
    def test_training_step(self, module):
        """Test training step."""
        batch = {
            "sequence": torch.randint(0, 6, (4, 100)),
            "mask": torch.ones(4, 100),
            "labels": {
                "temperature": torch.randint(0, 3, (4,)),
                "ph": torch.randint(0, 3, (4,)),
            },
        }
        
        loss = module.training_step(batch, 0)
        
        assert loss.ndim == 0
        assert loss.isfinite()
    
    def test_configure_optimizers_adamw(self, module):
        """Test optimizer configuration."""
        result = module.configure_optimizers()
        
        assert "optimizer" in result
        assert "lr_scheduler" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
