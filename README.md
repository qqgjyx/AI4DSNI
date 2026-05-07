# DSNI

**DSNI (DSMZ and NIH Integrated)** - A multi-task deep learning framework for predicting bacterial cultivation conditions from 16S rRNA sequences.

## Overview

DSNI uses deep learning to classify microbial sequences based on their cultivation requirements:

- **Temperature**: psychrophile (<20°C), mesophile (20-45°C), thermophile (>45°C)
- **pH**: acidophile (<6), neutrophile (6-8), alkaliphile (>8)
- **Oxygen**: aerobe, facultative, microaerophile, anaerobe
- **Media**: 42 standardized cultivation media categories

## Project Structure

```
AI4DSNI/
├── configs/
│   └── config.yaml          # Unified Hydra configuration
├── src/
│   ├── __init__.py
│   ├── data.py              # Dataset, preprocessing, splits
│   ├── models.py            # Encoders + decoder architectures
│   └── train.py             # Lightning module + training
├── scripts/
│   └── run.py               # Training entry point
├── tests/
│   └── test_core.py         # Unit tests
├── notebooks/
│   └── analysis.ipynb       # Analysis and visualization
├── data/                     # Released subset (7,470 strains)
│   ├── sequences.fasta
│   ├── metadata.csv
│   └── media_compositions.json
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Clone the repository


# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Data

The `data/` directory contains the released data:
- `sequences.fasta` — 16S rRNA sequences
- `metadata.csv` — per-strain table with sequence, taxonomy, and the four cultivation labels (temperature, pH, oxygen, media)
- `media_compositions.json` — growth-medium recipes

### 2. Train a Model

```bash
# Train with default configuration
python scripts/run.py

# Override configuration parameters
python scripts/run.py model.encoder_type=variability_gated training.max_epochs=50

# Use different encoder
python scripts/run.py model.encoder_type=rnabert
```

### 3. Run Tests

```bash
pytest tests/ -v
```

## Model Architectures

### Encoders

1. **FlatEncoder** (default): CNN-based encoder with multiple convolutional layers
   - 4 conv layers with batch normalization
   - Adaptive pooling for fixed-size output

2. **VariabilityGatedEncoder**: Variability-gated Transformer encoder
   - Biological feature-conditioned gating (V-region tags, GC content, Shannon entropy)
   - 6-layer Transformer with 8 attention heads
   - Gate modulates embeddings before Transformer encoding
   - [CLS] token pooling for sequence representation

3. **RNABertEncoder**: Pretrained RNA-BERT wrapper
   - Uses `multimolecule/rnabert` from HuggingFace
   - Supports layer freezing for fine-tuning

### Decoder

**MultiTaskDecoder**: Shared layers with task-specific heads
- Configurable hidden dimensions
- Task loss weights: media=1.0, temperature=0.5, pH=0.5, oxygen=0.75

## Configuration

All configuration is managed via Hydra. Key sections in `configs/config.yaml`:

```yaml
data:
  max_seq_len: 1500
  batch_size: 32
  train_ratio: 0.70

model:
  encoder_type: flat  # flat, variability_gated, rnabert
  hidden_dim: 256
  dropout: 0.1

training:
  max_epochs: 100
  learning_rate: 1e-4
  optimizer: adamw
  scheduler: cosine
```

## Data Splitting

The `get_splits()` function implements genus-level stratification to prevent data leakage:
- Sequences from the same genus are kept together in train/val/test
- Default split: 70% train, 15% validation, 15% test

## Outputs

Training outputs are saved to `outputs/`:
- `checkpoints/`: Model checkpoints (best and last)
- `logs/`: TensorBoard logs

View training progress:
```bash
tensorboard --logdir outputs/logs
```

## API Usage

```python
from src.data import SequenceEncoder, DSNIDataset, get_splits
from src.models import create_encoder, create_decoder
from src.train import DSNIModule

# Create encoder and decoder
encoder = create_encoder("flat", {"hidden_dim": 256})
decoder = create_decoder(encoder.output_dim, {"decoder": {"tasks": {...}}})

# Create Lightning module
module = DSNIModule(encoder, decoder, learning_rate=1e-4)

# Encode a sequence
seq_encoder = SequenceEncoder(max_len=1500)
encoded = seq_encoder("ACGTACGT...")
```

## License

MIT License
