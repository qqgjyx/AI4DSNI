"""Training module for AI4DSNI using PyTorch Lightning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchmetrics import Accuracy, F1Score, MetricCollection

from .models import BaseEncoder, MultiTaskDecoder, create_encoder, create_decoder

logger = logging.getLogger(__name__)


class DSNIModule(pl.LightningModule):
    """PyTorch Lightning module for Deep-Sea Niche Identification.
    
    Combines encoder and decoder for multi-task classification.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: MultiTaskDecoder,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        scheduler_config: Optional[Dict] = None,
        warmup_epochs: int = 5,
        tasks: Optional[List[str]] = None,
    ):
        """Initialize the module.
        
        Args:
            encoder: Sequence encoder.
            decoder: Multi-task decoder.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            optimizer: Optimizer type ('adam', 'adamw', 'sgd').
            scheduler: Scheduler type ('cosine', 'step', 'plateau', 'none').
            scheduler_config: Additional scheduler configuration.
            warmup_epochs: Number of warmup epochs.
            tasks: List of task names.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        
        self.encoder = encoder
        self.decoder = decoder
        self.tasks = tasks or list(decoder.heads.keys())
        
        # Create metrics for each task
        self._create_metrics()
        
    def _create_metrics(self):
        """Create metrics for each task."""
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        
        for task in self.tasks:
            num_classes = self.decoder.task_configs[task]["num_classes"]
            
            # Training metrics
            self.train_metrics[task] = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            }, prefix=f"train/{task}_")
            
            # Validation metrics
            self.val_metrics[task] = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            }, prefix=f"val/{task}_")
            
            # Test metrics
            self.test_metrics[task] = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            }, prefix=f"test/{task}_")
    
    def forward(
        self, 
        sequence: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            sequence: Input sequence tensor (batch, seq_len).
            mask: Attention mask (batch, seq_len).
            
        Returns:
            Dict of task logits.
        """
        embeddings = self.encoder(sequence, mask)
        logits = self.decoder(embeddings, self.tasks)
        return logits
    
    def _shared_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Shared step for train/val/test.
        
        Args:
            batch: Batch dict with 'sequence', 'mask', 'labels'.
            stage: One of 'train', 'val', 'test'.
            
        Returns:
            Tuple of (total_loss, logits_dict).
        """
        sequence = batch["sequence"]
        mask = batch["mask"]
        labels = batch["labels"]
        
        # Forward pass
        logits = self(sequence, mask)
        
        # Compute loss
        total_loss, task_losses = self.decoder.compute_loss(logits, labels)
        
        # Log losses
        self.log(f"{stage}/loss", total_loss, prog_bar=True, sync_dist=True)
        for task, loss in task_losses.items():
            self.log(f"{stage}/{task}_loss", loss, sync_dist=True)
        
        # Update metrics
        metrics_dict = getattr(self, f"{stage}_metrics")
        for task in self.tasks:
            if task in logits and task in labels:
                task_labels = labels[task]
                # Only compute metrics for valid labels (not -100)
                valid_mask = task_labels != -100
                if valid_mask.any():
                    preds = logits[task][valid_mask].argmax(dim=-1)
                    targets = task_labels[valid_mask]
                    metrics_dict[task].update(preds, targets)
        
        return total_loss, logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, _ = self._shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, _ = self._shared_step(batch, "val")
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, _ = self._shared_step(batch, "test")
        return loss
    
    def on_train_epoch_end(self):
        """Log training metrics at end of epoch."""
        for task in self.tasks:
            metrics = self.train_metrics[task].compute()
            self.log_dict(metrics, sync_dist=True)
            self.train_metrics[task].reset()
    
    def on_validation_epoch_end(self):
        """Log validation metrics at end of epoch."""
        for task in self.tasks:
            metrics = self.val_metrics[task].compute()
            self.log_dict(metrics, sync_dist=True)
            self.val_metrics[task].reset()
    
    def on_test_epoch_end(self):
        """Log test metrics at end of epoch."""
        for task in self.tasks:
            metrics = self.test_metrics[task].compute()
            self.log_dict(metrics, sync_dist=True)
            self.test_metrics[task].reset()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Create optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        if self.hparams.optimizer == "adam":
            optimizer = Adam(
                params, 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = AdamW(
                params, 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = SGD(
                params, 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Create scheduler
        scheduler_config = self.hparams.scheduler_config or {}
        
        if self.hparams.scheduler == "none":
            return optimizer
        
        elif self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("t_max", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        
        elif self.hparams.scheduler == "step":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        
        elif self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=scheduler_config.get("patience", 5),
                factor=scheduler_config.get("factor", 0.5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")


def train(cfg: Dict[str, Any]) -> pl.Trainer:
    """Main training function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration dict.
        
    Returns:
        Trained Lightning Trainer.
    """
    from .data import (
        load_fasta, 
        get_splits, 
        create_dataloaders, 
        SequenceEncoder
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Extract configs
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})
    hw_cfg = cfg.get("hardware", {})
    paths_cfg = cfg.get("paths", {})
    
    # Load data
    logger.info("Loading data...")
    fasta_path = data_cfg.get("fasta_path")
    metadata_path = data_cfg.get("metadata_path")
    
    if fasta_path and Path(fasta_path).exists():
        import pandas as pd
        ids, sequences = load_fasta(fasta_path)
        metadata = pd.read_csv(metadata_path)
    else:
        logger.warning("Data files not found. Creating dummy data for testing.")
        sequences, metadata = _create_dummy_data()
    
    # Create encoder
    seq_encoder = SequenceEncoder(max_len=data_cfg.get("max_seq_len", 1500))
    
    # Split data
    logger.info("Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = get_splits(
        sequences=sequences,
        metadata=metadata,
        train_ratio=data_cfg.get("train_ratio", 0.70),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        split_by=data_cfg.get("split_by", "genus"),
        seed=data_cfg.get("seed", 42),
        encoder=seq_encoder,
        max_seq_len=data_cfg.get("max_seq_len", 1500),
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    
    # Create model
    logger.info(f"Creating model with encoder: {model_cfg.get('encoder_type', 'flat')}")
    encoder = create_encoder(
        encoder_type=model_cfg.get("encoder_type", "flat"),
        config=model_cfg,
    )
    
    decoder = create_decoder(
        input_dim=encoder.output_dim,
        config=model_cfg,
    )
    
    # Create Lightning module
    module = DSNIModule(
        encoder=encoder,
        decoder=decoder,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        optimizer=train_cfg.get("optimizer", "adamw"),
        scheduler=train_cfg.get("scheduler", "cosine"),
        scheduler_config=train_cfg.get(train_cfg.get("scheduler", "cosine"), {}),
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
    )
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    ckpt_cfg = train_cfg.get("checkpoint", {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=paths_cfg.get("checkpoints", "outputs/checkpoints"),
        filename="{epoch}-{val/loss:.4f}",
        monitor=ckpt_cfg.get("monitor", "val/loss"),
        mode=ckpt_cfg.get("mode", "min"),
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        monitor=es_cfg.get("monitor", "val/loss"),
        patience=es_cfg.get("patience", 15),
        mode=es_cfg.get("mode", "min"),
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir=paths_cfg.get("logs", "outputs/logs"),
        name=log_cfg.get("project_name", "ai4dsni"),
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 100),
        accelerator=hw_cfg.get("accelerator", "auto"),
        devices=hw_cfg.get("devices", 1),
        precision=hw_cfg.get("precision", 32),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=log_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(module, train_loader, val_loader)
    
    # Test
    logger.info("Running test evaluation...")
    if test_loader is not None:
        trainer.test(module, test_loader, ckpt_path="best")
    
    return trainer


def _create_dummy_data(n_samples: int = 100) -> Tuple[List[str], "pd.DataFrame"]:
    """Create dummy data for testing when real data is not available."""
    import pandas as pd
    import random
    
    # Generate random sequences
    nucleotides = ["A", "C", "G", "T"]
    sequences = []
    for _ in range(n_samples):
        seq_len = random.randint(500, 1500)
        seq = "".join(random.choices(nucleotides, k=seq_len))
        sequences.append(seq)
    
    # Generate random metadata
    metadata = pd.DataFrame({
        "id": [f"seq_{i}" for i in range(n_samples)],
        "genus": [f"genus_{i % 10}" for i in range(n_samples)],
        "temperature": random.choices(["cold", "mesophilic", "thermophilic"], k=n_samples),
        "ph": random.choices(["acidic", "neutral", "alkaline"], k=n_samples),
        "oxygen": random.choices(["aerobic", "anaerobic"], k=n_samples),
        "media": random.choices(["minimal", "rich", "defined", "complex"], k=n_samples),
    })
    
    return sequences, metadata
