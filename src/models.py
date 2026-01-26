"""Model architectures for AI4DSNI."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for sequence encoders.
    
    All encoders must implement forward() that returns embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 6,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode sequences to embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len).
            mask: Attention mask of shape (batch_size, seq_len).
            
        Returns:
            Embeddings tensor of shape (batch_size, hidden_dim).
        """
        pass
    
    @property
    def output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.hidden_dim


class FlatEncoder(BaseEncoder):
    """CNN-based encoder with multiple convolutional layers.
    
    Architecture:
        - Embedding layer
        - Multiple 1D conv layers with ReLU and BatchNorm
        - Adaptive pooling to fixed size
        - Final projection to hidden_dim
    """
    
    def __init__(
        self,
        vocab_size: int = 6,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_conv_layers: int = 4,
        kernel_sizes: Optional[List[int]] = None,
        channels: Optional[List[int]] = None,
        pool_type: str = "adaptive_avg",
    ):
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        
        self.num_conv_layers = num_conv_layers
        self.kernel_sizes = kernel_sizes or [7, 5, 3, 3]
        self.channels = channels or [64, 128, 256, 256]
        self.pool_type = pool_type
        
        # Ensure we have enough kernel sizes and channels
        while len(self.kernel_sizes) < num_conv_layers:
            self.kernel_sizes.append(3)
        while len(self.channels) < num_conv_layers:
            self.channels.append(self.channels[-1] if self.channels else 128)
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=0
        )
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = embedding_dim
        for i in range(num_conv_layers):
            out_channels = self.channels[i]
            kernel_size = self.kernel_sizes[i]
            padding = kernel_size // 2
            
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        # Pooling
        if pool_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        
        # Final projection
        self.fc = nn.Linear(self.channels[-1], hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through CNN encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len).
            mask: Attention mask (not used in CNN, but kept for API consistency).
            
        Returns:
            Embeddings of shape (batch_size, hidden_dim).
        """
        # Embed: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # Transpose for conv1d: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply conv layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Pool: (batch, channels, seq_len) -> (batch, channels, 1)
        x = self.pool(x)
        
        # Flatten: (batch, channels)
        x = x.squeeze(-1)
        
        # Project: (batch, hidden_dim)
        x = self.fc(x)
        
        return x


class VariabilityGatedEncoder(BaseEncoder):
    """Variability-gated encoder architecture.
    
    Placeholder for custom architecture that uses variability information
    to gate sequence features. Can be extended with specific implementation.
    
    Architecture:
        - Embedding layer
        - Variability-gated convolutional blocks
        - Attention-based pooling
        - Final projection
    """
    
    def __init__(
        self,
        vocab_size: int = 6,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 4,
        gating_hidden: int = 64,
    ):
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        
        self.num_layers = num_layers
        self.gating_hidden = gating_hidden
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=0
        )
        
        # Gated convolutional blocks
        self.conv_blocks = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        
        in_channels = embedding_dim
        out_channels = embedding_dim
        
        for i in range(num_layers):
            # Main convolution
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ))
            
            # Gating network (learns position-specific importance)
            self.gate_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, gating_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(gating_hidden, out_channels, kernel_size=1),
                nn.Sigmoid(),
            ))
            
            in_channels = out_channels
        
        # Attention-based pooling
        self.attention = nn.Sequential(
            nn.Linear(out_channels, gating_hidden),
            nn.Tanh(),
            nn.Linear(gating_hidden, 1),
        )
        
        # Final projection
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through variability-gated encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len).
            mask: Attention mask of shape (batch_size, seq_len).
            
        Returns:
            Embeddings of shape (batch_size, hidden_dim).
        """
        # Embed: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        x = self.dropout_layer(x)
        
        # Transpose for conv1d: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply gated conv blocks
        for conv_block, gate_layer in zip(self.conv_blocks, self.gate_layers):
            # Compute features and gates
            features = conv_block(x)
            gates = gate_layer(x)
            
            # Gated combination with residual
            x = features * gates + x
        
        # Transpose back: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Attention pooling
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        
        if mask is not None:
            # Mask out padding positions
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            attention_weights = attention_weights.masked_fill(mask == 0, float("-inf"))
        
        attention_weights = F.softmax(attention_weights, dim=1)
        x = (x * attention_weights).sum(dim=1)  # (batch, channels)
        
        # Project to hidden_dim
        x = self.fc(x)
        
        return x


class RNABertEncoder(BaseEncoder):
    """Encoder using pretrained RNA-BERT from multimolecule.
    
    Wraps the multimolecule/rnabert model for fine-tuning.
    """
    
    def __init__(
        self,
        vocab_size: int = 6,  # Not used, kept for API consistency
        embedding_dim: int = 128,  # Not used
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pretrained_model: str = "multimolecule/rnabert",
        freeze_layers: int = 0,
    ):
        # Note: vocab_size and embedding_dim are from base class but not used here
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        
        self.pretrained_model_name = pretrained_model
        self.freeze_layers = freeze_layers
        self._model_loaded = False
        
        # Placeholder for lazy loading
        self.bert = None
        self.tokenizer = None
        self._bert_hidden_size = 768  # Default, will be updated after loading
        
        # Projection layer (created after loading model)
        self.projection = None
        self.dropout_layer = nn.Dropout(dropout)
        
    def _load_model(self):
        """Lazy load the pretrained model."""
        if self._model_loaded:
            return
            
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading pretrained model: {self.pretrained_model_name}")
            self.bert = AutoModel.from_pretrained(self.pretrained_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            
            # Get actual hidden size
            self._bert_hidden_size = self.bert.config.hidden_size
            
            # Create projection layer
            self.projection = nn.Linear(self._bert_hidden_size, self.hidden_dim)
            
            # Freeze layers if specified
            if self.freeze_layers > 0:
                self._freeze_layers()
            
            self._model_loaded = True
            logger.info(f"Model loaded successfully. Hidden size: {self._bert_hidden_size}")
            
        except ImportError:
            raise ImportError(
                "transformers package required for RNABertEncoder. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _freeze_layers(self):
        """Freeze bottom layers of BERT."""
        if self.bert is None:
            return
            
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        logger.info(f"Froze {self.freeze_layers} encoder layers")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through RNA-BERT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len).
                Should be tokenized using the BERT tokenizer.
            mask: Attention mask of shape (batch_size, seq_len).
            
        Returns:
            Embeddings of shape (batch_size, hidden_dim).
        """
        self._load_model()
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=x,
            attention_mask=mask,
            return_dict=True,
        )
        
        # Use [CLS] token representation or pooled output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Use mean of last hidden states
            last_hidden = outputs.last_hidden_state
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = last_hidden.mean(dim=1)
        
        # Project to hidden_dim
        x = self.dropout_layer(pooled)
        x = self.projection(x)
        
        return x
    
    @property
    def output_dim(self) -> int:
        return self.hidden_dim


class MultiTaskDecoder(nn.Module):
    """Multi-task decoder with separate heads for each task.
    
    Supports:
        - Temperature classification (3 classes)
        - pH classification (3 classes)
        - Oxygen classification (2 classes)
        - Media classification (4 classes)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        task_configs: Optional[Dict[str, Dict]] = None,
        dropout: float = 0.1,
    ):
        """Initialize multi-task decoder.
        
        Args:
            input_dim: Input dimension from encoder.
            hidden_dims: List of hidden layer dimensions for shared layers.
            task_configs: Dict mapping task names to config dicts with:
                - num_classes: Number of output classes
                - class_weights: Optional tensor of class weights
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        
        # Default task configs
        self.task_configs = task_configs or {
            "temperature": {"num_classes": 3, "class_weights": None},
            "ph": {"num_classes": 3, "class_weights": None},
            "oxygen": {"num_classes": 2, "class_weights": None},
            "media": {"num_classes": 4, "class_weights": None},
        }
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, config in self.task_configs.items():
            num_classes = config["num_classes"]
            self.heads[task_name] = nn.Linear(prev_dim, num_classes)
        
        # Store class weights as buffers (not parameters)
        for task_name, config in self.task_configs.items():
            weights = config.get("class_weights")
            if weights is not None:
                self.register_buffer(
                    f"{task_name}_weights",
                    torch.tensor(weights, dtype=torch.float)
                )
    
    def forward(
        self, 
        x: torch.Tensor,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through decoder.
        
        Args:
            x: Input embeddings of shape (batch_size, input_dim).
            tasks: List of tasks to compute. Uses all if None.
            
        Returns:
            Dict mapping task names to logits (batch_size, num_classes).
        """
        tasks = tasks or list(self.heads.keys())
        
        # Shared layers
        shared_out = self.shared(x)
        
        # Task-specific heads
        outputs = {}
        for task in tasks:
            if task in self.heads:
                outputs[task] = self.heads[task](shared_out)
        
        return outputs
    
    def compute_loss(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss.
        
        Args:
            logits: Dict of task logits.
            labels: Dict of task labels.
            reduction: Loss reduction method.
            
        Returns:
            Tuple of (total_loss, dict of per-task losses).
        """
        task_losses = {}
        total_loss = 0.0
        
        for task_name, task_logits in logits.items():
            if task_name not in labels:
                continue
                
            task_labels = labels[task_name]
            
            # Get class weights if available
            weight = getattr(self, f"{task_name}_weights", None)
            
            # Compute cross-entropy loss (ignores -100 labels)
            loss = F.cross_entropy(
                task_logits, 
                task_labels, 
                weight=weight,
                ignore_index=-100,
                reduction=reduction,
            )
            
            task_losses[task_name] = loss
            total_loss = total_loss + loss
        
        return total_loss, task_losses
    
    def get_class_weights(self, task_name: str) -> Optional[torch.Tensor]:
        """Get class weights for a task."""
        return getattr(self, f"{task_name}_weights", None)


def create_encoder(
    encoder_type: str,
    config: Dict,
) -> BaseEncoder:
    """Factory function to create encoder based on type.
    
    Args:
        encoder_type: One of 'flat', 'variability_gated', 'rnabert'.
        config: Configuration dict for the encoder.
        
    Returns:
        Initialized encoder instance.
    """
    # Common args
    common_args = {
        "vocab_size": config.get("vocab_size", 6),
        "embedding_dim": config.get("embedding_dim", 128),
        "hidden_dim": config.get("hidden_dim", 256),
        "dropout": config.get("dropout", 0.1),
    }
    
    if encoder_type == "flat":
        flat_config = config.get("flat", {})
        return FlatEncoder(
            **common_args,
            num_conv_layers=flat_config.get("num_conv_layers", 4),
            kernel_sizes=flat_config.get("kernel_sizes"),
            channels=flat_config.get("channels"),
            pool_type=flat_config.get("pool_type", "adaptive_avg"),
        )
    
    elif encoder_type == "variability_gated":
        vg_config = config.get("variability_gated", {})
        return VariabilityGatedEncoder(
            **common_args,
            num_layers=vg_config.get("num_layers", 4),
            gating_hidden=vg_config.get("gating_hidden", 64),
        )
    
    elif encoder_type == "rnabert":
        rnabert_config = config.get("rnabert", {})
        return RNABertEncoder(
            **common_args,
            pretrained_model=rnabert_config.get("pretrained_model", "multimolecule/rnabert"),
            freeze_layers=rnabert_config.get("freeze_layers", 0),
        )
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def create_decoder(
    input_dim: int,
    config: Dict,
) -> MultiTaskDecoder:
    """Factory function to create decoder.
    
    Args:
        input_dim: Input dimension from encoder.
        config: Configuration dict for the decoder.
        
    Returns:
        Initialized decoder instance.
    """
    decoder_config = config.get("decoder", {})
    
    # Build task configs
    task_configs = {}
    tasks = decoder_config.get("tasks", {})
    for task_name, task_config in tasks.items():
        task_configs[task_name] = {
            "num_classes": task_config.get("num_classes", 2),
            "class_weights": task_config.get("class_weights"),
        }
    
    return MultiTaskDecoder(
        input_dim=input_dim,
        hidden_dims=decoder_config.get("hidden_dims", [128, 64]),
        task_configs=task_configs if task_configs else None,
        dropout=config.get("dropout", 0.1),
    )
