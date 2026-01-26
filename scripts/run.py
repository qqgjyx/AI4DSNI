#!/usr/bin/env python
"""Entry point for AI4DSNI training with Hydra configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.
    
    Args:
        cfg: Hydra configuration.
    """
    # Print configuration
    logger.info("=" * 60)
    logger.info("AI4DSNI - Deep-Sea Niche Identification")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Convert to dict for training function
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Import and run training
    from src.train import train
    
    trainer = train(cfg_dict)
    
    logger.info("Training complete!")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
