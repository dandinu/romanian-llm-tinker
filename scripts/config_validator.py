#!/usr/bin/env python3
"""
Configuration validation using Pydantic.

This module provides validation for training configuration files
to catch errors early and provide helpful feedback.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration validation."""
    name: str = Field(..., description="Model name from HuggingFace")
    base_type: Literal["llama3", "llama2", "mistral", "gpt"] = Field(
        ..., description="Base model type for renderer"
    )
    context_length: int = Field(
        ..., gt=0, le=32768, description="Maximum context length"
    )

    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        if '/' not in v:
            raise ValueError(
                f"Model name should be in format 'organization/model', got: {v}"
            )
        return v


class LoRAConfig(BaseModel):
    """LoRA configuration validation."""
    rank: int = Field(..., gt=0, le=256, description="LoRA rank (r)")
    alpha: int = Field(..., gt=0, description="LoRA alpha scaling factor")
    dropout: float = Field(..., ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: str = Field(..., description="Target modules for LoRA")

    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v: int, info) -> int:
        """Warn if alpha is not typically 2*rank."""
        if 'rank' in info.data:
            rank = info.data['rank']
            if v != 2 * rank:
                logger.warning(
                    f"LoRA alpha ({v}) is not 2*rank ({2*rank}). "
                    f"This is unusual but may be intentional."
                )
        return v


class TrainingConfig(BaseModel):
    """Training configuration validation."""
    learning_rate: float = Field(
        ..., gt=0.0, lt=1.0, description="Learning rate"
    )
    warmup_steps: int = Field(..., ge=0, description="Warmup steps")
    lr_scheduler: Literal["constant", "linear", "cosine"] = Field(
        ..., description="Learning rate scheduler"
    )
    max_steps: int = Field(..., gt=0, description="Maximum training steps")
    eval_steps: int = Field(..., gt=0, description="Evaluation frequency")
    save_steps: int = Field(..., gt=0, description="Checkpoint save frequency")
    logging_steps: int = Field(..., gt=0, description="Logging frequency")
    batch_size: int = Field(..., ge=1, le=128, description="Batch size")
    gradient_accumulation_steps: int = Field(
        ..., ge=1, description="Gradient accumulation steps"
    )
    max_grad_norm: float = Field(
        ..., gt=0.0, description="Gradient clipping threshold"
    )
    max_seq_length: int = Field(
        ..., gt=0, le=32768, description="Maximum sequence length"
    )
    seed: int = Field(..., ge=0, description="Random seed")

    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Warn if learning rate is unusually high or low."""
        if v > 0.01:
            logger.warning(
                f"Learning rate {v} is unusually high (>0.01). "
                f"This may cause training instability."
            )
        if v < 1e-6:
            logger.warning(
                f"Learning rate {v} is very low (<1e-6). "
                f"Training may be very slow."
            )
        return v

    @model_validator(mode='after')
    def validate_step_relationships(self) -> 'TrainingConfig':
        """Validate relationships between different step parameters."""
        if self.eval_steps > self.max_steps:
            raise ValueError(
                f"eval_steps ({self.eval_steps}) must be <= max_steps ({self.max_steps})"
            )

        if self.save_steps > self.max_steps:
            raise ValueError(
                f"save_steps ({self.save_steps}) must be <= max_steps ({self.max_steps})"
            )

        if self.logging_steps > self.eval_steps:
            logger.warning(
                f"logging_steps ({self.logging_steps}) > eval_steps ({self.eval_steps}). "
                f"You may miss some evaluation results."
            )

        return self


class OptimizerConfig(BaseModel):
    """Optimizer configuration validation."""
    type: Literal["adamw", "adam", "sgd"] = Field(..., description="Optimizer type")
    weight_decay: float = Field(..., ge=0.0, le=1.0, description="Weight decay")
    beta1: float = Field(..., gt=0.0, lt=1.0, description="Adam beta1")
    beta2: float = Field(..., gt=0.0, lt=1.0, description="Adam beta2")
    epsilon: float = Field(..., gt=0.0, description="Adam epsilon")

    @field_validator('weight_decay')
    @classmethod
    def validate_weight_decay(cls, v: float) -> float:
        """Warn if weight decay is unusually high."""
        if v > 0.1:
            logger.warning(
                f"Weight decay {v} is quite high (>0.1). "
                f"This may over-regularize the model."
            )
        return v


class LossConfig(BaseModel):
    """Loss configuration validation."""
    type: Literal["cross_entropy"] = Field(..., description="Loss function type")
    ignore_index: int = Field(..., description="Index to ignore in loss")


class CheckpointConfig(BaseModel):
    """Checkpoint configuration validation."""
    save_total_limit: int = Field(..., ge=1, description="Max checkpoints to keep")
    save_strategy: Literal["steps", "epoch"] = Field(..., description="Save strategy")
    resume_from_checkpoint: Optional[str] = Field(None, description="Checkpoint to resume")


class EvaluationConfig(BaseModel):
    """Evaluation configuration validation."""
    strategy: Literal["steps", "epoch", "no"] = Field(..., description="Eval strategy")
    eval_accumulation_steps: int = Field(..., ge=1, description="Eval accumulation steps")
    per_device_eval_batch_size: int = Field(..., ge=1, description="Eval batch size")


class DataConfig(BaseModel):
    """Data configuration validation."""
    train_file: str = Field(..., description="Path to training file")
    validation_file: str = Field(..., description="Path to validation file")
    preprocessing_num_workers: int = Field(..., ge=1, description="Number of workers")


class RomanianConfig(BaseModel):
    """Romanian-specific configuration validation."""
    language_detection_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Language detection threshold"
    )
    remove_non_latin: bool = Field(..., description="Remove non-Latin characters")
    normalize_whitespace: bool = Field(..., description="Normalize whitespace")
    min_length: int = Field(..., gt=0, description="Minimum text length")
    max_length: int = Field(..., gt=0, description="Maximum text length")

    @model_validator(mode='after')
    def validate_length_bounds(self) -> 'RomanianConfig':
        """Validate min_length < max_length."""
        if self.min_length >= self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) must be < max_length ({self.max_length})"
            )
        return self


class WandBConfig(BaseModel):
    """Weights & Biases configuration validation."""
    enabled: bool = Field(..., description="Enable W&B logging")
    project: str = Field(..., description="W&B project name")
    entity: Optional[str] = Field(None, description="W&B entity/username")
    run_name: Optional[str] = Field(None, description="W&B run name")
    tags: List[str] = Field(default_factory=list, description="W&B tags")


class QuickTestConfig(BaseModel):
    """Quick test configuration validation."""
    enabled: bool = Field(..., description="Enable quick test mode")
    max_steps: int = Field(..., gt=0, description="Max steps for quick test")
    num_examples: int = Field(..., gt=0, description="Number of examples for quick test")


class ProductionConfig(BaseModel):
    """Production configuration validation."""
    enabled: bool = Field(..., description="Enable production mode")
    max_steps: int = Field(..., gt=0, description="Max steps for production")
    save_steps: int = Field(..., gt=0, description="Save frequency for production")
    eval_steps: int = Field(..., gt=0, description="Eval frequency for production")


class HyperparamsConfig(BaseModel):
    """Complete hyperparameters configuration validation."""
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    checkpointing: CheckpointConfig
    evaluation: EvaluationConfig
    data: DataConfig
    romanian: RomanianConfig
    wandb: WandBConfig
    quick_test: QuickTestConfig
    production: ProductionConfig

    @model_validator(mode='after')
    def validate_conflicting_modes(self) -> 'HyperparamsConfig':
        """Validate that quick_test and production are not both enabled."""
        if self.quick_test.enabled and self.production.enabled:
            raise ValueError(
                "Cannot enable both quick_test and production modes. "
                "Choose one or disable both for default mode."
            )
        return self

    @model_validator(mode='after')
    def validate_context_length_consistency(self) -> 'HyperparamsConfig':
        """Validate that max_seq_length <= context_length."""
        if self.training.max_seq_length > self.model.context_length:
            raise ValueError(
                f"training.max_seq_length ({self.training.max_seq_length}) "
                f"cannot exceed model.context_length ({self.model.context_length})"
            )
        return self


def validate_config(config_dict: dict) -> HyperparamsConfig:
    """
    Validate configuration dictionary.

    Args:
        config_dict: Configuration dictionary from YAML

    Returns:
        Validated HyperparamsConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        validated = HyperparamsConfig(**config_dict)
        logger.info("✓ Configuration validation passed")
        return validated
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {str(e)}")
        raise


if __name__ == '__main__':
    """Test config validation with example config."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / 'configs' / 'hyperparams.yaml'

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        try:
            validated = validate_config(config)
            print("✓ Configuration is valid!")
        except Exception as e:
            print(f"✗ Configuration validation failed:\n{e}")
    else:
        print(f"Config file not found: {config_path}")
