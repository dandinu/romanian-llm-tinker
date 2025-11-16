#!/usr/bin/env python3
"""
Unit tests for configuration validation.

Tests cover:
- Model configuration validation
- LoRA configuration validation
- Training configuration validation
- Complete config validation
- Error cases and edge conditions
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from config_validator import (
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    OptimizerConfig,
    RomanianConfig,
    HyperparamsConfig,
    validate_config
)
from pydantic import ValidationError


class TestModelConfig:
    """Test model configuration validation."""

    def test_valid_model_config(self):
        """Test that valid model config is accepted."""
        config = ModelConfig(
            name="meta-llama/Llama-3.1-8B",
            base_type="llama3",
            context_length=8192
        )
        assert config.name == "meta-llama/Llama-3.1-8B"
        assert config.context_length == 8192

    def test_invalid_model_name_format(self):
        """Test that invalid model name format is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                name="invalid-name-without-slash",
                base_type="llama3",
                context_length=8192
            )
        assert "organization/model" in str(exc_info.value)

    def test_invalid_context_length_too_large(self):
        """Test that context length above limit is rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="meta-llama/Llama-3.1-8B",
                base_type="llama3",
                context_length=99999  # Too large
            )

    def test_invalid_context_length_zero(self):
        """Test that zero context length is rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="meta-llama/Llama-3.1-8B",
                base_type="llama3",
                context_length=0
            )

    def test_invalid_base_type(self):
        """Test that invalid base type is rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="meta-llama/Llama-3.1-8B",
                base_type="invalid_type",  # Not in allowed literals
                context_length=8192
            )


class TestLoRAConfig:
    """Test LoRA configuration validation."""

    def test_valid_lora_config(self):
        """Test that valid LoRA config is accepted."""
        config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules="all_linear_layers"
        )
        assert config.rank == 8
        assert config.alpha == 16

    def test_invalid_rank_zero(self):
        """Test that zero rank is rejected."""
        with pytest.raises(ValidationError):
            LoRAConfig(
                rank=0,
                alpha=16,
                dropout=0.05,
                target_modules="all_linear_layers"
            )

    def test_invalid_rank_too_large(self):
        """Test that rank above limit is rejected."""
        with pytest.raises(ValidationError):
            LoRAConfig(
                rank=500,  # > 256
                alpha=1000,
                dropout=0.05,
                target_modules="all_linear_layers"
            )

    def test_invalid_dropout_negative(self):
        """Test that negative dropout is rejected."""
        with pytest.raises(ValidationError):
            LoRAConfig(
                rank=8,
                alpha=16,
                dropout=-0.1,
                target_modules="all_linear_layers"
            )

    def test_invalid_dropout_too_large(self):
        """Test that dropout > 0.5 is rejected."""
        with pytest.raises(ValidationError):
            LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.9,  # > 0.5
                target_modules="all_linear_layers"
            )


class TestTrainingConfig:
    """Test training configuration validation."""

    def test_valid_training_config(self):
        """Test that valid training config is accepted."""
        config = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=100,
            lr_scheduler="cosine",
            max_steps=1000,
            eval_steps=50,
            save_steps=100,
            logging_steps=10,
            batch_size=4,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            max_seq_length=2048,
            seed=42
        )
        assert config.learning_rate == 1e-4
        assert config.max_steps == 1000

    def test_invalid_learning_rate_zero(self):
        """Test that zero learning rate is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                learning_rate=0.0,
                warmup_steps=100,
                lr_scheduler="cosine",
                max_steps=1000,
                eval_steps=50,
                save_steps=100,
                logging_steps=10,
                batch_size=4,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_seq_length=2048,
                seed=42
            )

    def test_invalid_learning_rate_too_large(self):
        """Test that learning rate >= 1.0 is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                learning_rate=1.5,
                warmup_steps=100,
                lr_scheduler="cosine",
                max_steps=1000,
                eval_steps=50,
                save_steps=100,
                logging_steps=10,
                batch_size=4,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_seq_length=2048,
                seed=42
            )

    def test_eval_steps_greater_than_max_steps(self):
        """Test that eval_steps > max_steps is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                learning_rate=1e-4,
                warmup_steps=100,
                lr_scheduler="cosine",
                max_steps=100,
                eval_steps=200,  # > max_steps
                save_steps=50,
                logging_steps=10,
                batch_size=4,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_seq_length=2048,
                seed=42
            )
        assert "eval_steps" in str(exc_info.value)

    def test_save_steps_greater_than_max_steps(self):
        """Test that save_steps > max_steps is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                learning_rate=1e-4,
                warmup_steps=100,
                lr_scheduler="cosine",
                max_steps=100,
                eval_steps=50,
                save_steps=200,  # > max_steps
                logging_steps=10,
                batch_size=4,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_seq_length=2048,
                seed=42
            )
        assert "save_steps" in str(exc_info.value)

    def test_invalid_batch_size_zero(self):
        """Test that zero batch size is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                learning_rate=1e-4,
                warmup_steps=100,
                lr_scheduler="cosine",
                max_steps=1000,
                eval_steps=50,
                save_steps=100,
                logging_steps=10,
                batch_size=0,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_seq_length=2048,
                seed=42
            )

    def test_invalid_lr_scheduler(self):
        """Test that invalid scheduler type is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                learning_rate=1e-4,
                warmup_steps=100,
                lr_scheduler="invalid_scheduler",
                max_steps=1000,
                eval_steps=50,
                save_steps=100,
                logging_steps=10,
                batch_size=4,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_seq_length=2048,
                seed=42
            )


class TestOptimizerConfig:
    """Test optimizer configuration validation."""

    def test_valid_optimizer_config(self):
        """Test that valid optimizer config is accepted."""
        config = OptimizerConfig(
            type="adamw",
            weight_decay=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        assert config.type == "adamw"
        assert config.weight_decay == 0.001

    def test_invalid_weight_decay_negative(self):
        """Test that negative weight decay is rejected."""
        with pytest.raises(ValidationError):
            OptimizerConfig(
                type="adamw",
                weight_decay=-0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            )

    def test_invalid_weight_decay_too_large(self):
        """Test that weight decay > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            OptimizerConfig(
                type="adamw",
                weight_decay=1.5,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            )

    def test_invalid_beta1_out_of_range(self):
        """Test that beta1 outside (0, 1) is rejected."""
        with pytest.raises(ValidationError):
            OptimizerConfig(
                type="adamw",
                weight_decay=0.001,
                beta1=1.5,  # > 1.0
                beta2=0.999,
                epsilon=1e-8
            )

    def test_invalid_optimizer_type(self):
        """Test that invalid optimizer type is rejected."""
        with pytest.raises(ValidationError):
            OptimizerConfig(
                type="invalid_optimizer",
                weight_decay=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            )


class TestRomanianConfig:
    """Test Romanian-specific configuration validation."""

    def test_valid_romanian_config(self):
        """Test that valid Romanian config is accepted."""
        config = RomanianConfig(
            language_detection_threshold=0.9,
            remove_non_latin=True,
            normalize_whitespace=True,
            min_length=10,
            max_length=4096
        )
        assert config.min_length == 10
        assert config.max_length == 4096

    def test_invalid_threshold_out_of_range(self):
        """Test that threshold outside [0, 1] is rejected."""
        with pytest.raises(ValidationError):
            RomanianConfig(
                language_detection_threshold=1.5,  # > 1.0
                remove_non_latin=True,
                normalize_whitespace=True,
                min_length=10,
                max_length=4096
            )

    def test_min_length_greater_than_max_length(self):
        """Test that min_length >= max_length is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RomanianConfig(
                language_detection_threshold=0.9,
                remove_non_latin=True,
                normalize_whitespace=True,
                min_length=5000,  # > max_length
                max_length=4096
            )
        assert "min_length" in str(exc_info.value)

    def test_min_length_equal_to_max_length(self):
        """Test that min_length == max_length is rejected."""
        with pytest.raises(ValidationError):
            RomanianConfig(
                language_detection_threshold=0.9,
                remove_non_latin=True,
                normalize_whitespace=True,
                min_length=100,
                max_length=100  # Equal to min
            )


class TestCompleteConfigValidation:
    """Test complete hyperparameters configuration validation."""

    def get_valid_config(self):
        """Get a valid configuration dictionary."""
        return {
            'model': {
                'name': 'meta-llama/Llama-3.1-8B',
                'base_type': 'llama3',
                'context_length': 8192
            },
            'lora': {
                'rank': 8,
                'alpha': 16,
                'dropout': 0.05,
                'target_modules': 'all_linear_layers'
            },
            'training': {
                'learning_rate': 1e-4,
                'warmup_steps': 100,
                'lr_scheduler': 'cosine',
                'max_steps': 1000,
                'eval_steps': 50,
                'save_steps': 100,
                'logging_steps': 10,
                'batch_size': 4,
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'max_seq_length': 2048,
                'seed': 42
            },
            'optimizer': {
                'type': 'adamw',
                'weight_decay': 0.001,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8
            },
            'loss': {
                'type': 'cross_entropy',
                'ignore_index': -100
            },
            'checkpointing': {
                'save_total_limit': 3,
                'save_strategy': 'steps',
                'resume_from_checkpoint': None
            },
            'evaluation': {
                'strategy': 'steps',
                'eval_accumulation_steps': 1,
                'per_device_eval_batch_size': 4
            },
            'data': {
                'train_file': 'data/splits/train.jsonl',
                'validation_file': 'data/splits/val.jsonl',
                'preprocessing_num_workers': 4
            },
            'romanian': {
                'language_detection_threshold': 0.9,
                'remove_non_latin': True,
                'normalize_whitespace': True,
                'min_length': 10,
                'max_length': 4096
            },
            'wandb': {
                'enabled': False,
                'project': 'romanian-llama-finetune',
                'entity': None,
                'run_name': None,
                'tags': ['romanian', 'llama-3.1']
            },
            'quick_test': {
                'enabled': False,
                'max_steps': 50,
                'num_examples': 100
            },
            'production': {
                'enabled': False,
                'max_steps': 5000,
                'save_steps': 250,
                'eval_steps': 100
            }
        }

    def test_valid_complete_config(self):
        """Test that valid complete config is accepted."""
        config = self.get_valid_config()
        validated = validate_config(config)
        assert validated.model.name == 'meta-llama/Llama-3.1-8B'
        assert validated.training.max_steps == 1000

    def test_max_seq_length_exceeds_context_length(self):
        """Test that max_seq_length > context_length is rejected."""
        config = self.get_valid_config()
        config['training']['max_seq_length'] = 10000  # > context_length
        with pytest.raises(ValidationError) as exc_info:
            validate_config(config)
        assert "max_seq_length" in str(exc_info.value)

    def test_both_quick_test_and_production_enabled(self):
        """Test that both modes enabled is rejected."""
        config = self.get_valid_config()
        config['quick_test']['enabled'] = True
        config['production']['enabled'] = True
        with pytest.raises(ValidationError) as exc_info:
            validate_config(config)
        assert "quick_test and production" in str(exc_info.value)

    def test_missing_required_field(self):
        """Test that missing required field is rejected."""
        config = self.get_valid_config()
        del config['model']['name']
        with pytest.raises(ValidationError):
            validate_config(config)

    def test_missing_entire_section(self):
        """Test that missing entire section is rejected."""
        config = self.get_valid_config()
        del config['training']
        with pytest.raises(ValidationError):
            validate_config(config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
