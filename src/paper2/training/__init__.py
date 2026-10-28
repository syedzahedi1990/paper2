"""Paper 2 training, with optional GPU dependencies imported only when needed."""
from .config import ARMS, LORA_MODULES, TrainingConfig, experiment_matrix, load_config
from .data import answer_reward, validate_rows

__all__ = ['ARMS', 'LORA_MODULES', 'TrainingConfig', 'experiment_matrix', 'load_config', 'answer_reward', 'validate_rows']
