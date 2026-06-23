"""Simple Hydra configurator for dataset-metagradients-jax."""
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from pathlib import Path
import jax
from .train_utils import TrainConfig


def load_config(cfg: DictConfig) -> TrainConfig:
    """Convert Hydra DictConfig to TrainConfig with automatic type conversion.
    
    Args:
        cfg: Hydra DictConfig from composed configuration
        
    Returns:
        TrainConfig object with loaded configuration
    """
    # Convert to dict and handle missing fields with defaults
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Set defaults for missing fields
    if 'tokenizer_kwargs' not in config_dict:
        config_dict['tokenizer_kwargs'] = {}
    if 'wandb_tags' not in config_dict:
        config_dict['wandb_tags'] = []
    
    # Remove any Hydra-specific keys that shouldn't be in TrainConfig
    hydra_keys = {'defaults', '_target_', 'hydra'}
    config_dict = {k: v for k, v in config_dict.items() if k not in hydra_keys}
    
    # Create TrainConfig directly from the clean config dict
    return TrainConfig(**config_dict)


def load_hydra_config(config_path: str, overrides: list[str] = None, config_dir: Path = None) -> tuple[DictConfig, TrainConfig]:
    """Load Hydra config and convert to TrainConfig with JAX configuration.
    
    Args:
        config_file: Path to the config file
        overrides: List of Hydra override strings
        config_dir: Path to config directory. If None, defaults to conf/ relative to caller
        
    Returns:
        Tuple of (raw Hydra config, converted TrainConfig)
    """
    if overrides is None:
        overrides = []
    
    # Default config directory - relative to the repo root
    if config_dir is None and ("/" not in config_path and "\\" not in config_path):
        # Go up from <project>/src/config.py -> <project>/conf
        config_dir = Path(__file__).parent.parent / "conf"
    
    config_path_obj = Path(config_path)
    config_name = config_path_obj.stem
    config_dir = config_path_obj.parent if config_dir is None else config_dir
    print(f"Config path: {config_path_obj}")
    print(f"Config dir: {config_dir}")
    print(f"Config name: {config_name}")
    root_config_dir = Path(__file__).parent.parent.parent / config_dir # config.py is at <repo>/dataset_metagradients_jax/src/config.py - go up 3 dirs to reach the repo root (config paths are repo-root-relative)
    print(f"Root config dir: {root_config_dir}")
    
    with initialize_config_dir(config_dir=str(root_config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    
    # Configure JAX with settings from config
    jax.config.update("jax_compilation_cache_dir", cfg.jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compiler_enable_remat_pass", False)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    jax.config.update("jax_explain_cache_misses", True)
    
    # Convert to TrainConfig
    train_config = load_config(cfg)
    
    return cfg, train_config