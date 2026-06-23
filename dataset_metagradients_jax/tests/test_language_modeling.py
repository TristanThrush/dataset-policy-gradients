"""Basic language modeling test using TinyStories dataset with Hydra config."""
import os
os.environ["JAX_CAPTURED_CONSTANTS_REPORT_FRAMES"] = "-1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import argparse
import time
from dataset_metagradients_jax.config import load_hydra_config
from dataset_metagradients_jax.train_utils import setup_training

def test_language_modeling_basic(train_config):
    """Test basic language modeling training loop."""

    print(f"\nStarting basic language modeling test from Hydra config")
    
    # Use the already converted config
    config = train_config

    print(f"Using config: dtype={config.dtype}, dim={config.dim}, "
          f"n_layers={config.n_layers}, total_batches={config.total_batches}")
    components = setup_training(config)
    mesh = components.mesh
    dataloader = components.train_dataloader
    trainer = components.trainer

    # Run training for 1 epoch
    print("\nStarting training...")
    start_time = time.time()
    with mesh:
        trainer.train(
            train_dataloader=dataloader,
            with_metagrads=False,
            use_wandb=False,  # smoke test: skip the server-coupled wandb logging path
        )
        duration = time.time() - start_time
        print(f"Training completed in {duration:.3f}s")

def main() -> None:
    # Parse test-specific arguments with argparse
    parser = argparse.ArgumentParser(description='Run basic language modeling test using Hydra config')
    parser.add_argument('--config-path', default="dataset_metagradients_jax/tests/conf/config.yaml",
                      help='Config path, repo-root-relative (default: tests/conf/config.yaml)')

    # Parse known args to separate test args from Hydra overrides
    test_args, hydra_args = parser.parse_known_args()

    # Load config using helper function
    cfg, train_config = load_hydra_config(test_args.config_path, hydra_args)

    test_language_modeling_basic(
        train_config=train_config,
    )


if __name__ == "__main__":
    main()
