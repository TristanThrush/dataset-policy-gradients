import argparse
import getpass
import os

import jax
import jax.numpy as jnp
import numpy as np

from dataset_metagradients_jax.train_utils import get_config, setup_training

_SCRATCH = os.environ.get("LOCAL_FAST_STORAGE", f"/tmp/{getpass.getuser()}")


def create_configs(
    cache_dir=f"{_SCRATCH}/.jax_cache",
    checkpoint_dir=f"{_SCRATCH}/checkpoints",
    preset="small",
):
    batch_size = 16
    total_steps = 4
    config_exact_vjp = get_config(
        preset,
        seed=0,
        dtype="fp32",
        train_num_examples=batch_size * total_steps,
        val_num_examples=16,
        microbatch_size=8,
        grad_accumulation_steps=2,
        learning_rate=1e-4,
        eps_root=1e-2,
        jax_cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=False,  # Disable wandb for correctness tests
        # optimizer_type="sgd", # adam introduces a whole universe of numerical issues, so do basic tests with sgd
        optimizer_type="adamw_reparam",
        use_manual_vjp=False,
    )
    config_manual_vjp = get_config(
        preset,
        seed=0,
        dtype="fp32",
        train_num_examples=batch_size * total_steps,
        val_num_examples=16,
        microbatch_size=8,
        grad_accumulation_steps=2,
        learning_rate=1e-4,
        eps_root=1e-2,
        jax_cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=False,  # Disable wandb for correctness tests
        # optimizer_type="sgd", # adam introduces a whole universe of numerical issues, so do basic tests with sgd
        optimizer_type="adamw_reparam",
        use_manual_vjp=True,
    )
    return config_manual_vjp, config_exact_vjp


def test_metagrad_agreement_manual_vjp(
    cache_dir=f"{_SCRATCH}/.jax_cache",
    checkpoint_dir=f"{_SCRATCH}/checkpoints",
    preset="small",
):
    config_manual_vjp, config_exact_vjp = create_configs(
        cache_dir, checkpoint_dir, preset
    )
    components = setup_training(config_manual_vjp)
    trainer = components.trainer
    target_metric_fn = components.target_metric_fn

    # Run metagradient computation and get baseline model target metric
    with components.mesh:
        outputs = trainer.train(
            components.train_dataloader,
            target_metric_fn,
            use_wandb=False,
        )
        g_manual_vjp = np.array(outputs["final_data_weights"])

    components = setup_training(config_exact_vjp)
    trainer = components.trainer
    target_metric_fn = components.target_metric_fn

    with components.mesh:
        outputs = trainer.train(
            components.train_dataloader,
            target_metric_fn,
            use_wandb=False,
        )
        g_exact_vjp = np.array(outputs["final_data_weights"])

    print(f"Manual VJP: {g_manual_vjp}, Exact VJP: {g_exact_vjp}")
    cor_coef = np.corrcoef(g_manual_vjp, g_exact_vjp)[0, 1]
    print(f"Correlation: {cor_coef}")
    assert cor_coef > 0.9
    # Check that the two methods agree

    print("Manual VJP and exact VJP agree")


def test_metagrad_forward_model_agreement(
    cache_dir=f"{_SCRATCH}/.jax_cache",
    checkpoint_dir=f"{_SCRATCH}/checkpoints",
    preset="small",
):
    config_manual_vjp, config_exact_vjp = create_configs(
        cache_dir, checkpoint_dir, preset
    )
    target_metrics = []
    rng = np.random.RandomState(0)
    eps = 1.0
    for _ in range(20):
        delta = rng.randn(config_manual_vjp.train_num_examples)
        delta = delta / np.linalg.norm(delta) * eps
        data_weights = jnp.array(np.ones_like(delta) + delta)
        # clip any negative data weights to 0
        data_weights = jnp.maximum(data_weights, 0)
        print(f"Data weights: {data_weights}")

        components = setup_training(config_manual_vjp)
        trainer = components.trainer
        target_metric_fn = components.target_metric_fn

        # Run metagradient computation and get baseline model target metric
        with components.mesh:
            outputs = trainer.train(
                components.train_dataloader,
                target_metric_fn,
                with_metagrads=False,
                data_weights=data_weights,
                use_wandb=False,
            )
            actual_target_metric = float(np.array(target_metric_fn(outputs["final_model"])))
            target_metrics.append(actual_target_metric)

    print(f"Target metrics: {target_metrics}")
    # Check that target metrics are reasonably consistent
    std_target_metric = np.std(target_metrics)
    print(f"Standard deviation of target metrics: {std_target_metric}")
    assert std_target_metric < 1.0


def test_metagrad_data_weight_alignment(
    cache_dir=f"{_SCRATCH}/.jax_cache",
    checkpoint_dir=f"{_SCRATCH}/checkpoints",
    preset="small",
):
    NUM_PERTURBATIONS = 20
    config_manual_vjp, config_exact_vjp = create_configs(
        cache_dir, checkpoint_dir, preset
    )
    components = setup_training(config_manual_vjp)
    trainer = components.trainer
    target_metric_fn = components.target_metric_fn

    # Run metagradient computation and get baseline model target metric
    with components.mesh:
        outputs = trainer.train(
            components.train_dataloader,
            target_metric_fn,
            use_wandb=False,
        )
        baseline_model = outputs["final_model"]
        baseline_target_metric = float(np.array(target_metric_fn(baseline_model)))
        g = np.array(outputs["final_data_weights"])

    # Run with random data weights and compare to the linear approximation
    rng = np.random.RandomState(0)
    eps = 1.0
    actual_target_metrics = []
    predicted_target_metrics = []
    for _ in range(NUM_PERTURBATIONS):
        delta = rng.randn(*g.shape)
        delta = delta / np.linalg.norm(delta) * eps
        data_weights = jnp.array(np.ones_like(delta) + delta)
        # clip any negative data weights to 0
        data_weights = jnp.maximum(data_weights, 0)
        print(f"Data weights: {data_weights}")
        with components.mesh:
            # just to be sure, reset all the components.
            components = setup_training(config_manual_vjp)
            trainer = components.trainer
            target_metric_fn = components.target_metric_fn
            outputs = trainer.train(
                components.train_dataloader,
                target_metric_fn,
                with_metagrads=False,
                data_weights=data_weights,
                use_wandb=False,
            )
            actual_target_metric = float(np.array(target_metric_fn(outputs["final_model"])))
            predicted_target_metric = baseline_target_metric + float(np.dot(g, delta))
            actual_target_metrics.append(actual_target_metric)
            predicted_target_metrics.append(predicted_target_metric)
    # sort by actual target metric for ease of comparison
    keys = np.argsort(actual_target_metrics)
    sorted_actual_target_metric = np.array(actual_target_metrics)[keys]
    sorted_predicted_target_metric = np.array(predicted_target_metrics)[keys]
    print(f"Actual target metric: {sorted_actual_target_metric}, Predicted target metric: {sorted_predicted_target_metric}")
    # spearman correlation
    cor_coef = np.corrcoef(sorted_actual_target_metric, sorted_predicted_target_metric)[0, 1]
    print(f"Correlation: {cor_coef}")
    assert cor_coef > 0.7


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metagrad correctness tests")
    parser.add_argument(
        "--cache-dir",
        default=f"{_SCRATCH}/.jax_cache",
        help=f"JAX cache directory (default: {_SCRATCH}/.jax_cache)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=f"{_SCRATCH}/checkpoints",
        help=f"Model checkpoint directory (default: {_SCRATCH}/checkpoints)",
    )
    parser.add_argument(
        "--preset",
        default="small",
        help="Name of the preset model config to test (default: small). If no \
preset is found under this name, then the code will try to load a Hugging Face model with the \
name of the preset using EasyDeL",
    )
    args = parser.parse_args()

    # Configure JAX
    jax.config.update("jax_compilation_cache_dir", args.cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compiler_enable_remat_pass", False)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )
    jax.config.update("jax_explain_cache_misses", True)

    import sys

    try:
        test_metagrad_data_weight_alignment(
            args.cache_dir, args.checkpoint_dir, args.preset
        )
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    print("Test passed")
