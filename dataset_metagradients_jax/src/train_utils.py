import os
os.environ["JAX_CAPTURED_CONSTANTS_REPORT_FRAMES"] = "-1"

from dataclasses import dataclass
from typing import Any, Optional, Dict
import getpass
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from transformers import AutoTokenizer
import flax.nnx as nnx
import wandb
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

from .model import create_sharded_model, create_pretrained_easydel_sharded_model
from .memory_efficient_trainer import MemoryEfficientTrainer
from .optim import adamw_reparam
import grain

@dataclass
class TrainConfig:
    dtype: str
    tokenizer_name: str
    tokenizer_kwargs: Dict[str, Any]
    total_batches: int
    sequence_length: int
    microbatch_size: int
    grad_accumulation_steps: int
    easydel_pretrained_override: Optional[str]
    seed: int
    optimizer_type: str
    warmup_steps: Optional[int]
    learning_rate: float
    weight_decay: float
    eps_root: float
    eps: float
    b1: float
    b2: float
    use_manual_vjp: bool
    jax_cache_dir: str
    checkpoint_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_name: Optional[str]
    wandb_mode: str
    wandb_tags: list
    val_sample_fraction: Optional[float] = None
    target_metric_type: Optional[str] = "val_language_modeling"
    naive_metagrad_server: Optional[bool] = False
    cross_group_batching: Optional[bool] = True
    wandb_prefix: Optional[str] = ""
    replay_data_path: Optional[str] = None
    replay_ratio: Optional[float] = 0.1
    # nanoGPT-from-scratch model architecture + base dataset/loop params. These are
    # irrelevant when a pretrained easydel model and an externally-built target are used
    # (as in the verl-server configs), so they default to the standard values here and can
    # be omitted from those configs. Configs that train from scratch still set them.
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    mlp_ratio: float = 4.0
    dataset_name: str = 'roneneldan/TinyStories'
    train_split: str = 'train'
    val_split: str = 'validation'
    train_num_examples: Optional[int] = None
    val_num_examples: int = 256
    num_epochs: int = 1
    shuffle: bool = False
    # Build the val/target benchmark in-process from an lm-eval task (instead of
    # loading a pre-tokenized dataset). These mirror the args of the
    # old convert_eleuther_task_to_metagrads_target.py script.
    eleuther_benchmark_lm_eval_tasks: Optional[str] = None
    eleuther_benchmark_split: Optional[str] = None
    eleuther_benchmark_random_sample_ratio: Optional[float] = None
    eleuther_benchmark_add_data_source: Optional[bool] = False
    eleuther_benchmark_shuffle: Optional[bool] = False

@dataclass
class TrainingComponents:
    mesh: Any
    tokenizer: Any
    train_dataloader: Any
    val_dataloader: Any
    model: nnx.Module
    optimizer: optax.GradientTransformation
    trainer: Any
    target_metric_fn: Optional[Any]


# nanoGPT-from-scratch architecture presets (dim / n_layers / n_heads / mlp_ratio).
# Mirrors the conf/model/*.yaml presets; used by get_config for programmatic
# (non-Hydra) config construction in tests and scripts.
_MODEL_PRESETS = {
    "small": dict(dim=128, n_layers=2, n_heads=2, mlp_ratio=2.0),
    "gpt2": dict(dim=768, n_layers=12, n_heads=12, mlp_ratio=4.0),
    "large": dict(dim=2048, n_layers=16, n_heads=32, mlp_ratio=4.0),
}


def get_config(name: str, **overrides: Any) -> TrainConfig:
    """Build a TrainConfig programmatically, without Hydra.

    `name` is either a built-in architecture preset (one of _MODEL_PRESETS:
    small / gpt2 / large) to train a nanoGPT from scratch, or a Hugging Face
    model id, in which case the model is loaded via easydel
    (easydel_pretrained_override=name, tokenizer_name=name) and the nanoGPT arch
    fields are left at their preset defaults (easydel supplies the real shapes).

    Any TrainConfig field may be passed as a keyword to override the defaults
    below (which mirror conf/config.yaml). total_batches, if not given, is
    derived from train_num_examples / (microbatch_size * grad_accumulation_steps).

    This is the non-Hydra counterpart to load_hydra_config, for tests/scripts
    that construct configs in code.
    """
    scratch = os.environ.get("LOCAL_FAST_STORAGE", f"/tmp/{getpass.getuser()}")
    cfg = dict(
        dtype="fp32",
        tokenizer_name="gpt2",
        tokenizer_kwargs={},
        sequence_length=128,
        microbatch_size=128,
        grad_accumulation_steps=4,
        easydel_pretrained_override=None,
        seed=42,
        optimizer_type="adamw",
        warmup_steps=None,
        learning_rate=1e-3,
        weight_decay=0.1,
        eps_root=1e-8,
        eps=0.01,
        b1=0.9,
        b2=0.95,
        use_manual_vjp=True,
        jax_cache_dir=f"{scratch}/.jax_cache",
        checkpoint_dir=f"{scratch}/checkpoints",
        use_wandb=False,
        wandb_project="dataset-metagradients-jax",
        wandb_entity=None,
        wandb_name=None,
        wandb_mode="online",
        wandb_tags=[],
    )
    if name in _MODEL_PRESETS:
        cfg.update(_MODEL_PRESETS[name])
    else:
        # Treat `name` as a Hugging Face model id loaded through easydel.
        cfg["easydel_pretrained_override"] = name
        cfg["tokenizer_name"] = name
    cfg.update(overrides)
    if "total_batches" not in cfg:
        denom = cfg["microbatch_size"] * cfg["grad_accumulation_steps"]
        tne = cfg.get("train_num_examples")
        cfg["total_batches"] = max(1, tne // denom) if tne else 1
    return TrainConfig(**cfg)



def create_batch_dataset(
    dataset: Any,
    batch_size: int,
    sequence_length: int,
    grad_accum_size: int,
    shuffle: bool = True,
    drop_remainder: bool = True,
    do_tokenize: bool = True,
    pad_token_id: int = 0,
    eos_token_id: int = -100,
    seed: int = 42,
) -> Any:
    """Create a batched dataset of tokenized sequences.
    
    Args:
        dataset: Input dataset
        batch_size: Size of each microbatch
        sequence_length: Length of each sequence
        grad_accum_size: Number of microbatches to accumulate
        shuffle: Whether to shuffle the dataset
        drop_remainder: Whether to drop incomplete batches
        pad_token_id: Token ID to use for padding
        seed: Random seed for shuffling
        
    Returns:
        Batched dataset where each batch contains grad_accum_size * batch_size examples,
        shaped as [batch_size * grad_accum_size, sequence_length]
    """
    grain.config.update("py_debug_mode",True)
    data_source = grain.MapDataset.source(dataset)


    def pad_and_truncate_to_length(array, length, pad_token_id):
        if len(array) < length:
            array = np.pad(
                array,
                (0, length - len(array)),
                constant_values=pad_token_id,
            )
        return array[:length]

    def tokenize_and_chunk(index, example):
        tokens = np.array(example["input_ids"], dtype=np.int32)
        tokens = pad_and_truncate_to_length(tokens, sequence_length + 1, pad_token_id)
        inputs = np.array(tokens[:sequence_length])
        targets = np.array(tokens[1 : sequence_length + 1])
        return {"input_ids": inputs, "labels": targets, "index": index}
    
    def pad_and_truncate_only(index, example):
        input_ids = pad_and_truncate_to_length(np.array(example["input_ids"], dtype=np.int32), sequence_length, pad_token_id)
        labels = pad_and_truncate_to_length(np.array(example["labels"], dtype=np.int32), sequence_length, pad_token_id)
        # hack just for the code that uses -100 as a termination token
        labels[labels == -100] = eos_token_id
        # treat negative values as padding
        input_ids[input_ids < 0] = pad_token_id
        labels[labels < 0] = pad_token_id
        return {"input_ids": input_ids, "labels": labels, "index": index}


    if do_tokenize:
        ds = data_source.map_with_index(tokenize_and_chunk)
    else:
        ds = data_source.map_with_index(pad_and_truncate_only)

    if shuffle:
        ds = ds.shuffle(seed=seed)
    
    # Create batches that contain grad_accum_size microbatches
    return ds.batch(batch_size * grad_accum_size, drop_remainder=drop_remainder)


def setup_training(config: TrainConfig) -> TrainingComponents:
    """Builds model, data loaders, optimizer, and trainer from configuration."""
    # Initialize wandb if enabled.
    #
    # Train and val metrics go to SEPARATE wandb runs (suffix "_val"). This is required
    # because the two series live on different remapped_step scales -- val trains the target
    # on the whole val set, so its M (inner batches per GRPO step) differs from train's. In a
    # single run, wandb's one monotonically-increasing global `step` cannot represent both
    # scales, so it silently drops whichever series is on the smaller scale (e.g. 67 showed
    # val-only, lambada showed train-only). Two runs give each series its own monotonic step,
    # so both are kept.
    #
    # It also makes requeue/resume correct: each run logs with step=remapped_step (monotonic
    # within that run), so when verl resumes from the last *saved* checkpoint and re-runs the
    # unsaved GRPO steps, those re-done logs carry a remapped_step below the resumed run's
    # current step and wandb's monotonic-step rule cleanly drops the duplicates -- leaving a
    # single-valued curve instead of a backtrack. resume="allow" with a fixed id makes each
    # run resume in place on requeue. reinit="create_new" keeps BOTH runs live concurrently in
    # this one process (so we log via the run objects below, not the global wandb.log).
    #
    # IMPORTANT: create the runs ONCE per process and cache them on `config` (which persists
    # across the metagrad server's per-GRPO-step __init__/setup_training re-invocations, like
    # the eleuther/replay caches do). The non-naive server reinitializes every step, and
    # re-running wandb.init each time -- now twice, with reinit="create_new" forcing a real
    # re-init rather than a no-op reuse -- hammers the wandb backend and eventually hits a 90s
    # init timeout; that exception aborts __init__, leaving the checkpointer closed, which then
    # breaks every subsequent step ("cannot schedule new futures after shutdown"). Initializing
    # once avoids all per-step wandb.init calls.
    wandb_run_train = None
    wandb_run_val = None
    if config.use_wandb:
        _cached = getattr(config, "_wandb_runs", None)
        if _cached is not None:
            wandb_run_train, wandb_run_val = _cached
        else:
            # Run id is keyed on SLURM_JOB_ID (stable across a requeue, unique across separate
            # launches) while the display NAME stays clean (config.wandb_name). Using the bare
            # name as the id is fragile: once an experiment is re-run, you must delete the old
            # run, but a *deleted* id poisons wandb's resume="allow" -- it hangs ~90s on the
            # trashed id and the server crashes at startup. A per-launch id avoids that entirely
            # (a fresh launch's id never existed, so resume="allow" just creates it; a requeue
            # reuses the same SLURM_JOB_ID, so it resumes the same run). Falls back to the bare
            # name off-Slurm (no requeue there, so a stable id isn't needed).
            _job_uid = os.environ.get("SLURM_JOB_ID")
            _id_base = f"{config.wandb_name}_{_job_uid}" if _job_uid else config.wandb_name
            _wandb_common = dict(
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.wandb_name,   # group the train/val runs together in the UI
                mode=config.wandb_mode,
                tags=config.wandb_tags,
                config=vars(config),
                reinit="create_new",
                # Raise init_timeout from the 90s default so a transient wandb-backend slow
                # window doesn't time out and crash server startup.
                settings=wandb.Settings(init_timeout=600),
            )
            wandb_run_train = wandb.init(name=config.wandb_name, id=_id_base,
                                         resume="allow", job_type="train", **_wandb_common)
            wandb_run_val = wandb.init(name=f"{config.wandb_name}_val", id=f"{_id_base}_val",
                                       resume="allow", job_type="val", **_wandb_common)
            # Plot the target metric against the GRPO step (one tick per policy update): comparable
            # across the train/val runs and across experiments (convert to synthetic-examples-seen
            # via grpo_step * train_batch_size * rollout.n). All other (inner-loop) metrics keep the
            # reserved step = remapped_step as their x-axis.
            for _r in (wandb_run_train, wandb_run_val):
                _r.define_metric("grpo_step")
                _r.define_metric("target_metric*", step_metric="grpo_step")
            config._wandb_runs = (wandb_run_train, wandb_run_val)
    
    # Configure JAX
    jax.config.update("jax_compilation_cache_dir", config.jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compiler_enable_remat_pass", False)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    jax.config.update("jax_explain_cache_misses", True)

    num_devices = len(jax.devices())
    mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape(num_devices, 1),
        axis_names=("data", "model"),
    )

    print(f"[setup_training] num_devices={num_devices}, mesh axis names={mesh.axis_names}")

    # Calculate total examples needed
    examples_per_batch = config.microbatch_size * config.grad_accumulation_steps
    if config.train_num_examples is not None:
        # Round up to nearest complete batch
        num_complete_batches = (config.train_num_examples + examples_per_batch - 1) // examples_per_batch
        num_train = num_complete_batches * examples_per_batch
        if num_train > config.train_num_examples:
            print(f"[setup_training] Rounding up train examples from {config.train_num_examples} to {num_train} to ensure complete batches")
    else:
        num_train = examples_per_batch * config.num_epochs * config.total_batches

    print(f"[setup_training] num_train examples to load: {num_train} with {config.microbatch_size} microbatch_size")

    train_split = f"{config.train_split}[:{num_train}]"
    val_split = f"{config.val_split}[:{config.val_num_examples}]"

    raw_train = load_dataset(config.dataset_name, split=train_split)
    raw_val = load_dataset(config.dataset_name, split=val_split)
    print(f"[setup_training] Loaded raw_train examples: {len(raw_train)}, raw_val examples: {len(raw_val)}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        **config.tokenizer_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.sequence_length,
            return_tensors="np",
        )

    tokenized_train = raw_train.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw_train.column_names,
    )
    tokenized_val = raw_val.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw_val.column_names,
    )
    print(f"[setup_training] Tokenized train and val datasets: train length {len(tokenized_train)}, val length {len(tokenized_val)}")

    assert config.microbatch_size % num_devices == 0

    train_dataloader = create_batch_dataset(
        dataset=tokenized_train,
        batch_size=config.microbatch_size,
        sequence_length=config.sequence_length,
        grad_accum_size=config.grad_accumulation_steps,
        shuffle=config.shuffle,
        pad_token_id=tokenizer.pad_token_id,
    )
    val_dataloader = create_batch_dataset(
        dataset=tokenized_val,
        batch_size=config.microbatch_size,
        sequence_length=config.sequence_length,
        grad_accum_size=1,  # Use 1 for validation since we don't accumulate gradients
        shuffle=config.shuffle,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"[setup_training] Created train and val dataloaders with microbatch_size={config.microbatch_size}")

    model_kwargs = {
        "vocab_size": len(tokenizer.vocab),
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "max_seq_len": config.sequence_length,
        "mlp_ratio": config.mlp_ratio,
    }
    model_dtype = jnp.bfloat16 if config.dtype == 'bf16' else jnp.float32

    with mesh:
        if config.easydel_pretrained_override is not None:
            model = create_pretrained_easydel_sharded_model(config.easydel_pretrained_override, dtype=model_dtype)
            print(f"[setup_training] EasyDeL pretrained {config.easydel_pretrained_override} model created")
        else:
            model = create_sharded_model(
                seed=config.seed,
                dtype=model_dtype,
                **model_kwargs,
            )
            print(f"[setup_training] Model created with dim={config.dim}, n_layers={config.n_layers}, n_heads={config.n_heads}")

    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
        lr_schedule = optax.warmup_constant_schedule(
            init_value = config.learning_rate/warmup_steps,
            peak_value = config.learning_rate,
            warmup_steps = warmup_steps,
        )
    else:
        lr_schedule = config.learning_rate

    core_optimizer = None
    if config.optimizer_type == 'sgd':
        core_optimizer = optax.sgd(learning_rate=config.learning_rate)
    elif config.optimizer_type == 'adamw':
        core_optimizer = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            eps_root=config.eps_root,
            eps=config.eps,
            b2=config.b2,
            b1=config.b1,
        )
    elif config.optimizer_type == 'adamw_reparam':
        core_optimizer = adamw_reparam(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            eps_root=config.eps_root,
            eps=config.eps,
            b2=config.b2,
            b1=config.b1,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

    optimizer = optax.chain(
        core_optimizer,
    )


    
    

    print(f"[setup_training] Optimizer set up with type: {config.optimizer_type}")
    with mesh:
        trainer = MemoryEfficientTrainer(
            model=model,
            optimizer=optimizer,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=config.microbatch_size,
            grad_accum_size=config.grad_accumulation_steps,
            checkpoint_dir=config.checkpoint_dir,
            debug=True,
            use_manual_vjp=config.use_manual_vjp,
        )
    print(f"[setup_training] Trainer instantiated: MemoryEfficientTrainer")

    # target metric functions should take in a param and return a scalar. this is an example
    eval_batch = next(iter(val_dataloader))
    def target_metric_fn(model):
        inputs = jnp.array(eval_batch["input_ids"])
        targets = jnp.array(eval_batch["labels"])
        return -trainer._eval_step(model, inputs, targets)

    trainer.wandb_prefix = config.wandb_prefix
    # Train/val each log to their own run (see wandb.init above); pick by val_mode at log time.
    trainer.wandb_run_train = wandb_run_train
    trainer.wandb_run_val = wandb_run_val

    return TrainingComponents(
        mesh=mesh,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        target_metric_fn=target_metric_fn,
    ) 
