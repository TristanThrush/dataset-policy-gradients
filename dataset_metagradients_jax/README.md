# Overview of Dataset Metagradients JAX Code

This is the directory of the JAX **metagradient engine** that computes the metagradient
for training data weights. It is then served as a custom RL reward to the verl GRPO trainer over XML-RPC.

It's a standalone `uv` project (Python 3.12, jax 0.6.2, EasyDeL). The parent repo's
`custom_metagrad_reward.py` (running in the verl trainer env) is the RPC *client*; this
package is the *server* and implements the metagradient math.

## How it's used in the RL loop

`scripts/metagrad_server.py` boots a model (named the target model in our paper) and listens on port 29922. On each GRPO step
our verl reward function sends it the generated sequences (`load_data_and_run`); the server
runs a target model training loop on them (for potentially many training steps) and then returns the **metagradient of the target
metric w.r.t. the per-example data weights** as the reward. The parent repo's
`run_dpg_local.sh` launches this server alongside the verl trainer
(`METAGRAD_SERVER_CONFIG` points at one of the metagrad server YAMLs in
`../experiment_configs/` - it is in one of these YAMLs where one specifies the target model,
the target metric, and other parameters of target model training and evaluation).

## Layout

Modules live directly under `src/` and import as `dataset_metagradients_jax`.

- `model.py` — a **from-scratch** Llama-style transformer (`LlamaModel`,
  `create_sharded_model`) **and** the EasyDeL pretrained-model loader
  (`create_pretrained_easydel_sharded_model`). The from-scratch model is for a
  minimal/standalone version of this system and is used by most of the tests; **the RL
  experiments from our paper do not use it** — they set `easydel_pretrained_override` so
  `setup_training` loads a pretrained model (GPT-2 / Llama-3.2-Instruct) via EasyDeL instead. See
  the note at the top of `model.py`.
- `train_utils.py` — `TrainConfig`, `setup_training` (builds model + optimizer + trainer),
  and `get_config` (a programmatic, non-Hydra config builder for tests/scripts).
- `memory_efficient_trainer.py` — the memory-efficient forward/backward that computes the
  metagradient (rematerialized inner steps + VJP); this is where the reward is produced.
- `config.py` — Hydra loader (`load_hydra_config`) turning a server YAML into a `TrainConfig`.
- `eleuther_benchmark.py` — builds a tokenized target ("benchmark") dataset from an Eleuther lm-eval
  task in-process (used by `target_metric_type: val_language_modeling` runs, e.g. the LAMBADA/UUID tasks from the paper).
- `optim.py`, `checkpointing.py`, `utils.py` — optimizers (incl.
  `adamw_reparam`, which is only used in tests), disk checkpointing + misc helpers.
- `scripts/metagrad_server.py` — the XML-RPC server. `scripts/rick_roll.npy` is the
  rick_roll target image QR code.

## Target metric types

The server YAML config's `target_metric_type` selects the target metric:
- `val_language_modeling` — language modeling loss improvement on a validation/benchmark target (built via `eleuther_benchmark.py`).
- `rick_roll` / `sixseven` — match a fixed image pattern encoded in the lm-head weights.
- `l2_norm` — lower the L2 norm of the lm-head weights.

(`cross_group_batching` / `naive_metagrad_server` control how the generated batch is
grouped before the inner loop and whether metagrads are run; see the per-experiment server configs and the paper.)

## Setup & tests

```bash
uv sync   # build the env from pyproject.toml + uv.lock
```

Tests (need a GPU) live in `tests/` with their Hydra configs in `tests/conf/`:
- `test_language_modeling.py`, `test_metagrad_run.py` — exercise the from-scratch model
  end-to-end via `tests/conf/{config,metagrad_test}.yaml`.
- `test_metagrad_correctness.py` — checks the metagradient (with manual vs JAX VJP implementations, and with a
  linear data-weight approximation) on the from-scratch "small" preset LLM.
- `test_compare_easydel_and_hf_model.py` — checks the **EasyDeL** load against Hugging Face.

```bash
uv run python tests/test_metagrad_correctness.py
```
