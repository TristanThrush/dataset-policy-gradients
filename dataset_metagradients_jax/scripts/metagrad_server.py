"""Metagrad reward server (XML-RPC), queried by the verl GRPO trainer. Uses a Hydra config plus argparse for server args."""
import os
import shutil
import uuid
os.environ["JAX_CAPTURED_CONSTANTS_REPORT_FRAMES"] = "-1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.825"

import argparse
import sys
import jax
import jax.numpy as jnp
import numpy as np
import time
import json
import math
from dataclasses import asdict
from dataset_metagradients_jax.config import load_hydra_config
from dataset_metagradients_jax.train_utils import setup_training, create_batch_dataset
from dataset_metagradients_jax.eleuther_benchmark import build_eleuther_benchmark_dataset
from dataset_metagradients_jax.checkpointing import create_checkpointer
from datasets import load_dataset
import glob
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from concurrent.futures import ThreadPoolExecutor
import signal
import copy
import wandb
from datasets import concatenate_datasets

import orbax.checkpoint as ocp
import flax.nnx as nnx

TARGET_67 = jnp.array([
[1,1,1,1,-1,-1,-1],
[-1,-1,-1,1,1,1,-1],
[-1,1,1,1,1,1,-1],
[-1,-1,-1,1,1,1,-1],
[-1,1,-1,1,1,1,-1],
[-1,-1,-1,1,1,1,1],
], dtype=jnp.float32)

rick_roll = jnp.array(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rick_roll.npy')), dtype=jnp.float32)


class MetagradRewardRun:
    def __init__(self, metadata_save_dir, train_config):
        """Init state for the server - includes model and other components"""
        self.metadata_save_dir = metadata_save_dir
        self._bg = ThreadPoolExecutor(max_workers=1)
        self._prev_future = None

        # Store the already converted config
        self.config = train_config

        # NOTE: in the non-naive path this __init__ is re-run between GRPO steps --
        # load_data_and_run_inner calls self.shutdown() then self.__init__() to fully
        # reset model/optimizer state -- so the setup below runs on every reset there,
        # not just at startup. Full reinit (rather than the lighter reset_state) is used
        # there to reclaim as much device memory as possible between steps: it drops the
        # full reverse-mode VJP's retained buffers so we can fit the largest target model
        # possible. (The naive path resets via reset_state instead and runs this __init__
        # only once, at startup.)
        if not hasattr(self.config, "orig_checkpoint_dir"):
            self.config.orig_checkpoint_dir = self.config.checkpoint_dir
        # checkpoint_dir gets a fresh uuid every init. This is NOT to keep old
        # checkpoints around (they're rmtree'd on checkpointer close / reset_state /
        # shutdown) -- it isolates each checkpointer instance on its own path so a new
        # one can't collide with a previous one that may still be tearing down. orbax
        # saves/restores run async, and the naive (terminate_after_target_metric) path returns
        # with that I/O still in flight, deferring cleanup to a background reset_state;
        # a distinct path avoids the rmtree-vs-async-write race (and isolates concurrent
        # server processes sharing the same base dir). Tradeoff: a hard crash leaks the
        # uuid dir as an orphan. warmstart_checkpoint_dir instead has a fixed name, so
        # clear any stale copy left by a previous (possibly crashed) run before we
        # re-snapshot into it below.
        self.config.checkpoint_dir = self.config.orig_checkpoint_dir + "_" + str(uuid.uuid4())
        self.config.warmstart_checkpoint_dir = self.config.orig_checkpoint_dir + "_warmstart"
        if os.path.exists(self.config.warmstart_checkpoint_dir):
            shutil.rmtree(self.config.warmstart_checkpoint_dir)
        
        # Setup training components
        self.components = setup_training(self.config)
        self.mesh = self.components.mesh
        self.tokenizer = self.components.tokenizer
        self.trainer = self.components.trainer
        self.trainer.metadata_save_dir = metadata_save_dir

        # The server always trains a pretrained (easydel) target model.
        assert self.config.easydel_pretrained_override is not None, \
            "metagrad server requires a pretrained target model: set easydel_pretrained_override in the config"

        # Snapshot the initial params so reset_state can restore the target model to its
        # pretrained state between reward calls. Only the naive path uses reset_state, and
        # it never reinits (so this runs once at startup). The non-naive path resets by
        # re-running __init__ (which reloads easydel), so it neither reads nor needs this
        # snapshot -- guarding on naive avoids re-saving it on every GRPO-step reinit.
        if self.config.naive_metagrad_server:
            checkpointer = ocp.StandardCheckpointer()
            graphdef, init_params = nnx.split(self.components.model)
            checkpointer.save(self.config.warmstart_checkpoint_dir, init_params)
            checkpointer.wait_until_finished()
            print(f"Warmstart model saved to {self.config.warmstart_checkpoint_dir}")
            checkpointer.close()

        if self.config.target_metric_type != "val_language_modeling":
            # rick_roll / sixseven / l2_norm target metric against a fixed model-weight pattern,
            # not a validation dataset, so no val target is needed.
            self.target_metric_dataset = None
        elif self.config.eleuther_benchmark_lm_eval_tasks is not None:
            # Built once and cached on self.config, which persists across the __init__
            # re-invocations at the end of every reward call (same pattern as the
            # replay-dataset cache below), so the lm-eval load + tokenization does NOT
            # repeat on every GRPO step.
            if not hasattr(self.config, "_cached_eleuther_val_ds"):
                t0 = time.time()
                print(f"Building eleuther benchmark target in-process: "
                      f"{self.config.eleuther_benchmark_lm_eval_tasks} (split={self.config.eleuther_benchmark_split})")
                self.config._cached_eleuther_val_ds = build_eleuther_benchmark_dataset(
                    tokenizer=self.tokenizer,
                    tokenizer_name=self.config.tokenizer_name,
                    lm_eval_tasks=self.config.eleuther_benchmark_lm_eval_tasks,
                    split=self.config.eleuther_benchmark_split,
                    add_data_source=self.config.eleuther_benchmark_add_data_source,
                    shuffle=self.config.eleuther_benchmark_shuffle,
                    random_sample_ratio=self.config.eleuther_benchmark_random_sample_ratio,
                )
                print(f"Built+cached eleuther benchmark target "
                      f"({len(self.config._cached_eleuther_val_ds)} examples) in {time.time()-t0:.1f}s")
            self.target_metric_dataset = self.config._cached_eleuther_val_ds
        else:
            raise ValueError("No validation data: set eleuther_benchmark_lm_eval_tasks in the config (required for target_metric_type='val_language_modeling')")

        # Pre-cache replay dataset once to avoid repeated network I/O each reward step.
        # self.config is the same object across __init__ re-invocations, so this persists.
        # (Replay is unused in the paper experiments; provided as an option for future work.)
        if getattr(self.config, 'replay_data_path', None) is not None and not hasattr(self.config, '_cached_replay_ds'):
            t0 = time.time()
            parquet_files = sorted(glob.glob(self.config.replay_data_path + "/*.parquet"))
            # Pre-load first 2 files (~150k rows) — enough for any expected replay sample size
            n_preload = min(2, len(parquet_files))
            selected_files = parquet_files[:n_preload]
            self.config._cached_replay_ds = load_dataset(
                "parquet", data_files=selected_files, split="train", keep_in_memory=True
            )
            print(f"Cached {len(self.config._cached_replay_ds)} replay examples in memory ({time.time()-t0:.1f}s)")

        # NOTE: self.target_metric_batches is (re)built in load_data_and_run_inner on
        # every call before train() reads it, so __init__ only needs to set up the
        # target_metric_fn here.
        if self.config.target_metric_type == "val_language_modeling":
            def target_metric_fn(target_model, batch):
                print("running target metric")
                inputs = jnp.array(batch["input_ids"])
                targets = jnp.array(batch["labels"])
                return -self.trainer._eval_step(target_model, inputs, targets)

            self.target_metric_fn = target_metric_fn

        else:
            initial_lm_head = jnp.copy(self.trainer.model.lm_head.kernel.value)

            def _log_wandb_float(item_to_log, name, remapped_step_for_logging, val_mode=False, grpo_step_for_logging=None):
                print("Logging:", item_to_log)
                print("Logging with step:", remapped_step_for_logging)
                d = {name + self.trainer.wandb_prefix: float(item_to_log)}
                name_mod = name.split("/")[-1]
                # Log to the train or val run (separate runs; see wandb.init in setup_training),
                # with step=remapped_step (monotonic per run -> resume-backtrack dedup). grpo_step
                # is logged as a field so target_metric/* proxies plot against the comparable axis.
                run = self.trainer.wandb_run_val if val_mode else self.trainer.wandb_run_train
                if run is not None:
                    to_log = dict(d)
                    if grpo_step_for_logging is not None:
                        to_log["grpo_step"] = grpo_step_for_logging
                    run.log(to_log, step=remapped_step_for_logging)
                with open(f"{metadata_save_dir}/{remapped_step_for_logging}_{name_mod}_{self.trainer.wandb_prefix}.json", "w") as f:
                    json.dump(d, f)

            def _save_heatmap(patch, remapped_step_for_logging=None, name="heatmap", inv=False):
                import numpy as np, matplotlib.pyplot as plt
                patch = np.array(patch)
                if inv:
                    patch *= -1
                patch_id = f"step{remapped_step_for_logging}_{name}_{uuid.uuid4().hex[:8]}"

                plt.figure()
                plt.imshow(patch, cmap="gray", aspect="equal")
                plt.tight_layout()
                plt.savefig(f"{metadata_save_dir}/patch_{patch_id}.png", dpi=100)
                plt.close()
                np.save(f"{metadata_save_dir}/patch_{patch_id}.npy", patch)
                print(f"saved patch to {metadata_save_dir}/patch_{patch_id}.png")

            def l2_norm_target_metric(target_model, val_mode=False, remapped_step_for_logging=None, grpo_step_for_logging=None):
                norm = jnp.linalg.norm(target_model.lm_head.kernel.value)
                jax.debug.callback(_log_wandb_float, norm, "target_metric/lm_head_norm_val" if val_mode else "target_metric/lm_head_norm", remapped_step_for_logging=remapped_step_for_logging, val_mode=val_mode, grpo_step_for_logging=grpo_step_for_logging)
                return -norm

            def rick_roll_target_metric(target_model, i=0, j=0, strength=20.0, val_mode=False, remapped_step_for_logging=None, grpo_step_for_logging=None):
                # Extract the LM-head weights (a JAX array)
                W = target_model.lm_head.kernel.value
                S = rick_roll.astype(W.dtype)

                # Safe dynamic slicing (gradients only flow to this region)
                i = jnp.clip(jnp.asarray(i, jnp.int32), 0, W.shape[0]-21)
                j = jnp.clip(jnp.asarray(j, jnp.int32), 0, W.shape[1]-21)
                patch = jax.lax.dynamic_slice(W, (i, j), (21, 21))
                jax.debug.print("Patch {patch}", patch=patch)
                diff = patch - jax.lax.dynamic_slice(initial_lm_head, (i, j), (21, 21))
                acc = jnp.mean(jnp.sign(diff) == S)
                jax.debug.print("Acc {acc}", acc=acc)
                jax.debug.callback(_log_wandb_float, acc, "target_metric/pixel_accuracy_val" if val_mode else "target_metric/pixel_accuracy", remapped_step_for_logging=remapped_step_for_logging, val_mode=val_mode, grpo_step_for_logging=grpo_step_for_logging)

                jax.debug.callback(_save_heatmap, jnp.sign(diff), remapped_step_for_logging=remapped_step_for_logging, name="decoded_image")
                jax.debug.callback(_save_heatmap, jnp.log(1 + jnp.exp(-strength * S * diff)), remapped_step_for_logging=remapped_step_for_logging, name="pixel_loss")
                jax.debug.callback(_save_heatmap, patch, remapped_step_for_logging=remapped_step_for_logging, name="current_weights")
                jax.debug.callback(_save_heatmap, jax.lax.dynamic_slice(initial_lm_head, (i, j), (21, 21)), remapped_step_for_logging=remapped_step_for_logging, name="initial_weights")

                # Penalize deviation from target pattern
                return -jnp.mean(jnp.log(1 + jnp.exp(-strength * S * diff)))

            def sixseven_target_metric(target_model, i=0, j=0, strength=20.0, val_mode=False, remapped_step_for_logging=None, grpo_step_for_logging=None):
                # Extract the LM-head weights (a JAX array)
                W = target_model.lm_head.kernel.value
                S = TARGET_67.astype(W.dtype)

                # Safe dynamic slicing (gradients only flow to this region)
                i = jnp.clip(jnp.asarray(i, jnp.int32), 0, W.shape[0]-6)
                j = jnp.clip(jnp.asarray(j, jnp.int32), 0, W.shape[1]-7)
                patch = jax.lax.dynamic_slice(W, (i, j), (6, 7))
                diff = patch - jax.lax.dynamic_slice(initial_lm_head, (i, j), (6, 7))
                acc = jnp.mean(jnp.sign(diff) == S)
                jax.debug.print("Patch {patch}", patch=patch)
                jax.debug.print("Acc {acc}", acc=acc)
                
                jax.debug.callback(_save_heatmap, jnp.sign(diff), remapped_step_for_logging=remapped_step_for_logging, name="decoded_image")
                jax.debug.callback(_save_heatmap, jnp.log(1 + jnp.exp(-strength * S * diff)), remapped_step_for_logging=remapped_step_for_logging, name="pixel_loss")
                jax.debug.callback(_save_heatmap, patch, remapped_step_for_logging=remapped_step_for_logging, name="current_weights")
                jax.debug.callback(_save_heatmap, jax.lax.dynamic_slice(initial_lm_head, (i, j), (6, 7)), remapped_step_for_logging=remapped_step_for_logging, name="initial_weights")

                jax.debug.callback(_log_wandb_float, acc, "target_metric/pixel_accuracy_val" if val_mode else "target_metric/pixel_accuracy", remapped_step_for_logging=remapped_step_for_logging, val_mode=val_mode, grpo_step_for_logging=grpo_step_for_logging)

                # Penalize deviation from target pattern
                return -jnp.mean(jnp.log(1 + jnp.exp(-strength * S * diff)))

            self.target_metric_fn = {"rick_roll": rick_roll_target_metric, "sixseven": sixseven_target_metric, "l2_norm": l2_norm_target_metric}[self.config.target_metric_type]


    def _load_replay_dataset(self, n_examples: int, seed: int = 42):
        """Load and tokenize n_examples from the optional pretraining replay corpus
        (whatever parquet files `replay_data_path` points at -- not necessarily DCLM)."""
        prefix = (
            "Help read the following article and then rephrase it in different terms. "
            "Remember to keep the meaning and every content of the article intact, "
            "including the title, year, etc. Here is the article:\n"
        )
        if hasattr(self.config, '_cached_replay_ds') and len(self.config._cached_replay_ds) >= n_examples:
            ds = self.config._cached_replay_ds
        else:
            # Fallback: load from disk (cache too small or not yet populated)
            parquet_files = sorted(glob.glob(self.config.replay_data_path + "/*.parquet"))
            n_files = max(1, math.ceil(n_examples / 50000))
            selected_files = parquet_files[:n_files]
            ds = load_dataset("parquet", data_files=selected_files, split="train", keep_in_memory=True)
        ds = ds.shuffle(seed=seed).select(range(n_examples))

        seq_len = self.config.sequence_length

        def tokenize(example):
            text = example["prompt"][0]["content"].removeprefix(prefix)
            tokens = self.tokenizer.encode(text)[:seq_len + 1]
            return {
                "input_ids": tokens[:seq_len],
                "labels": tokens[1:seq_len + 1],
            }

        return ds.map(tokenize, remove_columns=ds.column_names)

    def get_config_dict(self) -> dict:
        """Return current configs as nested dicts suitable for wandb config entry."""
        # Convert the TrainConfig dataclass to a dictionary
        config_dict = asdict(self.config)

        # Create nested structure suitable for wandb
        wandb_config = {
            "metagrad": config_dict,
        }

        return wandb_config

    def reset_state(self):
        # Drain the previous checkpointer's in-flight async (orbax) saves before its
        # directory is deleted/recreated below. Without this, the new checkpointer's
        # rmtree races orbax's background writes.
        self.trainer.checkpointer.close()
        checkpointer = ocp.StandardCheckpointer()
        graphdef, init_params = nnx.split(self.components.model)
        warmstart_params = checkpointer.restore(self.config.warmstart_checkpoint_dir, init_params)
        checkpointer.wait_until_finished()
        print(f"Warmstart model loaded from {self.config.warmstart_checkpoint_dir}")
        checkpointer.close()
        self.trainer.model = nnx.merge(graphdef, warmstart_params)
        shutil.rmtree(self.config.checkpoint_dir, ignore_errors=True)
        self.trainer.checkpointer = create_checkpointer(strategy="disk", checkpoint_dir=self.config.checkpoint_dir)
        print(f"Reset checkpointer to {self.config.checkpoint_dir}")

    def get_metadata_save_dir(self):
        return self.metadata_save_dir

    def load_data_and_run(self, train_path: str, total_batch_size: int, val_mode: bool, grpo_step_for_logging: int, wandb_prefix: str = "") -> list[float]:
        try:
            self.trainer.wandb_prefix = wandb_prefix
            return self.load_data_and_run_inner(train_path, total_batch_size, val_mode, grpo_step_for_logging)
        except Exception:
            import traceback
            print("\n🚨🚨🚨 SERVER-SIDE ERROR TRACEBACK 🚨🚨🚨", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("🚨🚨🚨 END ERROR TRACEBACK 🚨🚨🚨\n", file=sys.stderr)
            raise

    def load_data_and_run_inner(self, train_path: str, total_batch_size: int, val_mode: bool, grpo_step_for_logging: int) -> list[float]:
        start_time = time.time()
        if self._prev_future is not None: # wait for the previous async job to finish
            print(f"Waiting for previous async job to finish")
            self._prev_future.result()
        print(f"Time taken to wait for previous async job to finish: {time.time() - start_time:.3f}s")

        start_time = time.time()

        file_list = glob.glob(train_path+"/*.arrow")
        print(f"Found {len(file_list)} files in {train_path}")
        train_dataset = load_dataset("arrow", data_files=file_list, split="train", keep_in_memory=True)
        n_synthetic = len(train_dataset)
        # Do a sanity check - we have to process all the rollouts and give it a score
        proper_total_batches = math.ceil(total_batch_size / (self.config.microbatch_size * self.config.grad_accumulation_steps))
        if proper_total_batches != self.config.total_batches:
            print(f"WARNING: the rollout batch size doesnt match the server config, changing total_batches from {self.config.total_batches} to {proper_total_batches}")
            self.config.total_batches = proper_total_batches
            self.trainer.total_batches = proper_total_batches

        # Build training dataset: synthetic rollouts plus optional pretraining replay.
        # Replay is not used in the paper experiments; we provide the option here for
        # future exploration (mixing real pretraining data into the inner loop).
        # Replay examples occupy indices [n_synthetic, n_synthetic+n_replay) in the
        # combined dataset, so final_data_weights[:n_synthetic] always holds the
        # per-example metagrads for the synthetic data only.
        if self.config.replay_data_path is not None:
            n_replay_raw = int(n_synthetic * self.config.replay_ratio)
            # Round to a multiple of grad_accumulation_steps so prepare_batch reshape is always valid
            n_replay = round(n_replay_raw / self.config.grad_accumulation_steps) * self.config.grad_accumulation_steps
            n_replay = max(self.config.grad_accumulation_steps, n_replay)
            print(f"Loading {n_replay} replay examples (ratio {self.config.replay_ratio}) for {n_synthetic} synthetic examples")
            replay_ds = self._load_replay_dataset(n_replay, seed=grpo_step_for_logging)
            training_dataset = concatenate_datasets([train_dataset, replay_ds])
        else:
            training_dataset = train_dataset

        dataloader = create_batch_dataset(
            training_dataset,
            self.config.microbatch_size,
            self.config.sequence_length,
            self.config.grad_accumulation_steps,
            do_tokenize=False,
            eos_token_id=self.tokenizer.eos_token_id,
            shuffle=True,
            drop_remainder=False
        )

        if self.config.target_metric_type == "val_language_modeling":
            working_target_metric_dataset = copy.deepcopy(self.target_metric_dataset)

            if self.config.val_sample_fraction is not None:
                working_target_metric_dataset = working_target_metric_dataset.select(range(int(self.config.val_sample_fraction*len(working_target_metric_dataset))))

            target_metric_dataloader = create_batch_dataset(
                working_target_metric_dataset,
                self.config.microbatch_size,
                self.config.sequence_length,
                1,
                do_tokenize=False,
                drop_remainder=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
                eos_token_id=self.tokenizer.eos_token_id,
                shuffle=False
            )
            self.target_metric_batches = [batch for batch in target_metric_dataloader]
        else:
            self.target_metric_batches = None
            
        print(f"Time taken to check params and create dataloader: {time.time() - start_time:.3f}s")

        print("\nStarting training with metagrads...")

        if self.config.naive_metagrad_server:
            with self.mesh:
                outputs = self.trainer.train(
                    train_dataloader=dataloader,
                    target_metric_fn=self.target_metric_fn,
                    target_metric_fn_batches=self.target_metric_batches,
                    with_metagrads=True,
                    init_params=None,
                    use_wandb=self.config.use_wandb,
                    terminate_after_target_metric=True,
                    val_mode=val_mode,
                    grpo_step_for_logging=grpo_step_for_logging,
                )
            self._prev_future = self._bg.submit(self.reset_state)
            return [float(outputs["final_target_metric"])] * n_synthetic
        else:
            start_time = time.time()
            with self.mesh:
                outputs = self.trainer.train(
                    train_dataloader=dataloader,
                    target_metric_fn=self.target_metric_fn,
                    target_metric_fn_batches=self.target_metric_batches,
                    with_metagrads=True,
                    init_params=None,
                    use_wandb=self.config.use_wandb,
                    val_mode=val_mode,
                    grpo_step_for_logging=grpo_step_for_logging,
                )

                duration = time.time() - start_time
                print(f"Training completed in {duration:.3f}s")

            # Full reinit (not reset_state) on purpose: the non-naive path just ran the
            # full reverse-mode VJP, so the trainer/components objects are still holding a
            # lot of device memory (batch_data_list, accumulated grads, compiled VJP
            # closures). Rebinding them to fresh objects drops those references so JAX can
            # free the buffers before the next step, instead of letting them accumulate as
            # reset_state (which reuses the same objects) would.
            self.shutdown()
            self.__init__(self.metadata_save_dir, self.config)
            return outputs["final_data_weights"][:n_synthetic].tolist()
    
    def shutdown(self):
        if os.path.exists(self.config.checkpoint_dir):
            shutil.rmtree(self.config.checkpoint_dir)
        if os.path.exists(self.config.warmstart_checkpoint_dir):
            shutil.rmtree(self.config.warmstart_checkpoint_dir)
        self.trainer.checkpointer.close()
        self._bg.shutdown()


def main() -> None:
    # Parse server-specific arguments with argparse
    parser = argparse.ArgumentParser(description='Run the metagrad reward server with Hydra ML config')
    parser.add_argument('--config-path', type=str, required=True,
                      help='Path to the server config YAML (repo-root-relative, e.g. '
                           'experiment_configs/lambada_experiments/adam/metagrad_server_..._lambada_es.yaml)')
    parser.add_argument('--port', type=int, default=29922,
                      help='Port to listen on (default: 29922)')
    
    # Parse known args to separate server args from Hydra overrides
    server_args, hydra_args = parser.parse_known_args()
    
    # Load config using helper function
    _, train_config = load_hydra_config(server_args.config_path, hydra_args)

    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ("/RPC2",)

    server = SimpleXMLRPCServer(("0.0.0.0", server_args.port),
                        requestHandler=RequestHandler,
                        allow_none=True, logRequests=True)

    train_config.wandb_name = os.environ.get(
        "METAGRAD_WANDB_NAME",
        os.path.splitext(os.path.basename(server_args.config_path))[0],
    )

    metadata_save_dir = os.environ.get(
        "METAGRAD_METADATA_SAVE_DIR",
        os.path.join(os.getcwd(), "metagrad_server_outputs"),
    )
    os.makedirs(metadata_save_dir, exist_ok=True)

    instance = MetagradRewardRun(
        metadata_save_dir=metadata_save_dir,
        train_config=train_config,
    )
    
    def handle_shutdown(signum, frame):
        print("\nShutting down gracefully...")
        instance.shutdown()
        server.server_close()
        exit(0)
        
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    server.register_instance(instance)
    print(f"XML‑RPC service listening on :{server_args.port}")
    print(f"Using Hydra config:")
    print(f"  Target Model: EasyDeL pretrained '{train_config.easydel_pretrained_override}'")
    print(f"  Tokenizer: {train_config.tokenizer_name}")
    print(f"  Learning rate: {train_config.learning_rate}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        handle_shutdown(None, None)


if __name__ == "__main__":
    main()
