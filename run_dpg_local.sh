#!/bin/bash
#
# run_dpg_local.sh
#
# Self-contained metagrad RL run. Everything it needs lives in THIS
# repo (dataset-policy-gradients) -- it does not reference the old
# metagrad-synthetic-pretraining directory at all:
#
#   - metagrad RPC server : dataset_metagradients_jax/scripts/metagrad_server.py  (run via uv)
#   - verl PPO trainer    : ./verl, installed editable in the ./.venv-verl venv
#   - custom reward fn     : ./custom_metagrad_reward.py
#   - server meta config   : experiment_configs/lambada_experiments/{adam,sgd}/metagrad_server_cross_group_batching_llama3.2_instruct_lambada_<lang>.yaml  (flattened standalone; builds the lambada benchmark in-process)
#   - verl trainer config  : experiment_configs/lambada_experiments/verl_llama3.2_instruct.yaml (-> hydra overrides via yaml_to_hydra.py)
#   - RL data (copied in)  : ./prompt_data/{wikipedia_{train,test}.parquet, prompt_ending_to_force.txt}
#   - val/target benchmark : built in-process by the server from an lm-eval task
#                            (eleuther_benchmark_* in the server config); tasks vendored in ./custom_eleuther_evals
#
# ASSUMPTIONS:
#   - You are already on a node with 8 visible GPUs.
#   - `uv` is installed and on PATH.
#   - The trainer venv ./.venv-verl has been built (see ./install-verl-env.sh).
#
# Usage: ./run_dpg_local.sh [experiment_name] [output_dir]

set -euo pipefail

EXPERIMENT_NAME=${1:-claude_local_test}
OUTPUT_DIR_ARG=${2:-}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR"

# --- Tooling on PATH ----------------------------------------------------------
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then echo "Error: uv not found on PATH"; exit 1; fi

# verl trainer env (build it with ./install-verl-env.sh)
VERL_VENV="$REPO_DIR/.venv-verl"
if [ ! -x "$VERL_VENV/bin/python" ]; then
    echo "Error: trainer venv missing at $VERL_VENV -- build it with ./install-verl-env.sh"; exit 1
fi

# --- Config selection ---------------------------------------------------------
# Override METAGRAD_SERVER_CONFIG / VERL_TRAINER_CONFIG (env vars) to run other experiments
# (uuid, qr_code, ...); defaults are the lambada/es metagrad run.
#   METAGRAD_SERVER_CONFIG       : repo-root-relative (metagrad_server.py resolves it against its
#                         own package root, not cwd).
#   VERL_TRAINER_CONFIG : absolute path.
REWARD_FN="$REPO_DIR/custom_metagrad_reward.py"
METAGRAD_SERVER_CONFIG="${METAGRAD_SERVER_CONFIG:-experiment_configs/lambada_experiments/adam/metagrad_server_cross_group_batching_llama3.2_instruct_lambada_es.yaml}"
VERL_TRAINER_CONFIG="${VERL_TRAINER_CONFIG:-$REPO_DIR/experiment_configs/lambada_experiments/verl_llama3.2_instruct.yaml}"
[ -e "$VERL_TRAINER_CONFIG" ] || { echo "Error: trainer config not found: $VERL_TRAINER_CONFIG"; exit 1; }

# RL data + prompt-ending paths live IN the trainer config; pull them out so the
# preflight checks the files this particular run will actually use.
TRAIN_FILES=$(awk '/^  train_files:/{print $2; exit}' "$VERL_TRAINER_CONFIG")
VAL_FILES=$(awk '/^  val_files:/{print $2; exit}' "$VERL_TRAINER_CONFIG")
PROMPT_ENDING=$(awk '/^  prompt_ending_to_force_path:/{print $2; exit}' "$VERL_TRAINER_CONFIG")

for p in "$TRAIN_FILES" "$VAL_FILES" "$PROMPT_ENDING" "$REWARD_FN" \
         "$REPO_DIR/$METAGRAD_SERVER_CONFIG" "$VERL_TRAINER_CONFIG" "$REPO_DIR/yaml_to_hydra.py" \
         "$REPO_DIR/verl/verl" "$REPO_DIR/dataset_metagradients_jax/scripts/metagrad_server.py"; do
    [ -e "$p" ] || { echo "Error: required path missing: $p"; exit 1; }
done
echo "Server config:  $METAGRAD_SERVER_CONFIG"
echo "Trainer config: $VERL_TRAINER_CONFIG"

# --- Output dir + scratch -----------------------------------------------------
timestamp=$(date +%Y%m%d_%H%M%S)
if [ -n "$OUTPUT_DIR_ARG" ]; then
    OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR_ARG")
else
    OUTPUT_DIR="$REPO_DIR/experiment_outputs/${EXPERIMENT_NAME}_${timestamp}"
fi
mkdir -p "$OUTPUT_DIR/checkpoints/${EXPERIMENT_NAME}-dpg" "$OUTPUT_DIR/verl_outputs" "$OUTPUT_DIR/metagrad_server_outputs"

# Shared scratch used by BOTH the server and the reward fn (rollout hand-off +
# jax cache + server checkpoints). They must agree on this path.
export LOCAL_FAST_STORAGE="${LOCAL_FAST_STORAGE:-/tmp/$USER}"
mkdir -p "$LOCAL_FAST_STORAGE"

echo "Repo:        $REPO_DIR"
echo "Experiment:  $EXPERIMENT_NAME"
echo "Output dir:  $OUTPUT_DIR"
echo "Scratch:     $LOCAL_FAST_STORAGE"

# --- Launch the local metagrad RPC server in the background -------------------
# These two env vars are consumed by the edits in metagrad_server.py (metadata_save_dir /
# wandb_name) -- set them here so they are actually populated.
export METAGRAD_METADATA_SAVE_DIR="$OUTPUT_DIR/metagrad_server_outputs"
export METAGRAD_WANDB_NAME="$EXPERIMENT_NAME"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Shutting down metagrad server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting local metagrad server (logging to $OUTPUT_DIR/server.log)..."
(
    cd "$REPO_DIR/dataset_metagradients_jax"
    export WANDB_RUN_GROUP=$EXPERIMENT_NAME
    export WANDB_JOB_TYPE=metagrad
    exec uv run python -u scripts/metagrad_server.py \
        --config-path "$METAGRAD_SERVER_CONFIG" >> "$OUTPUT_DIR/server.log" 2>&1
) &
SERVER_PID=$!

echo "Waiting for server to come up (PID $SERVER_PID)..."
# First run also builds the uv venv, so allow generous startup time. Wait until
# the server binds the RPC port, or bail if the process dies first.
for i in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: metagrad server exited during startup. Tail of server.log:"; tail -n 50 "$OUTPUT_DIR/server.log" || true; exit 1
    fi
    if grep -q "XML.RPC service listening" "$OUTPUT_DIR/server.log" 2>/dev/null; then
        echo "Server is up."; break
    fi
    sleep 10
done

# --- Run the verl PPO trainer against the local server ------------------------
# Use the repo-local trainer venv; its `verl` is installed editable from ./verl,
# so no PYTHONPATH override is needed.
source "$VERL_VENV/bin/activate"

export WANDB_PROJECT=dpg_nano
export WANDB_RUN_GROUP=$EXPERIMENT_NAME
export WANDB_JOB_TYPE=rl

echo "Starting verl PPO training..."
# Base overrides are generated from the cross_group_batching trainer config (the source of
# truth -- do not hand-edit them here; the config is self-contained, incl. local
# data paths). The dynamic overrides that follow only set things that are
# inherently per-run (reward fn path, run name, output dirs); since they come
# last, they win over any matching key in the config.
base_overrides=$(python3 "$REPO_DIR/yaml_to_hydra.py" "$VERL_TRAINER_CONFIG")
# Capture the trainer's real exit code (PIPESTATUS[0], not tee's) so a crash is
# not masked by the pipe. We still attempt the checkpoint merge below regardless,
# but the code is propagated as the script's exit status at the end (see `exit`),
# so a failed run reports FAILED to Slurm/sacct instead of a misleading COMPLETED.
train_rc=0
(
    cd "$REPO_DIR"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
        ${base_overrides} \
        custom_reward_function.path="$REWARD_FN" \
        trainer.experiment_name="${EXPERIMENT_NAME}-verl" \
        trainer.project_name=dpg \
        trainer.default_local_dir="$OUTPUT_DIR/checkpoints/${EXPERIMENT_NAME}-dpg" \
        hydra.run.dir="$OUTPUT_DIR/verl_outputs" \
        2>&1 | tee -a "$OUTPUT_DIR/${EXPERIMENT_NAME}-dpg.log"
    exit "${PIPESTATUS[0]}"
) || train_rc=$?
if [ "$train_rc" -ne 0 ]; then
    echo "Training exited with code $train_rc, but attempting checkpoint merge anyway"
fi

# --- Merge the final FSDP actor checkpoint -----------------------------------
checkpoint_folder=$(find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -type d ! -path "$OUTPUT_DIR/checkpoints" | head -1)
if [ -f "$checkpoint_folder/latest_checkpointed_iteration.txt" ]; then
    latest_it=$(cat "$checkpoint_folder/latest_checkpointed_iteration.txt")
    actor_dir="$checkpoint_folder/global_step_${latest_it}/actor"
    if [ -d "$actor_dir" ]; then
        python3 "$REPO_DIR/verl/scripts/model_merger.py" merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$OUTPUT_DIR/merged_final_checkpoint"
    else
        echo "Actor directory not found at: $actor_dir"
    fi
else
    echo "No checkpoint iteration file found under $checkpoint_folder"
fi

echo "Done. Outputs in: $OUTPUT_DIR"

# Propagate the trainer's exit code so a crashed run is reported as FAILED
# (not COMPLETED) by Slurm/sacct. A failed checkpoint merge above already aborts
# via `set -e`, so reaching here means only training may have failed.
exit "$train_rc"
