#!/bin/bash
# =============================================================================
# example_slurm_launch_all.sh  --  EXAMPLE ONLY, not required to use this repo.
#
# An example of how you *could* launch every experiment setting at once with
# Slurm (here, on 8xH200 "miso" nodes). The repo itself is scheduler-agnostic:
# run_dpg_local.sh runs on any node with 8 capable GPUs and nothing else assumes
# Slurm. Treat the SBATCH_ARGS below as a template -- edit the partition/account/
# resources for your cluster, or replace `sbatch --wrap` with your own launcher.
#
# It submits one independent 8-GPU job per (METAGRAD_SERVER_CONFIG,
# VERL_TRAINER_CONFIG) pair (~50 jobs). Comment out blocks you don't want.
# =============================================================================
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR"
mkdir -p slurm_logs

# EXAMPLE per-job Slurm resources (one 8-GPU node). Adapt to your cluster.
SBATCH_ARGS=(
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=160 --gres=gpu:8 --mem=2400G
  --time=14-00:00:00 --partition=miso --account=miso --requeue --open-mode=append
)

# submit <name> <metagrad_server_config> <verl_trainer_config>
submit() {
  local name="$1" server="$2" trainer="$3"
  sbatch "${SBATCH_ARGS[@]}" --job-name="dpg_${name}" \
    --output="slurm_logs/dpg_${name}_%j.out" --error="slurm_logs/dpg_${name}_%j.err" \
    --wrap="cd '$REPO_DIR'; export METAGRAD_SERVER_CONFIG='$server'; export VERL_TRAINER_CONFIG='$REPO_DIR/$trainer'; ./run_dpg_local.sh '$name' 'experiment_outputs/$name'"
}

# submit_baseline <name> <baseline_config>   (server-less, via run_dpg_baseline_local.sh)
# Pass an explicit, stable output dir (experiment_outputs/<name>) like submit() does;
# otherwise run_dpg_baseline_local.sh defaults to a fresh experiment_outputs/<name>_<timestamp>
# each launch, so a requeued/re-run baseline would start over instead of resuming from its
# latest checkpoint (verl resume_mode=auto keys off a fixed output dir).
submit_baseline() {
  local name="$1" cfg="$2"
  sbatch "${SBATCH_ARGS[@]}" --job-name="dpg_${name}" \
    --output="slurm_logs/dpg_${name}_%j.out" --error="slurm_logs/dpg_${name}_%j.err" \
    --wrap="cd '$REPO_DIR'; ./run_dpg_baseline_local.sh '$name' '$cfg' 'experiment_outputs/$name'"
}

EC=experiment_configs

# ---- lambada: es/fr/de/it x {adam, sgd, naive} + server-less baselines ----
for lang in es fr de it; do
  submit "lambada_adam_$lang"  "$EC/lambada_experiments/adam/metagrad_server_cross_group_batching_llama3.2_instruct_lambada_$lang.yaml" "$EC/lambada_experiments/verl_llama3.2_instruct.yaml"
  submit "lambada_sgd_$lang"   "$EC/lambada_experiments/sgd/metagrad_server_cross_group_batching_llama3.2_instruct_lambada_$lang.yaml"  "$EC/lambada_experiments/verl_llama3.2_instruct.yaml"
  submit "lambada_naive_$lang" "$EC/lambada_experiments/naive/metagrad_server_llama3.2_instruct_lambada_$lang.yaml"                      "$EC/lambada_experiments/verl_llama3.2_instruct_groupless_advantage.yaml"
  for b in levenshtein embedding_sim fasttext_lang_id; do
    submit_baseline "lambada_${b}_$lang" "$EC/lambada_experiments/baselines/$b/baseline_llama3.2_instruct_lambada_$lang.yaml"
  done
done

# ---- uuid x {adam, sgd, naive} ----
submit "uuid_adam"  "$EC/uuid_experiments/adam/metagrad_server_cross_group_batching_llama3.2_instruct_uuid.yaml" "$EC/uuid_experiments/verl_llama3.2_instruct.yaml"
submit "uuid_sgd"   "$EC/uuid_experiments/sgd/metagrad_server_cross_group_batching_llama3.2_instruct_uuid.yaml"  "$EC/uuid_experiments/verl_llama3.2_instruct.yaml"
submit "uuid_naive" "$EC/uuid_experiments/naive/metagrad_server_llama3.2_instruct_uuid.yaml"                     "$EC/uuid_experiments/verl_llama3.2_instruct_groupless_advantage.yaml"

# ---- qr_code (adam only) ----
submit "qr_code_adam" "$EC/qr_code_experiments/adam/metagrad_server_cross_group_batching_gpt2_qr_code.yaml" "$EC/qr_code_experiments/verl_llama3.2_instruct.yaml"

# ---- 67: bs256/bs2048/bs24576 x {adam, sgd, naive}; adam_nocross for bs24576 only ----
for bs in bs256 bs2048 bs24576; do
  submit "67_adam_$bs"          "$EC/67_experiments/adam/metagrad_server_cross_group_batching_gpt2_67.yaml"    "$EC/67_experiments/verl_llama3.2_instruct_$bs.yaml"
  [ "$bs" = bs24576 ] && submit "67_adam_nocross_$bs"  "$EC/67_experiments/adam/metagrad_server_no_cross_group_batching_gpt2_67.yaml" "$EC/67_experiments/verl_llama3.2_instruct_$bs.yaml"
  submit "67_sgd_$bs"           "$EC/67_experiments/sgd/metagrad_server_cross_group_batching_gpt2_67.yaml"     "$EC/67_experiments/verl_llama3.2_instruct_$bs.yaml"
  submit "67_naive_$bs"         "$EC/67_experiments/naive/metagrad_server_gpt2_67.yaml"                        "$EC/67_experiments/verl_llama3.2_instruct_${bs}_groupless_advantage.yaml"
done

# ---- l2_norm: bs256/bs2048/bs24576 x {adam, sgd, naive} ----
for bs in bs256 bs2048 bs24576; do
  submit "l2_norm_adam_$bs"  "$EC/l2_norm_experiments/adam/metagrad_server_cross_group_batching_gpt2_l2_norm.yaml" "$EC/l2_norm_experiments/verl_llama3.2_instruct_$bs.yaml"
  submit "l2_norm_sgd_$bs"   "$EC/l2_norm_experiments/sgd/metagrad_server_cross_group_batching_gpt2_l2_norm.yaml"  "$EC/l2_norm_experiments/verl_llama3.2_instruct_$bs.yaml"
  submit "l2_norm_naive_$bs" "$EC/l2_norm_experiments/naive/metagrad_server_gpt2_l2_norm.yaml"                     "$EC/l2_norm_experiments/verl_llama3.2_instruct_${bs}_groupless_advantage.yaml"
done

echo "Submitted all example jobs (see: squeue -u \$USER)."
