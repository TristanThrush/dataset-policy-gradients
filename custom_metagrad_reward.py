from transformers import AutoTokenizer
from datasets import Dataset
import os
import uuid
import numpy as np
import xmlrpc.client
import traceback
import wandb
import fasttext
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
fasttext_model = fasttext.load_model(fasttext_model_path)

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

rand_name = str(uuid.uuid4())

tmp_path = os.path.expandvars("$LOCAL_FAST_STORAGE")

import Levenshtein

emb_model = SentenceTransformer("tomaarsen/static-similarity-mrl-multilingual-v1")


def reward_fn_levenshtein(train_dataset, val_dataset):
    rewards = []
    num_val = len(val_dataset)
    for example in train_dataset:
        sum_dist = 0
        for example_val in val_dataset:
            sum_dist += Levenshtein.distance(example["input_ids"], example_val["input_ids"])
        rewards.append(-float(sum_dist) / num_val)
    return rewards


def reward_fn_fasttext_lang_id(train_dataset, val_dataset):
    labels, probs = fasttext_model.predict(" ".join([obj["prompt"] + " " + obj["completion"] for obj in val_dataset]).replace("\n", " "), k=len(fasttext_model.labels))
    val_lang = labels[0]
                
    def fasttext_lang_id_dist(s):
        labels, probs = fasttext_model.predict(s.replace("\n"," "), k=len(fasttext_model.labels))
        return probs[labels.index(val_lang)]
                
    rewards = []
    for example in train_dataset:
        rewards.append(fasttext_lang_id_dist(example["input_str"]))

    return rewards

def reward_fn_multilingual_embedding(train_dataset, val_dataset):

    train_embeddings = emb_model.encode(train_dataset["input_str"][:])
    val_embeddings = emb_model.encode([ex["prompt"] + " " + ex["completion"] for ex in val_dataset])
    scores = emb_model.similarity(train_embeddings, val_embeddings)    

    rewards = []
    for i in range(len(train_dataset)):
        reward_sum = 0
        for j in range(len(val_dataset)):
            reward_sum += scores[i][j]
        reward_sum /= len(val_dataset)
        rewards.append(float(reward_sum))

    return rewards


def update_wandb_config(get_server_config_fn):
    # Check if wandb is initialized and handle config
    if wandb.run is not None:
        try:
            server_config = wandb.config.get("server", None)
            if server_config is None:
                server_config = get_server_config_fn()
                wandb.config.update({"server": server_config})
                print(f"Added server config: {server_config}")
        except Exception as e:
            print(f"Error accessing/updating wandb config: {e}")
    

# --- Server-less baseline targets (levenshtein / embedding_sim / fasttext_lang_id) ---
# Built once from the eleuther benchmark config and cached in-process. compute_score runs
# many times per training run, so the target must not be rebuilt on every call.
_baseline_val_ds_cache = {}

def _build_eleuther_benchmark_dataset(tokenizer, tokenizer_name, lm_eval_tasks, split):
    # Load the builder straight from its file rather than via the dataset_metagradients_jax
    # package, whose __init__ imports jax/easydel that we do not want in the verl env.
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dataset_metagradients_jax", "src", "eleuther_benchmark.py")
    spec = importlib.util.spec_from_file_location("eleuther_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_eleuther_benchmark_dataset(tokenizer, tokenizer_name, lm_eval_tasks, split)

def _get_baseline_val_ds(configs, tokenizer):
    m = configs["metagrad"]
    tasks = m["eleuther_benchmark_lm_eval_tasks"]
    split = m.get("eleuther_benchmark_split", "train")
    key = (tasks, split)
    if key not in _baseline_val_ds_cache:
        _baseline_val_ds_cache[key] = _build_eleuther_benchmark_dataset(tokenizer, m["tokenizer_name"], tasks, split)
        print(f"Built+cached baseline target ({len(_baseline_val_ds_cache[key])} examples) for {tasks}/{split}")
    return _baseline_val_ds_cache[key]


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, val_mode=False, grpo_step=None) -> list[float]:

    print(f"STARTING REWARD COMPUTATION ON {len(solution_strs)} INPUTS")
    
    rollout_path = f'{tmp_path}/synthetic_pretraining_grpo_rollouts/{rand_name}/'
    os.makedirs(rollout_path, exist_ok=True)

    try:
        # Baseline mode (levenshtein / embedding_sim / fasttext_lang_id): read config
        # from a local yaml and skip the metagrad server entirely. Otherwise, talk to
        # the running verl server over RPC as usual.
        baseline_config_path = os.environ.get("METAGRAD_BASELINE_CONFIG")
        if baseline_config_path:
            import yaml
            with open(baseline_config_path) as f:
                configs = {"metagrad": yaml.safe_load(f)}
            metadata_save_dir = os.environ.get("METAGRAD_METADATA_SAVE_DIR", os.path.join(tmp_path, "metagrad_server_outputs"))
            os.makedirs(metadata_save_dir, exist_ok=True)
            proxy = None
        else:
            proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:29922", allow_none=True)
            configs = proxy.get_config_dict()
            metadata_save_dir = proxy.get_metadata_save_dir()
        try:

            tokenizer_name = configs["metagrad"]["tokenizer_name"]
            tokenizer_kwargs = configs["metagrad"].get("tokenizer_kwargs", {}) or {}
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
            solution_strs_input_ids = [tokenizer.encode(solution_str) for solution_str in solution_strs]
            solution_strs_labels = [obj[1:] + [-100] for obj in solution_strs_input_ids]
            d = {'domain': [None]*len(solution_strs), "gt": ground_truths, "input_str": solution_strs, "input_ids": solution_strs_input_ids, "labels": solution_strs_labels, "prompt_id": [obj["prompt_id"] for obj in extra_infos]}
            if not configs["metagrad"].get("cross_group_batching", True):
                inverse_groups = []
                inverse_group_sources = []
                map_back_to_original_index = {}
                for index in range(len(solution_strs)):
                    if len(inverse_groups) == 0:
                        inverse_groups.append({'domain': [None], "gt": [d["gt"][index]], "input_str": [d["input_str"][index]], "input_ids": [d["input_ids"][index]], "labels": [d["labels"][index]], "prompt_id": [d["prompt_id"][index]]})
                        inverse_group_sources.append(set([d["prompt_id"][index]]))
                        map_back_to_original_index[(0,0)] = index
                        continue
                    working_inverse_group_index = 0
                    while d["prompt_id"][index] in inverse_group_sources[working_inverse_group_index]:
                        working_inverse_group_index += 1
                        if working_inverse_group_index == len(inverse_groups):
                            break

                    if working_inverse_group_index == len(inverse_groups):
                        inverse_groups.append({'domain': [None], "gt": [d["gt"][index]], "input_str": [d["input_str"][index]], "input_ids": [d["input_ids"][index]], "labels": [d["labels"][index]], "prompt_id": [d["prompt_id"][index]]})
                        inverse_group_sources.append(set([d["prompt_id"][index]]))
                        map_back_to_original_index[(working_inverse_group_index,0)] = index
                    else:
                        inverse_groups[working_inverse_group_index]["domain"].append(None)
                        inverse_groups[working_inverse_group_index]["input_str"].append(d["input_str"][index])
                        inverse_groups[working_inverse_group_index]["input_ids"].append(d["input_ids"][index])
                        inverse_groups[working_inverse_group_index]["labels"].append(d["labels"][index])
                        inverse_groups[working_inverse_group_index]["prompt_id"].append(d["prompt_id"][index])
                        inverse_groups[working_inverse_group_index]["gt"].append(d["gt"][index])
                        inverse_group_sources[working_inverse_group_index].add(d["prompt_id"][index])
                        map_back_to_original_index[(working_inverse_group_index,len(inverse_groups[working_inverse_group_index]["domain"])-1)] = index
                
                grads_list = [0]*len(solution_strs)
                for inverse_group_index, inverse_group in enumerate(inverse_groups):
                    inverse_group_batch_size = len(inverse_group["input_str"])
                    ds = Dataset.from_dict(inverse_group)
                    ds.save_to_disk(rollout_path)
                    ds.save_to_disk(metadata_save_dir + f"/rollout_step_{str(grpo_step)}_inverse_group_{str(inverse_group_index)}_val_{str(val_mode)}")
                    if configs["metagrad"]["target_metric_type"] == "embedding_sim":
                        inverse_group_grads_list = reward_fn_multilingual_embedding(ds, _get_baseline_val_ds(configs, tokenizer))
                    elif configs["metagrad"]["target_metric_type"] == "levenshtein":
                        inverse_group_grads_list = reward_fn_levenshtein(ds, _get_baseline_val_ds(configs, tokenizer))
                    elif configs["metagrad"]["target_metric_type"] == "fasttext_lang_id":
                        inverse_group_grads_list = reward_fn_fasttext_lang_id(ds, _get_baseline_val_ds(configs, tokenizer))
                    else:
                        inverse_group_grads_list = proxy.load_data_and_run(rollout_path, inverse_group_batch_size, val_mode, grpo_step, f"_inverse_group_{str(inverse_group_index)}")

                    for index_within_inverse_group in range(len(inverse_group_grads_list)):
                        grads_list[map_back_to_original_index[(inverse_group_index, index_within_inverse_group)]] = inverse_group_grads_list[index_within_inverse_group]
            else:
                ds = Dataset.from_dict(d)
                ds.save_to_disk(metadata_save_dir + f"/rollout_step_{str(grpo_step)}_val_{str(val_mode)}")
                total_batch_size = len(solution_strs)
                ds.save_to_disk(rollout_path)
                grads_list = proxy.load_data_and_run(rollout_path, total_batch_size, val_mode, grpo_step)
                update_wandb_config(proxy.get_config_dict)
        except Exception as e:
            print("Error during RPC call to load_data_and_run:")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            traceback.print_exc()
            # Return zero rewards instead of propagating the exception
            return [0.0] * len(solution_strs)
            
        if grads_list is None:
            print("WARNING: RPC call returned None, using zero rewards")
            return [0.0] * len(solution_strs)
            
        rewards = np.array(grads_list)

        try:
            rewards_final = [float(r) for r in rewards]
            reward_mean = float(np.mean(rewards_final))
            reward_std = float(np.std(rewards_final))
            rewards_max = float(np.max(rewards_final))
            rewards_min = float(np.min(rewards_final))
            # The per-example rewards are metagradients (~1e-9 in magnitude), so the
            # batch mean is ~0 by construction. The actual GRPO signal lives in the
            # *spread* (advantages are normalized within prompt groups), so log
            # std/min/max too and print at full precision -- otherwise the mean alone
            # at 3 decimals just reads as a flat 0.000 even while training works. The
            # meaningful "is the target improving" curve is the server's primal
            # reward (wandb project dataset-metagradients-jax).
            prefix = "reward/reward_val" if val_mode else "reward/reward"
            wandb.log({
                prefix: reward_mean,
                f"{prefix}_std": reward_std,
                f"{prefix}_min": rewards_min,
                f"{prefix}_max": rewards_max,
            }, step=grpo_step)
            print(f"STATS: reward mean: {reward_mean:.3e}, reward std: {reward_std:.3e}, reward max: {rewards_max:.3e}, reward min: {rewards_min:.3e}")
            return rewards_final
        except (TypeError, ValueError) as e:
            print(f"Error converting rewards to float: {e}")
            print(f"Rewards type: {type(rewards)}, shape: {getattr(rewards, 'shape', 'no shape')}")
            return [0.0] * len(solution_strs)
            
    except Exception as e:
        print(f"CRITICAL ERROR in reward computation: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        # Return zero rewards to prevent Ray serialization issues
        return [0.0] * len(solution_strs)

