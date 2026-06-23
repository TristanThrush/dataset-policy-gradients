import argparse
import numpy as np
import jax
from dataset_metagradients_jax.train_utils import get_config, setup_training
from jax.sharding import PartitionSpec as P
from transformers import AutoModelForCausalLM, AutoTokenizer
import getpass
import os

_SCRATCH = os.environ.get("LOCAL_FAST_STORAGE", f"/tmp/{getpass.getuser()}")

def compare_easydel_and_hf(cache_dir=f"{_SCRATCH}/.jax_cache", checkpoint_dir=f"{_SCRATCH}/checkpoints", model_id="allenai/OLMo-1B-0724-hf"):
    config = get_config(model_id, jax_cache_dir=cache_dir, checkpoint_dir=checkpoint_dir, dtype="fp32")
    components = setup_training(config)
    easydel_model = components.model
    easydel_model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float32")
    hf_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    test_strings = ["This is a test string.", "023niefvf0vn 30fffd0vn e 0wdcsmx,", "Chris and Tatsu are cool!", "this is yet another string",\
"we want 8 such strings for sharding to work well", "woah this is a self-referential example", "this is the seventh item", "this is the 8th"]
    
    jax_input_ids = tokenizer(test_strings, return_tensors="jax", padding=True)['input_ids']
    pt_input_ids = tokenizer(test_strings, return_tensors="pt", padding=True)['input_ids']

    with components.mesh:
        easydel_inputs = jax.lax.with_sharding_constraint(jax_input_ids, P('data', None))
        easydel_logits = easydel_model(easydel_inputs)
    
    hf_logits = hf_model(pt_input_ids).logits
    print('HF logits:', hf_logits)
    print('EasyDeL logits:', easydel_logits)
    assert np.allclose(np.array(hf_logits.detach()), np.array(easydel_logits), rtol=1e1, atol=1e1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run metagrad correctness tests')
    parser.add_argument('--cache-dir', default=f"{_SCRATCH}/.jax_cache",
                      help=f'JAX cache directory (default: {_SCRATCH}/.jax_cache)')
    parser.add_argument('--checkpoint-dir', default=f"{_SCRATCH}/checkpoints",
                      help=f'Model checkpoint directory (default: {_SCRATCH}/checkpoints)')
    parser.add_argument('--model_id', default="allenai/OLMo-1B-0724-hf",
                      help='Name of the Hugging Face model id')
    args = parser.parse_args()

    # Configure JAX
    jax.config.update("jax_compilation_cache_dir", args.cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_compiler_enable_remat_pass", False)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    jax.config.update("jax_explain_cache_misses", True)

    try:
        compare_easydel_and_hf(args.cache_dir, args.checkpoint_dir, args.model_id)
    except AssertionError as e:
        print(f"Test failed: {e}")
