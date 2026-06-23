"""Build a tokenized metagrad target ("benchmark") dataset directly from lm-eval task(s).

Ported from convert_eleuther_task_to_metagrads_target.py so a target dataset can be
constructed in-process (by the verl server) or offline (by a builder script) instead of
loading a pre-saved arrow dataset. The tokenization is kept byte-identical to that script
(morse-code and top_n/top_n_ratio truncation are intentionally omitted).

Light dependencies only (lm_eval, datasets, transformers tokenizer passed in) -- no jax /
easydel -- so it can be imported by tools that don't need the full server stack.
"""
import os

from datasets import Dataset as HFDataset


def default_include_path():
    """Repo-root/custom_eleuther_evals, resolved from this module's location.

    This file lives at <repo>/dataset_metagradients_jax/src/, so the repo root is three
    directories up.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "custom_eleuther_evals")


def build_eleuther_benchmark_dataset(tokenizer, tokenizer_name, lm_eval_tasks, split,
                                     include_path=None, add_data_source=False, shuffle=False,
                                     random_sample_ratio=None):
    """Build a tokenized metagrad target dataset from lm-eval task(s).

    Returns an HF Dataset with columns prompt, completion, domain, input_ids, labels
    (+ data_source if add_data_source)."""
    from lm_eval import tasks as _lm_eval_tasks
    if include_path is None:
        include_path = default_include_path()

    task_names = lm_eval_tasks.split(",")
    tm = _lm_eval_tasks.TaskManager(include_path=include_path)
    task_list = _lm_eval_tasks.get_task_dict(task_names, task_manager=tm)

    rows = []
    for tname, task in task_list.items():
        if split == "train":
            docs = task.training_docs()
        elif split == "validation":
            docs = task.validation_docs()
        else:
            docs = task.test_docs()
        for doc in docs:
            prompt = task.doc_to_text(doc)
            target = task.doc_to_target(doc)
            # MC tasks return an index -- map it back to the string label
            if isinstance(target, int):
                target = task.doc_to_choice(doc)[target]
            row_to_append = {"prompt": prompt, "completion": target}
            if add_data_source:
                row_to_append["data_source"] = str(doc["article_id"])
            rows.append(row_to_append)

    ds = HFDataset.from_list(rows)

    add_bos_token = False
    if (hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token) or "llama-3.2" in tokenizer_name.lower():
        add_bos_token = True
    # This line makes the tokenizer consistent with dataset-metagradients-jax
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(ex):
        prompt_ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
        completion_ids = tokenizer.encode(ex["completion"], add_special_tokens=False)

        input_ids = prompt_ids + completion_ids
        labels = [tokenizer.pad_token_id]*len(prompt_ids) + completion_ids

        if add_bos_token:
            input_ids = [tokenizer.bos_token_id] + input_ids
            labels = [tokenizer.bos_token_id] + labels
        if "llama-3.2" in tokenizer_name.lower():
            input_ids = input_ids + [tokenizer.eos_token_id]
            labels = labels + [tokenizer.eos_token_id]
        tokenized_ex = {"domain": None, "input_ids": input_ids, "labels": labels[1:] + [tokenizer.pad_token_id]}
        if add_data_source:
            tokenized_ex["data_source"] = ex["data_source"]
        return tokenized_ex

    ds = ds.map(tokenize)

    if shuffle:
        ds = ds.shuffle(seed=42)

    if random_sample_ratio is not None:
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(int(len(ds)*random_sample_ratio)))

    return ds
