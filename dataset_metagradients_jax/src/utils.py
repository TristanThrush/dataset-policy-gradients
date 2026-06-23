import jax
import jax.numpy as jnp
from typing import Any, Dict
from jax._src import mesh as mesh_lib


def align_opt_state_sharding(opt_state: Any, param_state: Any) -> Any:
    """Fixes a (somewhat) bug in optimizer init, where the iteration counter ignores the mesh and is always single device
    this crashes the VJP, so we force the iteration 0 opt_state to be on the same device as the params.

    Args:
        opt_state: Optimizer state pytree
        param_state: Parameter state pytree

    Returns:
        Optimizer state with aligned sharding
    """
    try:

        current_mesh = mesh_lib.get_abstract_mesh() if mesh_lib.get_concrete_mesh() is not None else mesh_lib.thread_resources.env.physical_mesh
        replicated_sharding = jax.sharding.NamedSharding(current_mesh, jax.sharding.PartitionSpec())

        def align_leaf_sharding(opt_leaf):
            if hasattr(opt_leaf, 'sharding'):
                leaf_sharding = opt_leaf.sharding
                # Check if the leaf's sharding uses the current mesh
                if not hasattr(leaf_sharding, 'mesh') or leaf_sharding.mesh != current_mesh:
                    print(f"WARNING: Detected sharding mismatch for {opt_leaf}, aligning to {current_mesh}")
                    return jax.device_put(opt_leaf, replicated_sharding)
            return opt_leaf

        return jax.tree_util.tree_map(align_leaf_sharding, opt_state)
    except RuntimeError:
        # No mesh context, return opt_state as is
        return opt_state


def mean_norm_of_tree(tree):
    """Calculate mean norm across all arrays in a pytree."""
    norms = jax.tree_util.tree_map(
        lambda x: jnp.linalg.norm(x) if isinstance(x, jnp.ndarray) else 0.0,
        tree
    )
    total_norm = jax.tree_util.tree_reduce(lambda x, y: x + y, norms, 0.0)
    num_leaves = len(jax.tree_util.tree_leaves(tree))
    return total_norm / num_leaves if num_leaves > 0 else 0.0

def tree_statistics(tree, name: str = "tree"):
    """Calculate and print comprehensive statistics for a pytree.

    Args:
        tree: The pytree to analyze
        name: Name to use in the printed output
    """
    # Flatten all arrays in the tree
    arrays = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jnp.ndarray)]

    if not arrays:
        print(f"{name}: No arrays found")
        return

    # Concatenate all arrays into one for global statistics
    all_values = jnp.concatenate([x.flatten() for x in arrays])

    # Calculate statistics
    mean_val = jnp.nanmean(all_values)
    var_val = jnp.nanvar(all_values)
    num_nans = jnp.sum(jnp.isnan(all_values))
    num_pos_infs = jnp.sum(jnp.isposinf(all_values))
    num_neg_infs = jnp.sum(jnp.isneginf(all_values))
    total_elements = len(all_values)
    mean_norm = mean_norm_of_tree(tree)

    print(f"{name} - Mean: {mean_val}, Var: {var_val}, NaNs: {num_nans}/{total_elements}, +Infs: {num_pos_infs}/{total_elements}, -Infs: {num_neg_infs}/{total_elements}, Mean norm: {mean_norm}")

def filter_and_count_non_finite_dict(pytree: Any) -> Dict[str, str]:
    """
    Goes through a pytree, counts non-finite values (NaN + inf) in each element.
    Returns a dictionary mapping element paths to counts in detailed format,
    only including elements that contain non-finite values.

    Args:
        pytree: Input pytree to process

    Returns:
        Dictionary mapping path strings to formatted strings for elements with non-finite values,
        e.g., "NaN:2/1000, +Inf:1/1000, -Inf:0/1000"
    """
    # Get the flattened tree with paths
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)

    non_finite_counts = {}

    for path, leaf in leaves:
        # Convert to JAX array if needed
        if not isinstance(leaf, jnp.ndarray):
            try:
                leaf = jnp.asarray(leaf)
            except:
                continue

        # Skip non-float types that can't have NaNs/infs
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            continue

        # Count non-finite values and total elements
        nan_count = jnp.sum(jnp.isnan(leaf)).item()
        pos_inf_count = jnp.sum(jnp.isposinf(leaf)).item()
        neg_inf_count = jnp.sum(jnp.isneginf(leaf)).item()
        total_count = leaf.size

        # Only store if there are non-finite values
        if nan_count > 0 or pos_inf_count > 0 or neg_inf_count > 0:
            # Convert path to string representation
            path_str = '.'.join([str(p.key) if hasattr(p, 'key') else str(p) for p in path])
            non_finite_counts[path_str] = f"NaN:{nan_count}/{total_count}, +Inf:{pos_inf_count}/{total_count}, -Inf:{neg_inf_count}/{total_count}"

    return non_finite_counts
