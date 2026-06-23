from typing import Dict, Any, Optional, Tuple, Callable, List
import jax
import json
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import numpy as np
import time
import wandb
from dataset_metagradients_jax.utils import filter_and_count_non_finite_dict, tree_statistics, align_opt_state_sharding
from dataset_metagradients_jax.checkpointing import create_checkpointer
from jax.sharding import PartitionSpec as P
import nvtx
from jax._src import mesh as mesh_lib
import inspect


def run_batch_and_enforce_sharding(batch_step, data_weights, train_state, batch_data):
        """ wrapper around run_batch that guarantees that the sharding of the train state remains the same before and after the batch step"""
        current_mesh = mesh_lib.get_abstract_mesh() if mesh_lib.get_concrete_mesh() is not None else mesh_lib.thread_resources.env.physical_mesh
        all_pspecs = nnx.get_named_sharding(train_state, current_mesh)
        out_state, loss = batch_step(data_weights, train_state, batch_data)
        sharded_state = jax.lax.with_sharding_constraint(out_state, all_pspecs)
        return sharded_state, loss

def compute_weighted_loss(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        pad_token_id: int,
        weights: Optional[jnp.ndarray] = None,
        base_loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = optax.softmax_cross_entropy_with_integer_labels,
        reduce: bool = True,
    ) -> jnp.ndarray:
    """Compute weighted loss.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target tokens [batch, seq_len]
        weights: Per-sample weights [batch] or None for uniform weighting
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)

    losses = base_loss_fn(logits_flat, targets_flat)
    losses = losses.reshape(targets.shape)  # [batch_size, seq_len]

    if pad_token_id is not None:
        mask = (targets != pad_token_id).astype(jnp.float32)
        losses = losses * mask

    #losses = jnp.sum(losses, axis=-1)  # Sum over sequence length
    losses = jnp.mean(losses, axis=-1)  # Mean over sequence length

    if weights is not None:
        losses = losses * weights

    if reduce:
        return jnp.mean(losses)
    else:
        return losses

def make_train_functions(model_graphdef: Any, optimizer: optax.GradientTransformation, pad_token_id: int):
    """Takes in the static arguments and returns training functions.
    
    Training operates over single batches, where each batch consists of grad_accum_size microbatches.
    We checkpoint the inside of the batch step, so that during the VJP we only materialize each microbatch worth of data gradients at once.

    Note how we do the VJP in two steps - first propagating back to the summed gradient, and then back to the individual exampeles.
    For the second step, using a JVP is faster for batched inner products with per-example gradients so we have a manual version of that.
    """

    def microbatch_step_full(data_weights, train_state, carry, microbatch_data): 
        """ runs a single microbatch step, accumulating the gradients and the loss """
        grad_buffer, loss_sum = carry 
        model_params, opt_params = train_state
        #inputs = jax.lax.with_sharding_constraint(microbatch_data["input_ids"], P('data', None))
        inputs = microbatch_data["input_ids"]
        targets = microbatch_data["labels"]
        index = microbatch_data["index"]
        weights = data_weights[index]

        def weighted_loss(params):
            model = nnx.merge(model_graphdef, params)
            logits = model(inputs)
            return compute_weighted_loss(logits, targets, pad_token_id, weights)
        
        loss, grads = jax.remat(jax.value_and_grad(weighted_loss, allow_int=True))(model_params)
        # if float0, we assume we are dealing with a grad from an int. We further assume that we don't want to update ints. Just pass g.
        grad_buffer = jax.tree.map(lambda g_new, g: g + g_new if g_new.dtype != jax.dtypes.float0 else g, grads, grad_buffer)
        return (grad_buffer, loss_sum + loss), loss
    
    def compute_grads(data_weights, train_state, batch_data):
        """ computes the average gradient of the loss over the microbatches """
        model_params, opt_params = train_state
        loss_sum = 0.0
        grad_buffer = jax.tree.map(lambda x: jnp.zeros_like(x, shape=x.shape), model_params)
        microbatch_step = jax.tree_util.Partial(microbatch_step_full, data_weights, train_state)
        (grad_buffer, loss_sum), loss = jax.lax.scan(jax.remat(microbatch_step), (grad_buffer, loss_sum), batch_data, unroll=1)
        return grad_buffer, loss_sum/(batch_data["input_ids"].shape[0])
        
    
    def update_with_grads(avg_grad: jnp.ndarray, train_state):
        """ updates the model parameters with the average gradient """
        model_params, opt_params = train_state
        updates, opt_params = optimizer.update(avg_grad, opt_params, model_params)
        model_params = optax.apply_updates(model_params, updates)
        return (model_params, opt_params)
    
    def single_batch_step(data_weights: jnp.ndarray, train_state: Tuple[Any, Any], batch_data: Dict[str, jnp.ndarray]) -> Tuple[Tuple[Any, Any], jnp.ndarray]:
        """Runs for 1 global batch, which is grad_accum_steps * batch_size."""
        grad_buffer, loss = compute_grads(data_weights, train_state, batch_data)
        model_params, opt_params = update_with_grads(grad_buffer, train_state)
        return (model_params, opt_params), loss
    

    def unreduced_microbatch_losses(model_params, data_weights, microbatch_data):
        """ Computes the loss of a single microbatch """
        inputs = jax.lax.with_sharding_constraint(microbatch_data["input_ids"], P('data', None))
        targets = microbatch_data["labels"]
        index = microbatch_data["index"]
        weights = data_weights[index]

        def weighted_loss(params):
            model = nnx.merge(model_graphdef, params)
            logits = model(inputs)
            return compute_weighted_loss(logits, targets, pad_token_id, weights, reduce=False)
        
        example_losses = weighted_loss(model_params)
        return example_losses
        

    def microbatch_vjp_grad_fun(avg_grad_grad: jnp.ndarray, data_weights, train_state, batch_data):
        """ /Manually/ computes the per-example metagrad VJP. 
        Any time we make changes to the updates, we have to check this against the true VJP.
    
        The logic is simple - if we know the gradient wrt the average gradient, then upweighting any individual example's gradient
        has the derivative which is just the inner product of the example's gradient with the average gradient gradient (avg_grad_grad).
        """
   
        model_params, opt_params = train_state
        def jvp_microbatch(microbatch_data):
            recued_microbatch_fn = jax.tree_util.Partial(unreduced_microbatch_losses, data_weights=data_weights, microbatch_data=microbatch_data)
            return jax.jvp(recued_microbatch_fn, (model_params,), (avg_grad_grad,))[1]
        data_weights_list = jax.lax.scan(lambda carry, x: (None, jvp_microbatch(x)), None, batch_data)[1]
        # data_weights_list is now of size (grad_accum_size, batch_size) and need to flatten it.
        # The ordering matches that of flattening the microbatch, but we want to reorder it to match the data_weights, which is controlled by the index
        data_weights_list = data_weights_list.flatten()
        inverse_index = jnp.argsort(batch_data["index"].flatten())
        data_weights_list = data_weights_list[inverse_index]

        return data_weights_list, None, None
    
    
    def run_vjp_update(initial_params, initial_opt_state, data_weights, batch_data, params_grad, opt_grad, grad_sharding):
        """ computes the backwards functions that reverse state->compute_grads->update_with_grads->new_state.
        this computes the branch that does not involve the metagrads themselves  """
        train_state = (initial_params, initial_opt_state)
        current_mesh = mesh_lib.get_abstract_mesh() if mesh_lib.get_concrete_mesh() is not None else mesh_lib.thread_resources.env.physical_mesh # shard opt state across accelerators
        replicated_sharding = jax.sharding.NamedSharding(current_mesh, jax.sharding.PartitionSpec())
        # first, run forward from state -> average grads and do bookkeeping for the later backwards
        def train_state_to_grads_with_sharding(state):
            train_state = jax.lax.with_sharding_constraint(state, replicated_sharding)
            grad_buffer, loss = compute_grads(data_weights, train_state, batch_data)
            grad_buffer = jax.lax.with_sharding_constraint(grad_buffer, replicated_sharding)
            return grad_buffer, loss
        (grad_buffer, loss), vjp_grad_state_fun = jax.vjp(train_state_to_grads_with_sharding, train_state)

        # now the second forward part from avg grads + state -> new state, but this is very memory expensive (each residual is the size of a model) so we shard the train state and re-collect
        grad_buffer = jax.lax.with_sharding_constraint(grad_buffer, grad_sharding)
        _, vjp_update_fun = jax.vjp(update_with_grads, grad_buffer, train_state)  # do computation with sharding
        avg_grad_grad, train_state_grad = vjp_update_fun((params_grad, opt_grad)) 
        
        # re-collect results onto the same device, for computational efficiency in the next step
        avg_grad_grad = jax.lax.with_sharding_constraint(avg_grad_grad, replicated_sharding) 
        # now we do the second backwards through the grads (avg grads -> state)
        train_state_grad_through_grads = vjp_grad_state_fun((avg_grad_grad, 0.0))[0]
        # the state affects the next state via two paths, so sum the backward partials
        train_state_grad = jax.tree.map(lambda x, y: x + y if x.dtype is not jax.dtypes.float0 else x, train_state_grad, train_state_grad_through_grads) #float 0 is the opt state accumulator, so ignore it.
        train_state_grad = jax.lax.with_sharding_constraint(train_state_grad, replicated_sharding)
        return avg_grad_grad, train_state_grad, loss
    
    def run_vjp_update_last_step(initial_params, initial_opt_state, data_weights, batch_data, params_grad, opt_grad, grad_sharding):
        """ computes the backwards functions that reverse state->compute_grads->update_with_grads->new_state.
        this computes the branch that does not involve the metagrads themselves  """
        # TODO: Tons of code duplication here.. refactor one day to merge reused code from run_vjp_update
        train_state = (initial_params, initial_opt_state)
        current_mesh = mesh_lib.get_abstract_mesh() if mesh_lib.get_concrete_mesh() is not None else mesh_lib.thread_resources.env.physical_mesh # shard opt state across accelerators
        replicated_sharding = jax.sharding.NamedSharding(current_mesh, jax.sharding.PartitionSpec())
        # first, run forward from state -> average grads and do bookkeeping for the later backwards
        def train_state_to_grads_with_sharding(state):
            train_state = jax.lax.with_sharding_constraint(state, replicated_sharding)
            grad_buffer, loss = compute_grads(data_weights, train_state, batch_data)
            grad_buffer = jax.lax.with_sharding_constraint(grad_buffer, replicated_sharding)
            return grad_buffer, loss
        (grad_buffer, loss), vjp_grad_state_fun = jax.vjp(train_state_to_grads_with_sharding, train_state)

        # now the second forward part from avg grads + state -> new state, but this is very memory expensive (each residual is the size of a model) so we shard the train state and re-collect
        grad_buffer = jax.lax.with_sharding_constraint(grad_buffer, grad_sharding)
        _, vjp_update_fun = jax.vjp(update_with_grads, grad_buffer, train_state)  # do computation with sharding
        avg_grad_grad, train_state_grad = vjp_update_fun((params_grad, opt_grad)) 
        
        # re-collect results onto the same device, for computational efficiency in the next step
        avg_grad_grad = jax.lax.with_sharding_constraint(avg_grad_grad, replicated_sharding) 
        # in the last step, we can skip the grad-on-grad since there's no next stage that requires these backward components.
        return avg_grad_grad, train_state_grad, loss
    
    def run_vjp_grad_manual(initial_params, initial_opt_state, data_weights, batch_data, avg_grad_grad):
        """ computes the backwards functions that reverse data_weights -> compute grads
        this is the second part that propagates gradients at the batch level to each example and uses a hand computed VJP """
        train_state = (initial_params, initial_opt_state)
        manual_vjp_grad_fun = jax.tree_util.Partial(microbatch_vjp_grad_fun, data_weights=data_weights, train_state=train_state, batch_data=batch_data)
        return manual_vjp_grad_fun(avg_grad_grad)


    def run_vjp_grad_jax(initial_params, initial_opt_state, data_weights, batch_data, avg_grad_grad):
        """ computes the backwards functions that reverse data_weights -> compute grads
        this is the second part that propagates gradients at the batch level to each example. this is a slower, jax computed version to be used for a reference in tests"""
        train_state = (initial_params, initial_opt_state)
        _, vjp_grad_fun, loss = jax.vjp(
            compute_grads, data_weights, train_state, batch_data, has_aux=True
        )
        return vjp_grad_fun(avg_grad_grad)

    return single_batch_step, run_vjp_update, run_vjp_update_last_step, run_vjp_grad_manual, run_vjp_grad_jax

class MemoryEfficientTrainer:
    """Trainer with memory-efficient metagradient support using checkpointing and JVPs."""

    def __init__(
        self,
        model: nnx.Module,
        optimizer: optax.GradientTransformation,
        pad_token_id: int = 0,
        batch_size: int = 1,
        grad_accum_size: int = 1,
        checkpoint_dir: str = "./checkpoints",
        debug: bool = False,
        use_manual_vjp: bool = True,
        normalize_metagrads: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        # Separate wandb runs for train vs val metrics; set by setup_training (None if wandb off).
        self.wandb_run_train = None
        self.wandb_run_val = None
        self.pad_token_id = pad_token_id
        self.grad_accum_size = grad_accum_size
        self.batch_size = batch_size
        self.debug = debug
        self.checkpoint_dir = checkpoint_dir
        self.normalize_metagrads = normalize_metagrads
        self.checkpointer = create_checkpointer(strategy="disk", checkpoint_dir=self.checkpoint_dir)
        model_graphdef, _ = nnx.split(model)
        single_batch_step, run_vjp_update, run_vjp_update_last_step, run_vjp_grad_manual, run_vjp_grad_jax = make_train_functions(model_graphdef, optimizer, pad_token_id)
        if use_manual_vjp: # this should be the faster manual VJP that uses JVPs internally.
            self.run_vjp_grad = run_vjp_grad_manual
        else: # this is the 'exact' VJP chaining approach. used to check correctness.
            self.run_vjp_grad = run_vjp_grad_jax
        self.single_batch_step = single_batch_step
        self.run_vjp_update = run_vjp_update
        self.run_vjp_update_last_step = run_vjp_update_last_step
        # other functions we want to pre-jit
        self.run_batch = jax.jit(jax.tree_util.Partial(run_batch_and_enforce_sharding, self.single_batch_step))

    def _debug_print(self, *args, **kwargs):
        """Helper method for debug printing"""
        if self.debug:
            print(*args, **kwargs)

    def _eval_step(
        self,
        model: nnx.Module,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
    ):
        logits = model(inputs)
        return compute_weighted_loss(logits, targets, self.pad_token_id)

    def prepare_batch(
        self,
        batch: Any,
    ) -> Any:
        """Prepare batches for training
        Each batch from the dataloader contains grad_accum_size microbatches.
        """
        # Each batch is [batch_size * grad_accum_size, seq_len]
        # We need to reshape to [grad_accum_size, batch_size, seq_len]
        total_batch_size = batch["input_ids"].shape[0]
        batch_size = total_batch_size // self.grad_accum_size
        seq_len = batch["input_ids"].shape[1]
        
        batched = {
            "input_ids": batch["input_ids"].reshape(
                self.grad_accum_size, batch_size, seq_len
            ),
            "labels": batch["labels"].reshape(
                self.grad_accum_size, batch_size, seq_len
            ),
            "index": batch["index"].reshape(
                self.grad_accum_size, batch_size
            )
        }
        return batched

    def _reindex_interval_data(
        self,
        batch_data: Dict[str, jnp.ndarray]
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """Re-indexes batch data for data weights"""
        indices_np = np.asarray(batch_data["index"])
        indices_flat = indices_np.flatten()
        unique_indices_np, inverse_indices_np = np.unique(indices_flat, return_inverse=True)
        
        local_indices = jnp.array(inverse_indices_np.reshape(indices_np.shape))
        unique_indices = jnp.array(unique_indices_np)
        
        batch_data_local = batch_data.copy()
        batch_data_local["index"] = local_indices
        return batch_data_local, unique_indices

    def _vjp_update_step(
        self,
        batch_data: Dict[str, jnp.ndarray],
        data_weight_vector: jnp.ndarray,
        initial_params: Any,
        initial_opt_state: Any,
        params_grad: Any,
        opt_grad: Any,
        sharded_update: Any,
        sharded_update_last_step: Any,
        use_wandb: bool = True,
        last_step: bool = False,
        val_mode = False,
    ) -> Tuple[Any, Any, jnp.ndarray, jnp.ndarray]:
        """Performs a single VJP update step, walking backwards from the final iteration to the first, passing partial derivatives backward and computing metagrads for the batch.
        
        Returns:
            Tuple of (updated_params_grad, updated_opt_grad, metagrads_for_batch, unique_indices)
        """
        data_processing_start = time.time()
        
        with nvtx.annotate("train_vjp_data_proc"):
            batch_data_local, unique_indices = self._reindex_interval_data(batch_data)
            interval_weights = data_weight_vector.at[unique_indices].get()
            print(f"Prefetch and reindex took {time.time() - data_processing_start:.3f}s")
                
        vjp_start = time.time()

        with nvtx.annotate("train_vjp_update"):
            if last_step:
                avg_grad_grad, train_state_grad, loss = jax.jit(sharded_update_last_step, donate_argnums=(4, 5))(
                    initial_params, initial_opt_state, interval_weights, 
                    batch_data_local, params_grad, opt_grad
                )
            else:
                avg_grad_grad, train_state_grad, loss = jax.jit(sharded_update, donate_argnums=(4, 5))(
                    initial_params, initial_opt_state, interval_weights, 
                    batch_data_local, params_grad, opt_grad
                )
            params_grad_new, opt_grad_new = train_state_grad

        with nvtx.annotate("train_vjp_grad_compute"):
            metagrads_for_batch_local, _, _ = jax.jit(self.run_vjp_grad, donate_argnums=(0, 1, 4))(
                initial_params, initial_opt_state, interval_weights, 
                batch_data_local, avg_grad_grad
            )

        # Normalize metagrads by standard deviation if requested
        if self.normalize_metagrads:
            metagrads_for_batch_local = metagrads_for_batch_local / (jnp.std(metagrads_for_batch_local) + 1e-15)

        to_log = None

        with nvtx.annotate("train_vjp_rest"):
            self._debug_print(f"Params grad non-finite values: {filter_and_count_non_finite_dict(params_grad_new)}")
            self._debug_print(f"Opt grad non-finite values: {filter_and_count_non_finite_dict(opt_grad_new)}")
            self._debug_print(f"VJP for gradients and metagrads took {time.time() - vjp_start:.3f}s with loss {loss}")

            tree_statistics(params_grad_new, "params_grad")
            
            if use_wandb:
                to_log = {
                    "metagrads/mean" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.mean(metagrads_for_batch_local).item(),
                    "metagrads/std" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.std(metagrads_for_batch_local).item(),
                    "metagrads/max" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.max(metagrads_for_batch_local).item(),
                    "metagrads/min" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.min(metagrads_for_batch_local).item(),
                    "metagrads/processing_time" + ("_val" if val_mode else "") + self.wandb_prefix: time.time() - data_processing_start,
                    "metagrads/examples_per_second" + ("_val" if val_mode else "") + self.wandb_prefix: (self.batch_size * self.grad_accum_size) / (time.time() - data_processing_start),
                    "metagrads/tokens_per_second" + ("_val" if val_mode else "") + self.wandb_prefix: (self.batch_size * self.grad_accum_size * batch_data_local["input_ids"].shape[-1]) / (time.time() - data_processing_start)
                }

        return params_grad_new, opt_grad_new, metagrads_for_batch_local, unique_indices, to_log

    @staticmethod
    def _make_opt_sharding(x, dim):
            current_mesh = mesh_lib.get_abstract_mesh() if mesh_lib.get_concrete_mesh() is not None else mesh_lib.thread_resources.env.physical_mesh
            if x.size > 1024:
                if x.shape[0] % dim == 0:
                    return jax.NamedSharding(current_mesh, P("data"))
                if x.shape[1] % dim == 0:
                    return jax.NamedSharding(current_mesh, P(None, "data"))
                if x.shape[2] % dim == 0:
                    return jax.NamedSharding(current_mesh, P(None, None, "data"))
                raise ValueError(f"Neither first, second, nor third dim of tensor divides {dim}")
            return None


    def forward_pass_with_checkpoints(
        self,
        params: Any,
        data_iterator: Any,
        total_samples: Optional[int] = None,
        data_weights: Optional[jnp.ndarray] = None,
        save_checkpoints: bool = True,
        use_wandb: bool = True,
        grpo_step_for_logging = None,
        val_mode = False,
    ) -> Tuple[Any, Any, int, List[Dict[str, jnp.ndarray]]]:
        """Runs the forward pass, saving checkpoints to disk."""
        if grpo_step_for_logging is None:
            grpo_step_for_logging = 0
        if data_weights is None:
            if total_samples is None:
                raise ValueError("total_samples must be provided if data_weights is None")
            data_weights = jnp.ones(total_samples)

        train_state = (params, self.optimizer.init(params))
        train_state = (params, align_opt_state_sharding(train_state[1], params))

        total_loss = 0.0
        start_time = time.time()
        batch_data_list = []
   
        to_log = [] 

        
        for num_saves, batch_data in enumerate(data_iterator):
            
            save_start = time.time()
            if save_checkpoints:
                self.checkpointer.save(num_saves, train_state)
            save_duration = time.time() - save_start
        
            compute_start = time.time()
            train_state, loss = self.run_batch(
                data_weights, train_state, self.prepare_batch(batch_data)
            )
            batch_data_list.append(self.prepare_batch(batch_data))
            compute_duration = time.time() - compute_start
        
            total_loss += loss
            
            wandb_start = time.time()
            if use_wandb:
                to_log.append({
                    "train/loss" + ("_val" if val_mode else "") + self.wandb_prefix: loss.item(),
                    "train/avg_loss" + ("_val" if val_mode else "") + self.wandb_prefix: (total_loss / (num_saves + 1)).item(),
                    "train/step" + ("_val" if val_mode else "") + self.wandb_prefix: num_saves + 1,
                    "train/batches_seen" + ("_val" if val_mode else "") + self.wandb_prefix: num_saves + 1,
                    "train/tokens_seen" + ("_val" if val_mode else "") + self.wandb_prefix: (num_saves + 1) * self.grad_accum_size * batch_data["input_ids"].shape[-1],
                    "perf/save_duration" + ("_val" if val_mode else "") + self.wandb_prefix: save_duration,
                    "perf/compute_duration" + ("_val" if val_mode else "") + self.wandb_prefix: compute_duration,
                    "perf/time_per_batch" + ("_val" if val_mode else "") + self.wandb_prefix: (time.time() - start_time) / (num_saves + 1),
                    "perf/examples_per_second" + ("_val" if val_mode else "") + self.wandb_prefix: (self.batch_size * self.grad_accum_size) / compute_duration,
                    "perf/tokens_per_second" + ("_val" if val_mode else "") + self.wandb_prefix: (self.batch_size * self.grad_accum_size * batch_data["input_ids"].shape[-1]) / compute_duration,
                })
            wandb_duration = time.time() - wandb_start

            print(f"Batch {num_saves}, loss: {loss}, ckpt_call_time: {save_duration:.3f}s, compute_time: {compute_duration:.3f}s, wandb_time: {wandb_duration:.3f}s")

        final_model_params, final_opt_state = train_state
        num_saves += 1
        if use_wandb:
            for log_index, d in enumerate(to_log):
                step = grpo_step_for_logging*(num_saves+len(batch_data_list)) + log_index
                run = self.wandb_run_val if val_mode else self.wandb_run_train
                run.log({**d, "grpo_step": grpo_step_for_logging}, step=step)
                with open(f"{self.metadata_save_dir}/{str(step)}_forward_val_{str(val_mode)}_{self.wandb_prefix}.json", "w") as f:
                    json.dump(d, f)
                print(f"Logged forward metrics for step {step} (grpo step {grpo_step_for_logging})")
            print(f"Logged {len(to_log)} forward steps to wandb")
        return final_model_params, final_opt_state, num_saves, batch_data_list

    def train(
        self,
        train_dataloader: Any,
        target_metric_fn: Optional[Callable[[Any], jnp.ndarray]] = None,
        target_metric_fn_batches: Optional[List[dict]] = None,
        with_metagrads: bool = True,
        data_weights: Optional[jnp.ndarray] = None,
        use_wandb: bool = True,
        init_params: Optional[Any] = None,
        terminate_after_target_metric = False,
        val_mode = False,
        grpo_step_for_logging = None
    ) -> Dict[str, Any]:
        """Main training loop using memory-efficient backward pass."""
        # In standalone training (no GRPO outer loop) there is no GRPO step to
        # offset wandb logging by; use 0 so the logging-step arithmetic below
        # (and in forward_pass_with_checkpoints) doesn't hit None * int.
        if grpo_step_for_logging is None:
            grpo_step_for_logging = 0
        if init_params is None:
            params = nnx.state(self.model)
        else:
            params = init_params
        model_graphdef, _ = nnx.split(self.model)

        # Do initializations
        self._debug_print("Preparing batches...")
        time_start = time.time()
        total_batches = len(train_dataloader)
        total_samples = total_batches * self.batch_size * self.grad_accum_size
        data_iterator = train_dataloader
        self._debug_print(f"Data iterator prepared in {time.time() - time_start:.3f}s")
        
        if data_weights is None:
            if total_samples is None:
                raise ValueError("total_samples must be provided if data_weights is None")
            data_weight_vector = jnp.ones(total_samples)
        else:
            data_weight_vector = data_weights

        # Run the forward pass, saving checkpoints to disk as we go.
        final_params, final_opt_state, num_batches, batch_data_list = self.forward_pass_with_checkpoints(
            params, data_iterator, total_samples, data_weight_vector, save_checkpoints=with_metagrads, use_wandb=use_wandb, val_mode=val_mode, grpo_step_for_logging=grpo_step_for_logging
        )
        layout = (final_params, final_opt_state)

        if not with_metagrads:
            final_model = nnx.merge(model_graphdef, final_params)
            self.checkpointer.close()
            return {"final_model": final_model, "final_data_weights": None}
        
        # Async load the next checkpoint needed for the first iteration of the VJP phase. 
        ckpt_future = self.checkpointer.restore_async(num_batches - 1, layout)
    
        print("Now switching to the metagrads VJP phase")
        
        # Setup for the VJP.
        metagrads = jnp.zeros_like(data_weight_vector)
        self.model = nnx.merge(model_graphdef, final_params)

        # Get gradient of the target metric with respect to final parameters
        sig = inspect.signature(target_metric_fn).parameters

        # All the "maybe" arguments you could pass, with their values
        possible_kwargs = {
            "val_mode": val_mode,
            # The target_metric_fn gets a REMAPPED step (raw GRPO step expanded into the
            # per-inner-metagrad-step wandb timeline), not the raw GRPO step.
            "remapped_step_for_logging": grpo_step_for_logging*(num_batches+len(batch_data_list))+len(batch_data_list)-1,
            # ...and the raw GRPO step, so target_metric_fns that log proxies (pixel_accuracy /
            # lm_head_norm) can plot them against the comparable grpo_step axis.
            "grpo_step_for_logging": grpo_step_for_logging,
            # add more optional args here later if you want
        }

        # Filter down to only the ones that target_metric_fn actually accepts
        extra_kwargs = {
            name: value
            for name, value in possible_kwargs.items()
            if name in sig
        }

        if target_metric_fn_batches is None:
            # In this case, we assume the target_metric_fn is self-contained
            def target_metric_wrapper(params, opt_state):
                return target_metric_fn(nnx.merge(model_graphdef, params), **extra_kwargs)

            target_metric, (params_grad, opt_grad) = jax.jit(jax.value_and_grad(target_metric_wrapper, argnums=(0, 1), allow_int=True))(final_params, final_opt_state)
            print("returned target metric")
        else:
            def target_metric_wrapper(params, opt_state, batch):
                return target_metric_fn(nnx.merge(model_graphdef, params), batch, **extra_kwargs)

            # Get gradient of the target metric with respect to final parameters
            num_target_metric_batches = len(target_metric_fn_batches)
            target_metric_sum = None
            final_grad_fn = jax.jit(jax.value_and_grad(target_metric_wrapper, argnums=(0, 1), allow_int=True))

            for batch in target_metric_fn_batches:
                target_metric, (params_grad, opt_grad) = final_grad_fn(final_params, final_opt_state, batch)

                if target_metric_sum is None:
                    target_metric_sum = target_metric
                    params_grad_sum = params_grad
                    opt_grad_sum = opt_grad
                else:
                    target_metric_sum += target_metric
                    params_grad_sum = jax.tree_util.tree_map(lambda a, b: a + b if a.dtype != jax.dtypes.float0 else a, params_grad_sum, params_grad)
                    opt_grad_sum = jax.tree_util.tree_map(lambda a, b: a + b if a.dtype != jax.dtypes.float0 else a, opt_grad_sum, opt_grad)

            target_metric = target_metric_sum / num_target_metric_batches
            params_grad = jax.tree_util.tree_map(lambda a: a / num_target_metric_batches if a.dtype != jax.dtypes.float0 else a, params_grad_sum)
            opt_grad = jax.tree_util.tree_map(lambda a: a / num_target_metric_batches if a.dtype != jax.dtypes.float0 else a, opt_grad_sum)

        print(f"Final target metric (primal, higher=better): {target_metric}")
        if use_wandb:
            # The target metric (target_metric_fn value) is now higher=better, so this rises as
            # training improves. Distinct from the RL reward, which is its metagradient.
            # Key is "target_metric/value" (not bare "target_metric") so wandb groups it in the
            # same "target_metric/" section as the readable proxies (target_metric/pixel_accuracy,
            # target_metric/lm_head_norm) instead of dropping it in the default "Charts" section.
            d = {"target_metric/value" + ("_val" if val_mode else "") + self.wandb_prefix: target_metric.item()}
            step = grpo_step_for_logging*(num_batches+len(batch_data_list))+len(batch_data_list)-1
            run = self.wandb_run_val if val_mode else self.wandb_run_train
            run.log({**d, "grpo_step": grpo_step_for_logging}, step=step)
            with open(f"{self.metadata_save_dir}/{str(step)}_target_metric_val_{str(val_mode)}_{self.wandb_prefix}.json", "w") as f:
                json.dump(d, f)
        if terminate_after_target_metric:
            return {"final_target_metric": target_metric}
        self.checkpointer.wait_until_finished()
        initial_params, initial_opt_state = ckpt_future.result()
        # Define a sharding plan for the gradients that will be used to save memory during the VJP.
        grad_sharding = jax.tree.map(lambda x: MemoryEfficientTrainer._make_opt_sharding(x, self.model.dim), params_grad)
        sharded_update = jax.tree_util.Partial(self.run_vjp_update, grad_sharding=grad_sharding)
        sharded_update_last_step = jax.tree_util.Partial(self.run_vjp_update_last_step, grad_sharding=grad_sharding)

        # Walk backwards through the batches, computing metagrads
        nvtx_iter = 7
        for log_index, i in enumerate(reversed(range(num_batches))):
            print(f"Start VJP for batch {i}")
            
            batch_data = batch_data_list[i]
            if i > 0:  # prefetch next checkpoint we will need
                ckpt_future = self.checkpointer.restore_async(i - 1, layout)
            
            # Use context manager for nvtx annotation
            if i == nvtx_iter:
                nvtx.push_range("train_vjp_step")
            # Perform VJP update step
            params_grad, opt_grad, metagrads_for_batch_local, unique_indices, to_log = self._vjp_update_step(
                batch_data=batch_data,
                data_weight_vector=data_weight_vector,
                initial_params=initial_params,
                initial_opt_state=initial_opt_state,
                params_grad=params_grad,
                opt_grad=opt_grad,
                sharded_update=sharded_update,
                sharded_update_last_step=sharded_update_last_step,
                use_wandb=use_wandb,
                last_step=(i == 0),
                val_mode=val_mode,
            )
            if use_wandb:
                step = grpo_step_for_logging*(num_batches+len(batch_data_list))+log_index+len(batch_data_list)-1
                run = self.wandb_run_val if val_mode else self.wandb_run_train
                run.log({**to_log, "grpo_step": grpo_step_for_logging}, step=step)
                with open(f"{self.metadata_save_dir}/{str(step)}_vjp_val_{str(val_mode)}_{self.wandb_prefix}.json", "w") as f:
                    json.dump(to_log, f)
            
            # Accumulate metagrads by scattering local grads into the global tensor
            metagrads = metagrads.at[unique_indices].add(metagrads_for_batch_local)
            
            # Update initial state for next iteration from prefetch
            block_start = time.time()
            if i > 0:
                self.checkpointer.wait_until_finished()
                initial_params, initial_opt_state = ckpt_future.result()
            block_duration = time.time() - block_start
            self._debug_print(f"Ckpt load for batch {i} had overhead {block_duration:.3f}s")

        # Load the final model and opt state
        final_params, final_opt_state = layout
        final_model = nnx.merge(model_graphdef, final_params)
        self.checkpointer.wait_until_finished()
        self.checkpointer.close()
            
        self._debug_print(f"Final weights: {metagrads}")
        
        # Log final metagradient statistics
        if use_wandb:
            step = grpo_step_for_logging*(num_batches+len(batch_data_list))+num_batches+len(batch_data_list)-1
            d = {"metagrads/final_mean" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.mean(metagrads).item(),
                 "metagrads/final_std" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.std(metagrads).item(),
                 "metagrads/final_max" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.max(metagrads).item(),
                 "metagrads/final_min" + ("_val" if val_mode else "") + self.wandb_prefix: jnp.min(metagrads).item()}
            run = self.wandb_run_val if val_mode else self.wandb_run_train
            run.log({**d, "grpo_step": grpo_step_for_logging}, step=step)
            with open(f"{self.metadata_save_dir}/{str(step)}_metagrads_final_val_{str(val_mode)}_{self.wandb_prefix}.json", "w") as f:
                json.dump(d, f)
        
        return {
            "final_model": final_model,
            "final_data_weights": metagrads,
            "final_target_metric": float(target_metric),
        }
