import optax
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src import utils
from optax._src import numerics
from typing import Optional, Union, Any, Callable, NamedTuple
import jax.numpy as jnp
import jax


def adamw_reparam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformationExtraArgs:
  r"""Adam with weight decay regularization - copied from optax.adamw to hack on
   This saves the second moment in log space, which makes the gradient computations more accurate.

   NOTE: the paper experiments do not use this; they use stock optax.adamw (optimizer_type
   'adamw') / optax.sgd. adamw_reparam is kept here for future development and hacking.
  """
  return combine.chain(
      scale_by_adam_reparam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      transform.add_decayed_weights(weight_decay, mask),
      transform.scale_by_learning_rate(learning_rate),
  )

class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm. - copied from optax.ScaleByAdamState"""

  count: int  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  log_nu: base.Updates

def scale_by_adam_reparam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[jnp.dtype] = None,
    log_nu_dtype: Optional[jnp.dtype] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Adam algorithm - copied from optax.scale_by_adam  """

  
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  log_nu_dtype = utils.canonicalize_dtype(log_nu_dtype)

  def init_fn(params):
    if eps_root > 0.0:
      print('WARNING: eps_root > 0.0 is not supported for reparam Adam, ignoring')
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    log_nu = optax.tree.zeros_like(params, dtype=log_nu_dtype)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, log_nu=log_nu)
  
  def _moments(updates, state_mu, state_log_nu):
    """First and second moments (≈ 3× param size)."""
    mu = optax.tree.update_moment(updates, state_mu, b1, 1)
    cur_nu = jax.tree.map(jnp.exp, state_log_nu)
    nu = optax.tree.update_moment_per_elem_norm(updates, cur_nu, b2, 2)
    log_nu = jax.tree.map(jnp.log, nu)
    return mu, log_nu

  def _scale_and_apply(mu, log_nu, count_inc):
    """Bias correction, scaling and final multiply (≈ 2× param size)."""
    #mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
    mu_hat = mu # dont do bias correction - its faster and a bit more stable..
    bias = jnp.log1p(-b2 ** count_inc)
    #nu_hat = jax.tree.map(lambda t: jnp.exp(t - bias_correction_.astype(t.dtype)), log_nu)
    # nu hat is exp(t-bias), and the scaling function is 1/(sqrt(nu_hat)+eps)
    scale = jax.tree.map(
        lambda t: jnp.exp(-jnp.logaddexp(jnp.log(eps), (t - bias) / 2)), log_nu
    )
    return jax.tree.map(lambda m, s: m * s, mu_hat, scale)

  def update_fn(updates, state, params=None):
    del params
    count_inc = numerics.safe_increment(state.count)
    mu, log_nu = jax.remat(_moments)(updates, state.mu, state.log_nu)
    updates   = jax.remat(_scale_and_apply)(mu, log_nu, count_inc)
    return updates, ScaleByAdamState(count=count_inc, mu=optax.tree.cast(mu, mu_dtype), log_nu=optax.tree.cast(log_nu, log_nu_dtype))

  

  return base.GradientTransformation(init_fn, jax.remat(update_fn))


