#%%
import os
os.chdir("/home/aiops/max/offrl")
os.environ['SOTA_API_KEY'] = ''

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from algos.model import *
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from utilities.utils import set_random_seed
from utilities.jax_utils import next_rng, value_and_multi_grad
import tqdm
import distrax
from tensorflow_probability.substrates import jax as tfp
from copy import deepcopy

set_random_seed(42)
# %%
rhos = set([-0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.3, -0.1, 0])
# rhos = set([-0.7, -0.6, -0.5, -0.3, -0.1, 0])
for r in deepcopy(rhos):
  rhos.add(-r)

n_rhos = 19
rhos = np.linspace(-0.95, 0.95, n_rhos)

data_dims = [64, 96, 128]
data_dims = [2, 16, 32]

if os.path.isfile('res_pytorch.pkl'):
  with open('res_pytorch.pkl', 'rb') as fin:
    import pickle
    results = pickle.load(fin)
  
  for d_dim in data_dims:
    for method in results.keys():
      vals = results[method][d_dim]
      rhos = sorted(rhos)
      vals = [results[method][d_dim][r] for r in rhos]
      plt.plot(list(rhos), vals, label=method)
    
    plt.legend(loc='lower right')
    plt.title(f"data dim = {d_dim}")
    plt.tight_layout()
    plt.show()
  
# %%

n_rhos = 19
rhos = np.linspace(-0.95, 0.95, n_rhos)

data_dims = [2, 16, 32]

if os.path.isfile('res_pytorch.pkl'):
  with open('res_pytorch.pkl', 'rb') as fin:
    import pickle
    results = pickle.load(fin)
  
  new_res = dict()
  for k, v in results.items():
    if k == 'real':
      continue
    
    d = int(k.split('-')[0])
    name = '_'.join(k.split('-')[1:])
    if d not in new_res:
      new_res[d] = dict()
    
    new_res[d][name] = v
  
  for d in data_dims:
    real = []
    for rho in rhos:
      cov = np.eye(2*d)
      cov[d:2*d, 0:d] = rho * np.eye(d) 
      cov[0:d, d:2*d] = rho * np.eye(d)
      real_val = - 0.5 * np.log(np.linalg.det(cov)) / np.log(2)
      real.append(real_val)
    new_res[d]['real'] = np.array(real)
  
  for d_dim in data_dims:
    for k, v in new_res[d].items():
      plt.plot(rhos, v, label=k)

    plt.legend(loc='lower right')
    plt.title(f"data dim = {d_dim}")
    plt.tight_layout()
    plt.show()

# %%

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128):
  """Generate samples from a correlated Gaussian distribution."""
  x, eps = np.split(np.random.normal(size=(batch_size, 2 * dim)), 2, axis=1)
  y = rho * x + np.sqrt(1. - rho**2) * eps
  dist = distrax.MultivariateNormalDiag(
    np.zeros_like(x), np.ones_like(x)
  )
  log_y = dist.log_prob(y)
  return x, y, np.array(log_y)

def rho_to_mi(dim, rho):
  return -0.5  * np.log(1-rho**2) * dim

def mi_to_rho(dim, mi):
  return np.sqrt(1-np.exp(-2.0 / dim * mi))

# def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128):
#   """Generate samples from a correlated Gaussian distribution."""
#   d = dim
#   cov = np.eye(2*d) 
#   cov[d:2*d, 0:d] = rho * np.eye(d) 
#   cov[0:d, d:2*d] = rho * np.eye(d)
#   f = tfp.distributions.MultivariateNormalFullCovariance(
#     jnp.zeros(2 * d), jnp.array(cov, dtype=jnp.float32) 
#   )
#   Z = f.sample((batch_size,), next_rng())
#   x, y = Z[:,:d], Z[:,d:2*d]
#   log_y = f.log_prob(Z)
#   return x, y, np.array(log_y)

# def rho_to_mi(d, rho):
#   cov = np.eye(2*d)
#   cov[d:2*d, 0:d] = rho * np.eye(d) 
#   cov[0:d, d:2*d] = rho * np.eye(d)
#   return - 0.5 * np.log(np.linalg.det(cov)) / np.log(2)

# def mi_to_rho(dim, mi):
#   return np.sqrt(1-np.exp(-2.0 / dim * mi))


# %%

config = ConfigDict()
config.arch = '256-256'
config.lr = 5e-4
config.optimizer_type = 'adam'
config.n_samples = 64
config.seed = 42
config.dataset_size = 100000
config.batch_size = 256
config.data_dim = 1
config.rho = 0.9
config.train_steps = 1000
config.ibal = False
config.unbiased_grad = False
config.mcmc_num_leapfrog_steps = 2
config.mcmc_burnin_steps = 5
config.mcmc_step_size = 1
config.unbiased_weight = 1
config.infonce = True
config.gmine = False
config.fmine = False
config.mcmc = False
 


#%%

class BarberAgakovBound:
  def __init__(self, config) -> None:
    self.config = config
    self.policy = GaussianPolicy(
      self.config.data_dim,
      self.config.data_dim,
      self.config.data_dim,
      self.config.arch,
    )

    optimizer_class = getattr(
      optax, self.config.optimizer_type
    )

    self._train_states = dict()
    policy_params = self.policy.init(
      next_rng(), next_rng(), jnp.zeros((10, self.policy.embedding_dim))
    )
    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=optimizer_class(self.config.lr),
      apply_fn=None
    )

    self._model_keys = ('policy',)
  
  def train(self, batch):
    self._train_states, metrics = self._train_step(
      self._train_states, next_rng(), batch
    )
    return metrics
  
  @partial(jax.jit, static_argnames=('self',))
  def _train_step(self, train_states, rng, batch):
    train_params = {key: train_states[key].params for key in self._model_keys}
    states = batch['states']
    actions = batch['actions']
    log_as = batch['log_as']

    def loss_fn(train_params, rng):
      loss_collection = {}

      log_prob = self.policy.apply(
        train_params['policy'],
        states,
        actions,
        method=self.policy.log_prob
      )

      policy_loss = - log_prob.mean()
      mi_estimation = (log_prob - log_as).mean()

      loss_collection['policy'] = policy_loss
      return tuple(loss_collection[key] for key in self._model_keys), locals()
    
    train_params = {
      key: train_states[key].params for key in self._model_keys
    }
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self._model_keys), has_aux=True
    )(train_params, rng)

    new_train_states = {
      key: train_states[key].apply_gradients(grads=grads[i][key])
      for i, key in enumerate(self._model_keys)
    }
    metrics = dict(
      policy_loss = aux_values['policy_loss'],
      mi_estimation=aux_values['mi_estimation'],
      log_prob=aux_values['log_prob']
    )
    return new_train_states, metrics


# %%

class MineBound:
  def __init__(self, config) -> None:
    self.config = config
    self.qf = FullyConnectedQFunction(
      self.config.data_dim,
      self.config.data_dim,
      self.config.arch,
    )
    self.policy = GaussianPolicy(
      self.config.data_dim,
      self.config.data_dim,
      self.config.data_dim,
      self.config.arch,
    )

    optimizer_class = getattr(
      optax, self.config.optimizer_type
    )

    self._train_states = dict()
    qf_params = self.qf.init(
      next_rng(), jnp.zeros((10, self.config.data_dim)),
      jnp.zeros((10, self.config.data_dim))
    )
    self._train_states['qf'] = TrainState.create(
      params=qf_params,
      tx=optimizer_class(self.config.lr),
      apply_fn=None
    )
    policy_params = self.policy.init(
      next_rng(), next_rng(), jnp.zeros((10, self.policy.embedding_dim))
    )
    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=optimizer_class(self.config.lr),
      apply_fn=None
    )

    self._model_keys = ('qf', 'policy')
  
  def train(self, batch):
    self._train_states, metrics = self._train_step(
      self._train_states, next_rng(), batch
    )
    return metrics
  
  @partial(jax.jit, static_argnames=('self',))
  def _train_step(self, train_states, rng, batch):
    train_params = {key: train_states[key].params for key in self._model_keys}
    states = batch['states']
    actions = batch['actions']
    log_as = batch['log_as']

    def loss_fn(train_params, rng):
      loss_collection = {}

      q_pred = self.qf.apply(
        train_params['qf'],
        states,
        actions
      )
      rng, split_rng = jax.random.split(rng)
      new_actions, new_act_log_prob = self.policy.apply(
        train_params['policy'],
        split_rng,
        states,
        repeat=self.config.n_samples
      )
      neg_q_pred = self.qf.apply(
        train_params['qf'],
        states,
        new_actions,
      )

      log_prob = self.policy.apply(
        train_params['policy'],
        states,
        actions,
        method=self.policy.log_prob
      )

      unnormalized_log_prob = q_pred - jax.scipy.special.logsumexp(neg_q_pred, axis=-1)

      if self.config.infonce and not self.config.ibal:
        tiled_action = jnp.repeat(
          jnp.expand_dims(actions, axis=1),
          self.config.batch_size,
          axis=1
        )
        tiled_states = jnp.repeat(
          jnp.expand_dims(states, axis=0),
          self.config.batch_size,
          axis=0
        )

        tiled_q = self.qf.apply(
          train_params['qf'],
          tiled_states,
          tiled_action,
        )
        pos_q = jnp.diagonal(tiled_q)
        neg_q = jax.scipy.special.logsumexp(
          tiled_q, axis=0
        )
        unnormalized_log_prob = pos_q - neg_q
      
      elif self.config.gmine and self.config.ibal:
        unnormalized_log_prob = q_pred - jax.scipy.special.logsumexp(neg_q_pred)
      
      elif self.config.fmine and self.config.ibal:
        unnormalized_log_prob = q_pred - jnp.exp(neg_q_pred - 1).mean()

      policy_loss = - log_prob.mean()
      loss_collection['policy'] = policy_loss

      qf_loss = -unnormalized_log_prob.mean()
      mi_estimation = (unnormalized_log_prob).mean()

      if (self.config.ibal and not self.config.fmine) or (self.config.infonce and not self.config.ibal):
        mi_estimation += jnp.log(self.config.batch_size)

      if self.config.ibal:
        barber_agakov = (log_prob - log_as).mean()
        mi_estimation += barber_agakov

      loss_collection['qf'] = qf_loss

      if self.config.unbiased_grad and self.config.ibal:
        if self.config.mcmc:
          rng, split_rng = jax.random.split(rng)
          def log_prob_fn(x):
            action_prob = self.policy.apply(
              train_params['policy'],
              states,
              x,
              method=self.policy.log_prob
            )
            q = self.qf.apply(
              train_params['qf'], states, x
            )

            return action_prob + q

          num_results = self.config.n_samples
          num_burnin_steps = self.config.mcmc_burnin_steps
          adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
              tfp.mcmc.HamiltonianMonteCarlo(
                  target_log_prob_fn=log_prob_fn,
                  num_leapfrog_steps=self.config.mcmc_num_leapfrog_steps,
                  step_size=self.config.mcmc_step_size),
              num_adaptation_steps=int(num_burnin_steps * 0.8))
    
          rng, split_rng = jax.random.split(rng)
          mcmc_action_samples, _ = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=actions,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=split_rng)
        
          mcmc_action_samples = jax.lax.stop_gradient(
            jnp.transpose(
              mcmc_action_samples, (1, 0, 2)
            )
          )

          correction_log_pis = self.policy.apply(
            train_params['policy'],
            states,
            mcmc_action_samples,
            method=self.policy.log_prob
          )
          loss_collection['policy'] += correction_log_pis.mean() * self.config.unbiased_weight
        
        else:
          loss_collection['policy'] += loss_collection['qf']

      return tuple(loss_collection[key] for key in self._model_keys), locals()
    
    train_params = {
      key: train_states[key].params for key in self._model_keys
    }
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self._model_keys), has_aux=True
    )(train_params, rng)

    new_train_states = {
      key: train_states[key].apply_gradients(grads=grads[i][key])
      for i, key in enumerate(self._model_keys)
    }
    metrics = dict(
      policy_loss = aux_values['policy_loss'],
      qf_loss=aux_values['qf_loss'],
      mi_estimation=aux_values['mi_estimation'],
      log_prob=aux_values['log_prob']
    )
    if self.config.ibal:
      metrics['barber_agakov'] = aux_values['barber_agakov']
      metrics['unnormalized'] = aux_values['unnormalized_log_prob'].mean()
    return new_train_states, metrics

# %%

states, actions, log_as = sample_correlated_gaussian(
  rho=config.rho,
  dim=config.data_dim,
  batch_size=config.dataset_size
)

ba_bound = BarberAgakovBound(config)
mine_bound = MineBound(config)

bound = mine_bound

for step in tqdm.tqdm(range(config.train_steps)):
  indices = np.random.randint(
    config.dataset_size, size=config.batch_size
  )
  s, a, log_a = states[indices], actions[indices], log_as[indices]
  batch = dict(
    states=s,
    actions=a,
    log_as=log_a
  )
  metrics = bound.train(batch)

print(metrics['mi_estimation'])
if config.ibal:
  print(metrics['barber_agakov'])
  print(metrics['unnormalized'])

print(rho_to_mi(config.data_dim, config.rho))
# %%

# def compute_mutual_info(estimator_fn, config):
#   states, actions, log_as = sample_correlated_gaussian(
#     rho=config.rho,
#     dim=config.data_dim,
#     batch_size=config.dataset_size,
#   )

#   for step in tqdm.tqdm(range(config.train_steps)):
#     indices = np.random.randint(
#       config.dataset_size, size=config.batch_size
#     )
#     s, a, log_a = states[indices], actions[indices], log_as[indices]
#     batch = dict(
#       states=s,
#       actions=a,
#       log_as=log_a
#     )
#     metrics = estimator_fn.train(batch)
  
#   return metrics


# cfgs = []
# rhos = set([-0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.3, -0.1, 0])
# for r in deepcopy(rhos):
#   rhos.add(-r)

# data_dims = [64, 96, 128]

# results = dict(
#   gts=dict(),
#   ba=dict(),
#   ibal=dict(),
#   infonce=dict(),
#   fmine=dict(),
#   gmine=dict(),
# )

# def get_model(rho, data_dim, name):
#   cur_cfg = deepcopy(config)
#   cur_cfg['rho'] = rho
#   cur_cfg['data_dim'] = data_dim

#   if name == 'ba':
#     model = BarberAgakovBound(cur_cfg)
#   elif name == 'ibal':
#     update_cfg = dict(
#       ibal=True,
#       infonce=False,
#       gmine=False,
#       fmine=False
#     )
#     cur_cfg.update(update_cfg)
#     model = MineBound(cur_cfg)
#   elif name == 'infonce':
#     update_cfg = dict(
#       ibal=False,
#       infonce=True,
#       fmine=False,
#       gmine=False,
#     )
#     cur_cfg.update(update_cfg)
#     model = MineBound(cur_cfg)
#   elif name == 'fmine':
#     update_cfg = dict(
#       ibal=True,
#       infonce=False,
#       fmine=True,
#       gmine=False,
#     )
#     cur_cfg.update(update_cfg)
#     model = MineBound(cur_cfg)
#   elif name == 'gmine':
#     update_cfg = dict(
#       ibal=True,
#       infonce=False,
#       fmine=False,
#       gmine=True,
#     )
#     cur_cfg.update(update_cfg)
#     model = MineBound(cur_cfg)
  
#   return model

# def train(model, rho, d_dim):
#   states, actions, log_as = sample_correlated_gaussian(
#     rho=rho,
#     dim=d_dim,
#     batch_size=config.dataset_size
#   )
 
#   for step in tqdm.tqdm(range(config.train_steps)):
#     indices = np.random.randint(
#       config.dataset_size, size=config.batch_size
#     )
#     s, a, log_a = states[indices], actions[indices], log_as[indices]
#     batch = dict(
#       states=s,
#       actions=a,
#       log_as=log_a
#     )
#     metrics = model.train(batch)
  
#   return metrics['mi_estimation']


# gts = dict()
# ba_results = dict()
# ibal_results = dict()
# infonce_results = dict()

# for d_dim in data_dims:
#   for r in rhos:
#     if d_dim not in gts:
#       gts[d_dim] = dict()
    
#     if d_dim not in results['gts']:
#       results['gts'][d_dim] = dict()
#     results['gts'][d_dim][r] = rho_to_mi(d_dim, r)
    
#     for method in ['ba', 'ibal', 'infonce', 'fmine', 'gmine']:
#       model = get_model(r, d_dim, method)
#       mi_estimation = train(model, r, d_dim)
#       if d_dim not in results[method]:
#         results[method][d_dim] = dict()
#       results[method][d_dim][r] = float(mi_estimation)

# with open('res.pkl', 'wb') as fout:
#   import pickle
#   pickle.dump(results, fout, pickle.HIGHEST_PROTOCOL)

# %%
