import numpy as np
from symbol import compound_stmt
import torch


class RandSampler(object):
    def __init__(self, max_size: int, batch_size: int = 1) -> None:
        self._max_size = max_size
        self._batch_size = batch_size

    def sample(self):
        return np.random.randint(self._max_size, size=self._batch_size)


class PrefetchBalancedSampler(object):
    """A prefetch balanced sampler."""
    def __init__(self, probs, max_size: int, batch_size: int, n_prefetch: int) -> None:
        self._max_size = max_size
        self._batch_size = batch_size
        self.n_prefetch = min(n_prefetch, max_size//batch_size)
        self._probs = probs.squeeze() / np.sum(probs)
        self.cnt = self.n_prefetch - 1

    def sample(self):
        self.cnt = (self.cnt+1)%self.n_prefetch
        if self.cnt == 0:
            self.indices = np.random.choice(self._max_size, 
            size=self._batch_size * self.n_prefetch, p=self._probs)
        return self.indices[self.cnt*self._batch_size : (self.cnt+1)*self._batch_size]

    def replace_prob(self, probs):
        self._probs = probs.squeeze() / np.sum(probs)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, batch_size, max_size=int(1e6), 
                base_prob=0.0, resample=False, reweight=False, n_step=3, discount=0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.batch_size = batch_size
        self.base_prob = base_prob
        self.resample = resample
        self.reweight = reweight
        self.n_step = n_step
        self.discount = discount


        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # use for bc advantage
    def sample_by_ind(self, ind):
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.dones_float[ind]).to(self.device),
            torch.FloatTensor(self.ret[ind]).to(self.device)
        )

    def bc_eval_sample(self):
        # sample by the distribution of rebalanced behavior policy
        ind = self.bc_sampler.sample()
        return self.sample_by_ind(ind)

    def sample_n_step_by_ind(self, ind):
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.state_n[ind]).to(self.device),
            torch.FloatTensor(self.ret_n[ind]).to(self.device),
            torch.FloatTensor(self.done_n[ind]).to(self.device),
        )

    def bc_eval_sample_n(self):
        ind = self.bc_sampler.sample()
        return self.sample_n_step_by_ind(ind)


    # use for training
    def sample(self):
        ind = self.sampler.sample()
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.weights[ind]).to(self.device)
        )
    
    
    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]
        # compute time limit
        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1
        self.dones_float = dones_float.reshape(-1,1) # time limit truncated or terminal state

        # discounted return-to-go
        ret = np.zeros((self.size+1, 1))
        for t in reversed(range(self.size)):
            ret[t] = self.reward[t] + self.discount * (1 - self.dones_float[t]) * ret[t+1]
        self.ret = ret[:-1]

        # accumulative return for traj
        returns = self.compute_return()
        self.returns = returns
        probs = (returns - returns.min()) / (returns.max() - returns.min()) + self.base_prob
        self.probs = probs / probs.sum()
        # rebalance
        if self.reweight:
            self.weights = self.probs * self.size
        else:
            self.weights = np.ones_like(self.probs)

        if self.resample:
            self.sampler = PrefetchBalancedSampler(self.probs, self.size, self.batch_size, n_prefetch=1000)
        else:
            self.sampler = RandSampler(self.size, self.batch_size)
        # At the first behavior policy iteration, uniform sample
        self.bc_sampler = RandSampler(self.size, self.batch_size)

        # n-step bootstrap for bc eval
        if self.n_step == 1: return
        ret_n = np.copy(self.reward)
        # done_n = 1 - self.not_done
        done_n = np.copy(self.dones_float)
        for n in range(1, self.n_step):
            # alternatively calculate return_n_step and done_n_step
            ret_n[:-n] += (self.discount ** n) * self.reward[n:] * (1 - done_n[:-n])
            done_n[:-n] = np.maximum(done_n[:-n], self.dones_float[n:])
        # !  While here does not estimate value of the truncated state, we (intuitively) should.
        state_n = np.zeros_like(self.state)
        state_n[:-self.n_step+1] = self.next_state[self.n_step-1:]
        
        self.ret_n, self.done_n, self.state_n = ret_n, done_n, state_n

    
    def compute_return(self):
        returns, ret, start = [], 0, 0
        for i in range(self.size):
            ret = ret+self.reward[i]
            if self.dones_float[i]: 
                returns.extend([ret]*(i-start+1))
                start = i + 1
                ret = 0
        assert len(returns) == self.size
        return np.stack(returns)

    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        if self.n_step > 1:
            self.state_n = (self.state_n - mean)/std
        return mean, std

    def replace_weights(self, weight, weight_func, exp_lambd=1.0, std=1.0, eps=0.0):
        #? need set adv_prob_base?
        if weight_func == 'linear':
            weight = weight - weight.min()
            prob = weight / weight.sum()
            # keep mean, scale std
            scale = std / (prob.std() * self.size)
            prob = np.maximum(scale*(prob - 1/self.size) + 1/self.size, eps/self.size)
            prob = prob/prob.sum() # norm to 1 again
        elif weight_func == 'exp':
            weight = weight / np.abs(weight).mean()
            weight = np.exp(exp_lambd * weight)
            prob = weight / weight.sum()
        self.probs = prob

        if self.reweight:
            if len(prob.shape) == 1:
                prob = np.expand_dims(prob, 1)
            self.weights = prob * self.size
        if self.resample:
            self.sampler.replace_prob(self.probs)

    def reset_bc(self, weight):
        # At the first behavior policy iteration, uniform sample
        self.bc_sampler = PrefetchBalancedSampler(weight, self.size, self.batch_size, n_prefetch=1000)