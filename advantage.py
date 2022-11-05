import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TD3_BC import Critic
import math
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValueNet(nn.Module):
	def __init__(self, state_dim):
		super(ValueNet, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state):
		v1 = F.relu(self.l1(state))
		v1 = F.relu(self.l2(v1))
		v1 = self.l3(v1)
		return v1


class DoubleValueNet(nn.Module):
	def __init__(self, state_dim):
		super(DoubleValueNet, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)



	def forward(self, state):
		v1 = F.relu(self.l1(state))
		v1 = F.relu(self.l2(v1))
		v1 = self.l3(v1)

		v2 = F.relu(self.l4(state))
		v2 = F.relu(self.l5(v2))
		v2 = self.l6(v2)
		return v1, v2


class Advantage(nn.Module):
	def __init__(
		self,
		state_dim,
		action_dim,
		discount=0.99,
		tau=0.005,
		td_type='nstep',
		adv_type='gae',
		lambd=0.95,
		**kwargs,
	):
		super().__init__()
		self.discount = discount
		self.tau = tau
		self.total_it = 0
		self.td_type = td_type
		self.adv_type = adv_type
		self.lambd = lambd

	# for train (target)
	@torch.no_grad()
	def get_value_target(self, state):
		return self.value_target(state)

	# for 
	@torch.no_grad()
	def td(self, data):
		if self.td_type == 'nstep' and self.replay_buffer.n_step > 1:
			state, action, state_n, ret_n, done_n = data
			return ret_n + (1-done_n) * (self.discount**self.n_step) * self.get_value_target(state_n)	
		
		state, action, next_state, reward, not_done, dones_float, rets = data
		if self.td_type == 'nstep' and self.replay_buffer.n_step == 1:
				return reward + not_done * self.discount * self.get_value_target(next_state)	
		elif self.td_type == 'mc': # 1. from current timestep t from T; 2. discounted
			return rets
		else:
			raise NotImplementedError
	
	# for eval
	def get_value(self, state):
		return self.value(state)
	
	def set_replay_buffer(self, replay_buffer):
		self.replay_buffer = replay_buffer
		self.n_step = replay_buffer.n_step

	@torch.no_grad()
	def eval(self, batch_size=256):
		n_step = self.replay_buffer.n_step
		if self.adv_type == 'nstep' and n_step > 1:
			ret_n, done_n = self.replay_buffer.ret_n, self.replay_buffer.done_n
			values, next_values = [], []
			for l in range(0, self.replay_buffer.size, batch_size):
				r = min(l+batch_size, self.replay_buffer.size)
				ind = list(range(l, r))
				data = self.replay_buffer.sample_n_step_by_ind(ind)
				state, next_state = data[0], data[2]
				v0 = self.get_value(state) 
				v1 = self.get_value(next_state)
				values.append(v0.cpu())
				next_values.append(v1.cpu())
			values, next_values = np.concatenate(values),  np.concatenate(next_values)
			q = ret_n + (1 - done_n) * (self.discount**n_step) * next_values
			adv = q - values
			return adv, q, values

		values, next_values = [], []
		for l in range(0, self.replay_buffer.size, batch_size):
			r = min(l+batch_size, self.replay_buffer.size)
			data = self.replay_buffer.sample_by_ind(list(range(l, r)))
			state, next_state = data[0], data[2]
			v0 = self.get_value(state) 
			v1 = self.get_value(next_state)
			values.append(v0.cpu())
			next_values.append(v1.cpu())
		values, next_values = np.concatenate(values),  np.concatenate(next_values)
		rewards, not_dones, dones_float = self.replay_buffer.reward, self.replay_buffer.not_done, self.replay_buffer.dones_float
		bs = rewards.shape[0]
		if self.adv_type == 'nstep' and n_step == 1:
			q = rewards + not_dones * self.discount * next_values
			adv = q - values
		if self.adv_type == 'gae':
			adv = np.zeros((bs+1, 1))
			delta = rewards + not_dones * self.discount * next_values - values 
			for t in reversed(range(bs)):
				# ?unbiased gae
				adv[t] = delta[t] + self.discount * self.lambd * (1 - dones_float[t]) * adv[t+1]
			adv = adv[:-1]
			q = adv + values
		if self.adv_type == 'mc':
			q = self.replay_buffer.ret
			adv = q - values
			
		return adv, q, values

		
class V_Advantage(Advantage):
	def __init__(
		self,
		state_dim,
		action_dim,
		bc_lr_schedule,
		maxstep,
		discount=0.99,
		tau=0.005,
		**kwargs,
	):
		super().__init__(state_dim, action_dim, discount, tau, **kwargs)
		self.value = ValueNet(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
		self.value_lr_scheduler = get_scheduler(self.value_optimizer, bc_lr_schedule, maxstep)

	
	def train(self, replay_buffer):
		self.total_it += 1
		# Sample replay buffer 
		if self.td_type == 'nstep' and self.replay_buffer.n_step > 1:
			data = replay_buffer.bc_eval_sample_n()
		else:
			data = replay_buffer.bc_eval_sample()
		state = data[0]

		with torch.no_grad():
			v_target = self.td(data)
		v = self.value(state)
		# Compute critic loss
		value_loss = F.mse_loss(v, v_target).mean()

		# Optimize the critic
		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		# Update the frozen target models
		self.update_target()
		self.value_lr_scheduler.step()


		return {
			"value_loss": value_loss.mean().cpu(),
			"v": v.mean().cpu(),
			"value_lr": self.value_optimizer.param_groups[0]['lr'],
		}
	
	def update_target(self):
		for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.value.state_dict(), filename + "_critic")
		torch.save(self.value_optimizer.state_dict(), filename + "_critic_optimizer")


	def load(self, filename):
		self.value.load_state_dict(torch.load(filename + "_critic"))
		self.value_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.value_target = copy.deepcopy(self.value)



class DoubleV_Advantage(V_Advantage):
	def __init__(
		self,
		state_dim,
		action_dim,
		bc_lr_schedule,
		maxstep,
		discount=0.99,
		tau=0.005,
		**kwargs,
	):
		super().__init__(state_dim, action_dim, bc_lr_schedule, maxstep, discount, tau, **kwargs)
		self.value = DoubleValueNet(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
		self.value_lr_scheduler = get_scheduler(self.value_optimizer, bc_lr_schedule, maxstep)

	
	def train(self, replay_buffer):
		self.total_it += 1
		# Sample replay buffer 
		if self.td_type == 'nstep' and self.replay_buffer.n_step > 1:
			data = replay_buffer.bc_eval_sample_n()
		else:
			data = replay_buffer.bc_eval_sample()
		state = data[0]

		with torch.no_grad():
			v_target = self.td(data)
		v1, v2 = self.value(state)
		# Compute critic loss
		value_loss = F.mse_loss(v1, v_target).mean() + F.mse_loss(v2, v_target).mean()

		# Optimize the critic
		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		# Update the frozen target models
		self.update_target()
		self.value_lr_scheduler.step()


		return {
			"value_loss": value_loss.mean().cpu(),
			"v1": v1.mean().cpu(),
			"v2": v2.mean().cpu(),
			"value_lr": self.value_optimizer.param_groups[0]['lr'],
		}
	
	# for train (target)
	@torch.no_grad()
	def get_value_target(self, state):
		v1, v2 =  self.value_target(state)
		return torch.minimum(v1, v2)

	# for eval: use value network rather than target value_network
	def get_value(self, state):
		v1, v2 = self.value(state)
		return torch.minimum(v1, v2)


class VQ_Advantage(Advantage):
	def __init__(
		self,
		state_dim,
		action_dim,
		bc_lr_schedule,
		maxstep,
		discount=0.99,
		tau=0.005,
		**kwargs
	):
		super().__init__(state_dim, action_dim, discount, tau, **kwargs)
		self.value = ValueNet(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
		self.value_lr_scheduler = get_scheduler(self.value_optimizer, bc_lr_schedule, maxstep)
		
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_lr_scheduler = get_scheduler(self.critic_optimizer, bc_lr_schedule, maxstep)
	
	def train(self, replay_buffer):
		self.total_it += 1
		# Sample replay buffer 
		data = replay_buffer.bc_eval_sample()
		state, action = data[0], data[1]

		with torch.no_grad():
			v_target = self.td(data)
		q1, q2 = self.critic(state, action)
		# Compute critic loss
		critic_loss = (q1 - v_target)**2 + (q2 - v_target)**2
		critic_loss = critic_loss.mean()
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		with torch.no_grad():
			q1, q2 = self.critic_target(state, action)
			q = torch.minimum(q1, q2)
		v = self.value(state)
		value_loss = F.mse_loss(v, q).mean()
		# Optimize the critic
		self.value_optimizer.zero_grad()
		value_loss.backward()
		self.value_optimizer.step()

		# Update the frozen target models
		self.update_target()
		self.critic_lr_scheduler.step()
		self.value_lr_scheduler.step()

		return {
			"critic_loss": critic_loss.mean().cpu(),
			"value_loss": value_loss.mean().cpu(),
			"v": v.mean().cpu(),
			"q": q.mean().cpu(),
			"critic_lr": self.critic_optimizer.param_groups[0]['lr'],
			"value_lr": self.value_optimizer.param_groups[0]['lr'],
		}
	
	def update_target(self):
		for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.value.state_dict(), filename + "_critic")
		torch.save(self.value_optimizer.state_dict(), filename + "_critic_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)
		self.value.load_state_dict(torch.load(filename + "_critic"))
		self.value_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.value_target = copy.deepcopy(self.value)


def get_scheduler(optimizer, adjust_lr, epochs, warmup_epoch=0) :
	"""Return a learning rate scheduler
		Parameters:
		optimizer -- 网络优化器
		opt.lr_policy -- 学习率scheduler的名称: linear | step | plateau | cosine
	"""
	def warmup_rule(epoch) :
		if epoch < warmup_epoch :
			lr_l = epoch / warmup_epoch
		else:
			T = epoch - warmup_epoch
			total_epoch = epochs - warmup_epoch

			if adjust_lr == 'cosine':
				lr_l = 0.5 * (1 + math.cos(T / total_epoch * math.pi))
			# elif adjust_lr == 'step':
			# 	gamma = opt.step_gamma
			# 	step_size = opt.step_size
			# 	lr_l = gamma ** (T//step_size)
			elif adjust_lr == 'linear':
				lr_l = 1.0 - T / total_epoch
			elif adjust_lr == 'none':
				lr_l = 1.0
			else:
				raise NotImplementedError('learning rate policy [%s] is not implemented', adjust_lr)
		return lr_l

	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_rule)

	return scheduler