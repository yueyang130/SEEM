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


class Advantage(nn.Module):
	def __init__(
		self,
		state_dim,
		action_dim,
		td_type,
		discount=0.99,
		tau=0.005,
	):
		super().__init__()
		self.discount = discount
		self.tau = tau
		self.total_it = 0
		self.td_type = td_type

	@torch.no_grad()
	def get_target_value(self, data):
		state, action, next_state, reward, not_done, returns = data
		if self.td_type == 'onestep':
			return reward + not_done * self.discount * self.value_target(next_state)
		elif self.td_type == 'mc':
			# 1. from current timestep t from T; 2. discounted
			raise NotImplementedError
		elif self.td_type == 'gae':
			raise NotImplementedError
		else:
			raise NotImplementedError

	@torch.no_grad()
	def adv(self, data):
		q = self.get_target_value(data)
		v = self.value(data[0])
		return q - v, q, v


class V_Advantage(Advantage):
	def __init__(
		self,
		state_dim,
		action_dim,
		td_type,
		bc_lr_schedule,
		maxstep,
		discount=0.99,
		tau=0.005,
	):
		super(V_Advantage, self).__init__(state_dim, action_dim, td_type)
		self.value = ValueNet(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
		self.value_lr_scheduler = get_scheduler(self.value_optimizer, bc_lr_schedule, maxstep)
		
		self.discount = discount
		self.tau = tau
		self.total_it = 0

	
	def train(self, replay_buffer):
		self.total_it += 1
		# Sample replay buffer 
		data = replay_buffer.bc_eval_sample()
		state, action, next_state, reward, not_done, returns = data

		with torch.no_grad():
			v_target = self.get_target_value(data)
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



class VQ_Advantage(Advantage):
	def __init__(
		self,
		state_dim,
		action_dim,
		td_type,
		bc_lr_schedule,
		maxstep,
		discount=0.99,
		tau=0.005,
	):
		super(VQ_Advantage, self).__init__(state_dim, action_dim, td_type)
		self.value = ValueNet(state_dim).to(device)
		self.value_target = copy.deepcopy(self.value)
		self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
		self.value_lr_scheduler = get_scheduler(self.value_optimizer, bc_lr_schedule, maxstep)
		
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_lr_scheduler = get_scheduler(self.critic_optimizer, bc_lr_schedule, maxstep)

		self.discount = discount
		self.tau = tau
		self.total_it = 0

	
	def train(self, replay_buffer):
		self.total_it += 1
		# Sample replay buffer 
		data = replay_buffer.bc_eval_sample()
		state, action, next_state, reward, not_done, returns = data

		with torch.no_grad():
			v_target = self.get_target_value(data)
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