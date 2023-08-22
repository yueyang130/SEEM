import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_grads(grads):
        return torch.cat([g.view(-1) for g in grads])

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layernorm, hidden_dim=256):
        super(Critic, self).__init__()
  
        # # Q1 architecture
        # self.l1 = nn.Linear(state_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 1)

        # # Q2 architecture
        # self.l4 = nn.Linear(state_dim + action_dim, 256)
        # self.l5 = nn.Linear(256, 256)
        # self.l6 = nn.Linear(256, 1)
  
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


class TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        reweight_eval,
        reweight_improve,
        reweight_constraint,
        clip_constraint,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        bc_coef=1.0,
        qf_layer_norm=False,
        optimizer_type='adam',
        **kwargs,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
  

        self.critic = Critic(state_dim, action_dim, qf_layer_norm).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        if optimizer_type == 'adam':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        elif optimizer_type == 'sgd':
            self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=3e-4)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=3e-4)
        else:
            raise NotImplementedError
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.bc_coef = bc_coef

        self.reweight_eval = reweight_eval
        self.reweight_improve = reweight_improve
        self.reweight_constraint = reweight_constraint
        self.clip_constraint = clip_constraint

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def compute_grad(self, batch):
        state, action, next_state, reward, not_done, _ = batch
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
    
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
        critic_loss = critic_loss.mean()

        # critic grad
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad = [p.grad for p in self.critic.q1.parameters() if p.grad is not None]
        
        # find max action and max Q
        pi = self.actor(state)
        max_Q = self.critic.Q1(state, pi)
        
        grad = flatten_grads(grad).cpu()
        # scale
        grad = grad / (grad.shape[0])**0.5
        
        return grad.detach(), current_Q1.detach(), max_Q.detach(), pi.detach()
        

    def train(self, replay_buffer, two_sampler=False):
        self.total_it += 1
        # Sample replay buffer 
        if two_sampler:
            state, action, next_state, reward, not_done, weight = replay_buffer.sample(uniform=True)
        else:
            state, action, next_state, reward, not_done, weight = replay_buffer.sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
        if self.reweight_eval:
            critic_loss *= weight
        critic_loss = critic_loss.mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad = [p.grad for p in self.critic.q1.parameters() if p.grad is not None]
        grad = flatten_grads(grad)
        grad = grad / (grad.shape[0])**0.5
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_infos = {}
        if self.total_it % self.policy_freq == 0:
            if two_sampler:
                state, action, next_state, reward, not_done, weight = replay_buffer.sample()
            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha/Q.abs().mean().detach()

            # policy improvement
            actor_loss = Q
            if self.reweight_improve:
                actor_loss *= weight
            actor_loss = actor_loss.mean()
            # policy constraint
            constraint_loss = F.mse_loss(pi, action, reduction='none') 
            if self.reweight_constraint:
                if self.clip_constraint == 1:
                    c_weight = torch.clamp(weight, 1.0)
                elif self.clip_constraint == 2:
                    c_weight = copy.deepcopy(weight)
                    c_weight[weight < 1] = torch.sqrt(weight[weight < 1])
                else:
                    c_weight = weight
                constraint_loss *= c_weight
            constraint_loss = constraint_loss.mean()
            actor_loss = -lmbda * actor_loss + constraint_loss * self.bc_coef
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # log actor training
            actor_infos = { 
                "critic_loss": critic_loss.mean().cpu(),
                "actor_loss": actor_loss.mean().cpu(),
                "constraint_loss": constraint_loss.mean().cpu(),
                "lambda": lmbda.cpu(), 
            }

        def flatten_parameters(model):
            return torch.cat([param.view(-1) for param in model.parameters()])

        def model_weights_norm(m1):
            m1_flat = flatten_parameters(m1)
            m1_norm = torch.norm(m1_flat, p=2)
            return m1_norm.item()

        return {
            "Q1": current_Q1.mean().cpu(),
            "Q2": current_Q2.mean().cpu(),
            "Q1_norm": model_weights_norm(self.critic.q1),
            "Q2_norm": model_weights_norm(self.critic.q2),
            **actor_infos,
        }, grad
        
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

