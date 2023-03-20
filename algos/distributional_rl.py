import jax
import jax.numpy as jnp

class DistAgent:
    def __init__(self, qf, policy, gamma) -> None:
        self.qf = qf
        self.policy = policy
        self.gamma = gamma
    
    def dist_critic_loss(self, qf_params, tgt_params, policy_params, obs, actions, rewards, next_obs, dones):
        raise NotImplementedError
    
    def dist2value(self, dist):
        raise NotImplementedError


class C51Agent(DistAgent):
    def __init__(self, qf, policy, vmin, vmax, num_atoms, gamma=0.99):
        DistAgent.__init__(self, qf, policy, gamma)
        
        # c51
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms
        self.gamma = gamma
        self.model_support = jnp.linspace(vmin, vmax, num_atoms)


    def categorical_projection(self, target_probs, target_support):
        """
        Perform categorical projection of the target probabilities onto the model support.

        Args:
        target_probs: The target probabilities, shape (batch_size, num_atoms).
        target_support: The support of the target distribution, shape (batch_size, num_atoms).

        Returns:
        The projected probabilities, shape (batch_size, num_atoms).
        """
        # Clip the target_support to the range [self.vmin, self.vmax]
        clipped_target_support = jnp.clip(target_support, self.vmin, self.vmax)

        # Calculate the indices in the model_support that are closest to the clipped_target_support
        delta_z = (self.vmax - self.vmin) / (self.num_atoms - 1)
        normalized_diff = (clipped_target_support - self.vmin) / delta_z
        index_lower = jnp.floor(normalized_diff).astype(jnp.int32)
        index_upper = jnp.ceil(normalized_diff).astype(jnp.int32)

        # Calculate the contribution of the target probabilities to the lower and upper indices
        contribution_lower = (1 - (normalized_diff - index_lower)) * target_probs
        contribution_upper = (1 - (index_upper - normalized_diff)) * target_probs

        # Perform the projection by adding contributions to the corresponding indices
        projected_probs = jnp.zeros_like(target_probs)
        projected_probs = jax.ops.index_add(projected_probs, jax.ops.index[jnp.arange(target_probs.shape[0])[:, None], index_lower], contribution_lower)
        projected_probs = jax.ops.index_add(projected_probs, jax.ops.index[jnp.arange(target_probs.shape[0])[:, None], index_upper], contribution_upper)

        return projected_probs
    
    def dist_critic_loss(self, qf_params, tgt_params, policy_params, obs, actions, rewards, next_obs, dones):
        # Obtain logits from the two networks for the current observation
        logits1, logits2 = self.qf.apply(qf_params, obs, actions)

        # Generate next actions using the policy network
        next_actions = self.policy.apply(policy_params, next_obs)

        # Obtain logits from the two target networks for the next observation
        next_logits1, next_logits2 = self.qf.apply(tgt_params, next_obs, next_actions)

        # Calculate Q values for the two networks
        next_q_values1 = jnp.sum(next_logits1 * self.model_support, axis=-1)
        next_q_values2 = jnp.sum(next_logits2 * self.model_support, axis=-1)

        # Calculate target probabilities for the two networks
        target_probs1 = jax.nn.softmax(next_logits1, axis=-1)
        target_probs2 = jax.nn.softmax(next_logits2, axis=-1)

        # Choose the better target_probs based on next_q_values
        better_target_probs = jnp.where(next_q_values1 < next_q_values2, target_probs1, target_probs2)

        # Calculate the target support
        target_support = rewards[:, None] + self.gamma * self.model_support[None, :] * (1 - dones)[:, None]

        # Perform categorical projection for the better target_probs
        projected_probs = self.categorical_projection(better_target_probs, target_support, self.model_support, self.vmin, self.vmax, self.num_atoms)

        # Calculate the loss for logits1 and logits2
        loss1 = -jnp.mean(jnp.sum(projected_probs * jax.nn.log_softmax(logits1, axis=-1), axis=-1))
        loss2 = -jnp.mean(jnp.sum(projected_probs * jax.nn.log_softmax(logits2, axis=-1), axis=-1))

        # Combine the losses
        loss = loss1 + loss2
        return loss


class QRAgent(DistAgent):
    def __init__(self, qf, policy, num_quantiles=51, gamma=0.99):
        DistAgent.__init__(self, qf, policy, gamma)
        self.num_quantiles = num_quantiles
        
    def dist_critic_loss(self, qf_params, tgt_params, policy_params, obs, actions, rewards, next_obs, dones, rng):
        """
        Compute the Quantile Regression DQN loss.

        Args:
        params: The parameters of the Q-networks and policy network.
        obs: The current observations, shape (batch_size, obs_dim).
        actions: The current actions, shape (batch_size, action_dim).
        rewards: The rewards, shape (batch_size,).
        next_obs: The next observations, shape (batch_size, obs_dim).
        dones: The done flags, shape (batch_size,).

        Returns:
        The QR-DQN loss.
        """

        # Compute current quantiles
        quantiles1 = self.qf.apply(qf_params[0], obs, actions)  # shape: (batch_size, num_quantiles)
        quantiles2 = self.qf.apply(qf_params[1], obs, actions)  # shape: (batch_size, num_quantiles)

        # Compute next actions
        next_actions = self.policy.apply(policy_params, rng, next_obs)

        # Compute next quantiles
        next_quantiles1 = self.qf.apply(tgt_params[0], next_obs, next_actions)  # shape: (batch_size, num_quantiles)
        next_quantiles2 = self.qf.apply(tgt_params[1], next_obs, next_actions)  # shape: (batch_size, num_quantiles)
    
        # Double Q-learning
        cond = jnp.repeat(jnp.expand_dims(
            next_quantiles1.mean(axis=-1) < next_quantiles2.mean(axis=-1), axis=-1), self.num_quantiles - 1, axis=-1
        )
        next_quantiles = jnp.where(cond, next_quantiles1, next_quantiles2)

        # Compute target quantiles
        target_quantiles = rewards[:, None] + (1 - dones)[:, None] * self.gamma * next_quantiles

        # Compute pairwise differences
        td_errors1 = target_quantiles[:, None, :] - quantiles1[:, :, None]  # shape: (batch_size, num_quantiles, num_quantiles)
        td_errors2 = target_quantiles[:, None, :] - quantiles2[:, :, None]  # shape: (batch_size, num_quantiles, num_quantiles)

        huber_loss1 = jnp.where(jnp.abs(td_errors1) <= 1.0, 0.5 * td_errors1 ** 2, jnp.abs(td_errors1) - 0.5)
        huber_loss2 = jnp.where(jnp.abs(td_errors2) <= 1.0, 0.5 * td_errors1 ** 2, jnp.abs(td_errors2) - 0.5)
        
        taus = jnp.linspace(0, 1, self.num_quantiles)
        mid_quantiles = (taus[:-1] + taus[1:]) / 2  # (num_quantiles,)
        quantile_weights1 = jnp.abs(mid_quantiles[:, None] - (td_errors1 < 0).astype(jnp.float32)) # shape: (batch_size, num_quantiles, num_quantiles)
        quantile_weights2 = jnp.abs(mid_quantiles[:, None] - (td_errors2 < 0).astype(jnp.float32))
        
        # qr_loss1 = jnp.mean(jnp.sum(huber_loss1 * quantile_weights1, axis=-1))
        # qr_loss2 = jnp.mean(jnp.sum(huber_loss2 * quantile_weights2, axis=-1))
        qr_loss1 = jnp.mean(huber_loss1 * quantile_weights1)
        qr_loss2 = jnp.mean(huber_loss2 * quantile_weights2)
        
        # logging locals
        qf_loss = qr_loss1 + qr_loss2
        qf1_loss, qf2_loss = qr_loss1, qr_loss2
        cur_q1, cur_q2 = quantiles1.mean(axis=-1), quantiles2.mean(axis=-1)
        tgt_q1, tgt_q2 = next_quantiles1.mean(axis=-1), next_quantiles2.mean(axis=-1)
        tgt_q = jnp.minimum(tgt_q1, tgt_q2)
        
        return (qr_loss1, qr_loss2), locals()
    
    def dist2value(self, dist):
        return dist.mean(axis=-1)