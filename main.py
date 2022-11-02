import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import wandb
import utils
import TD3_BC
import time
from advantage import *

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1,-1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


def eval_bc(bc_advantage, replay_buffer, batch_size=256):
    advs, qs, vs = [], [], []
    for l in range(0, replay_buffer.size, batch_size):
        r = min(l+batch_size, replay_buffer.size)
        data = replay_buffer.sample_by_ind(list(range(l, r)))
        adv, q, v = bc_advantage.adv(data)
        advs.append(adv.squeeze().cpu())
        qs.append(q.squeeze().cpu())
        vs.append(v.squeeze().cpu())
    return np.concatenate(advs), np.concatenate(qs), np.concatenate(vs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")               # Policy name
    parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--log_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=10, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    # generate weight
    parser.add_argument("--bc_eval", type=int, default=1)       
    parser.add_argument("--bc_eval_steps", type=int, default=1e6)       
    parser.add_argument("--critic_type", type=str, default='v', choices=['v', 'vq'])   
    parser.add_argument("--td_type", type=str, default='onestep', choices=['onestep', 'mc', 'gae']) 
    parser.add_argument("--bc_lr_schedule", type=str, default='cosine', choices=['cosine', 'linear', 'none']) 
    parser.add_argument("--weight_freq", default=5e4, type=int)    
    # TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    # rebalance
    parser.add_argument("--base_prob", default=0.0, type=float)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--reweight_eval", default=1, type=int)
    parser.add_argument("--reweight_improve", default=1, type=int)
    parser.add_argument("--reweight_constraint", default=1, type=int)
    parser.add_argument("--clip_constraint", default=0, type=int)  # 0: no clip; 1: hard clip; 2 soft clip
    parser.add_argument("--tag", default='', type=str)
    args = parser.parse_args()

    # resample and reweight can not been applied together
    assert not args.resample or not args.reweight

    file_name = f"{args.critic_type}_{args.td_type}_{args.bc_lr_schedule}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")


    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # generate weight
        "bc_eval": args.bc_eval,
        "bc_eval_steps": args.bc_eval_steps,
        "critic_type": args.critic_type,
        "td_type": args.td_type,
        "bc_lr_schedule": args.bc_lr_schedule,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha, 
        "reweight_eval": args.reweight_eval, 
        "reweight_improve": args.reweight_improve,
        "reweight_constraint": args.reweight_constraint,
        "clip_constraint": args.clip_constraint,
    }

    wandb.init(project="TD3_BC", config={
            "env": args.env, "seed": args.seed, "tag": args.tag,
            "resample": args.resample, "reweight": args.reweight, "p_base": args.base_prob,
            **kwargs
            })

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.batch_size,
        base_prob=args.base_prob, resample=args.resample, reweight=args.reweight)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    
    if 'antmaze' in args.env:
        replay_buffer.reward -= 1.0
    if args.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1

    if args.bc_eval:
        if args.critic_type == 'v':
            bc_advantage = V_Advantage(state_dim, action_dim, args.td_type, args.bc_lr_schedule, args.bc_eval_steps, args.discount, args.tau)
        elif args.critic_type == 'vq':
            bc_advantage = VQ_Advantage(state_dim, action_dim, args.td_type, args.bc_lr_schedule, args.bc_eval_steps, args.discount, args.tau)
        else:
            raise NotImplementedError

        bc_eval_results = {}
        for t in range(int(args.bc_eval_steps)):
            infos = bc_advantage.train(replay_buffer)
            if (t + 1) % args.log_freq == 0:
                for k, v in infos.items():
                    wandb.log({f'bc/train/{k}': v}, step=t+1)
            if (t + 1) % args.weight_freq == 0:
                adv, q, v = eval_bc(bc_advantage, replay_buffer)
                bc_eval_results[t+1] = {'adv': adv, 'q': q, 'v': v}
                wandb.log({f'bc/eval/q': q.mean()}, step=t+1)
                wandb.log({f'bc/eval/v': v.mean()}, step=t+1)
                wandb.log({f'bc/eval/abs_adv': np.abs(adv).mean()}, step=t+1)
                wandb.log({f'bc/eval/positive_adv': (adv-adv.min()).mean()}, step=t+1)
        replay_buffer.replace_weights(bc_eval_results[t+1]['adv'])
        np.save(f'./weights/{file_name}.npy', bc_eval_results)

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    # time0 = time.time()
    evaluations = []
    for t in range(int(args.bc_eval_steps), int(args.bc_eval_steps + args.max_timesteps)):
        infos = policy.train(replay_buffer)
        if (t + 1) % args.log_freq == 0:
            for k, v in infos.items():
                wandb.log({f'train/{k}': v}, step=t+1)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            evaluations.append(eval_policy(policy, args.env, args.seed, mean, std, args.eval_episodes))
            wandb.log({f'eval/score': evaluations[-1]}, step=t+1)
            wandb.log({f'eval/avg10_score': np.mean(evaluations[-min(10, len(evaluations)):])}, step=t+1)
            # np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
        # if (t + 1) % 100 == 0:
        # 	dt = time.time() - time0
        # 	time0 += dt
        # 	print(f"Time steps: {t+1}, speed: {round(100/dt, 1)}itr/s")
        
