import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import wandb
import utils
from advantage import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")               # Policy name
    parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--log_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    # generate weight
    parser.add_argument("--iter", type=int, default=5, help='how many times to iteratively rebalance')       
    parser.add_argument("--scale", action='store_true', help='If true, scale weights for larger standard deviation')       
    parser.add_argument("--first_eval_steps", type=int, default=1e6, help='the first number of steps to eval a bc')       
    parser.add_argument("--bc_eval_steps", type=int, default=1e6, help='number of steps to eval a bc')       
    parser.add_argument("--critic_type", type=str, default='doublev', choices=['v', 'vq', 'doublev'])   
    parser.add_argument("--td_type", type=str, default='nstep', choices=['nstep', 'mc']) 
    parser.add_argument("--adv_type", type=str, default='nstep', choices=['nstep', 'mc', 'gae']) 
    parser.add_argument("--n_step", type=int, default=1, help='n-step return') 
    parser.add_argument("--lambd", type=float, default=0.95, help='gae lambda') 
    parser.add_argument("--bc_lr_schedule", type=str, default='cosine', choices=['cosine', 'linear', 'none']) 
    parser.add_argument("--weight_freq", default=5e4, type=int)    
    parser.add_argument("--weight_func", default='linear', choices=['linear', 'exp', 'power'])    
    parser.add_argument("--exp_lambd", default=1.0, type=float)    
    parser.add_argument("--std", default=2.0, type=float, help="scale weights' standard deviation.")    
    parser.add_argument("--eps", default=0.1, type=float, help="")    
    # TD3
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005) 
    parser.add_argument("--normalize", default=1, type=int)
    parser.add_argument("--tag", default='', type=str)                   # Target network update rate
    args = parser.parse_args()


    # file_name = f"{args.critic_type}_{args.td_type}_{args.adv_type}_{args.n_step}_{args.lambd}_{args.bc_lr_schedule}_{args.env}_{args.seed}"
    # file_name = f"{args.critic_type}_{args.td_type}_{args.adv_type}_{args.n_step}_{args.lambd}_{args.bc_lr_schedule}_{args.iter}_{args.bc_eval_steps}_scale={args.scale}_{args.env}_{args.seed}"
    file_name = f"{args.td_type}_{args.adv_type}_nstep={args.n_step}_{args.first_eval_steps}_{args.iter}_{args.bc_eval_steps}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    adv_kawargs = {
        "critic_type": args.critic_type,
        "td_type": args.td_type,
        "adv_type": args.adv_type,
        "n_step": args.n_step,
        "lambd": args.lambd,
    }

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "normalize": args.normalize,
        # generate weight
        "first_eval_steps": args.first_eval_steps,
        "bc_eval_steps": args.bc_eval_steps,
        "iter": args.iter,
        "scale": args.scale,
        "bc_lr_schedule": args.bc_lr_schedule,
        "weight_func": args.weight_func,
        "exp_lambd": args.exp_lambd,
        "std": args.std,
        "eps": args.eps,
        **adv_kawargs,
    }

    wandb.init(project="TD3_BC", config={
            "env": args.env, "seed": args.seed, "tag": args.tag, **kwargs
            })

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.batch_size, 
    n_step=args.n_step, discount=args.discount)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    # save return dist
    np.save(f'./weights/{args.env}_returns.npy', replay_buffer.returns)
    
    #! Note: IQL and TD3+BC have different preprocess.
    # IQL: For mujoco, reward norm; for antmaze, reward minus.
    # TD3+BC: For all, state norm; for antmaze, reward minus.
    if 'antmaze' in args.env:
        replay_buffer.reward -= 1.0
        # replay_buffer.reward = (replay_buffer.reward - 0.5) * 4 
    if args.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1

    wp = f'./weights/{file_name}.npy'
    
    if args.critic_type == 'v':
        bc_advantage = V_Advantage(state_dim, action_dim, args.bc_lr_schedule, args.first_eval_steps, args.discount, args.tau, **adv_kawargs)
    if args.critic_type == 'doublev':
        bc_advantage = DoubleV_Advantage(state_dim, action_dim, args.bc_lr_schedule, args.first_eval_steps, args.discount, args.tau, **adv_kawargs)
    elif args.critic_type == 'vq':
        bc_advantage = VQ_Advantage(state_dim, action_dim, args.bc_lr_schedule, args.first_eval_steps, args.discount, args.tau, **adv_kawargs)
    else:
        raise NotImplementedError
    bc_advantage.set_replay_buffer(replay_buffer)

    bc_eval_results = {'iter': args.iter, 'first_eval_steps': args.first_eval_steps, 'eval_steps': args.bc_eval_steps}
    weight = np.ones((replay_buffer.size, 1), dtype=np.float32)
    for t in range(int(args.bc_eval_steps*(args.iter-1) + args.first_eval_steps)):
        infos = bc_advantage.train(replay_buffer)
        if (t + 1) % args.log_freq == 0:
            for k, v in infos.items():
                wandb.log({f'bc/train/{k}': v}, step=t+1)
        if (t + 1) % args.weight_freq == 0:
            adv, q, v = bc_advantage.eval()
            padv = adv-adv.min()
            
            wandb.log({f'bc/eval/q': q.mean()}, step=t+1)
            wandb.log({f'bc/eval/v': v.mean()}, step=t+1)
            wandb.log({f'bc/eval/abs_adv': np.abs(adv).mean()}, step=t+1)
            wandb.log({f'bc/eval/positive_adv': padv.mean()}, step=t+1)
        if (t + 1 - args.first_eval_steps) >= 0 and (t + 1 - args.first_eval_steps) % args.bc_eval_steps == 0: 
            curr_itr = int((t + 1 - args.first_eval_steps) / args.bc_eval_steps + 1)
            # reset optimizer and lr schedule
            bc_advantage.reset_optimizer(args.bc_eval_steps)
            # reset behavior policy, i.e., resampling
            current_weight = padv / padv.sum() * replay_buffer.size
            # Note: var1 += var2 is not the same as var1 = var1 + var2 with collections.
            # see https://stackoverflow.com/questions/35910577/why-does-python-numpys-mutate-the-original-array
            # weight *= current_weight
            weight = weight * current_weight
            weight = weight / weight.sum() * replay_buffer.size
            wandb.log({f'bc/eval/weight_std': weight.std()}, step=t+1)
            if args.scale:
                scale = args.std / weight.std()
                if scale > 1:
                    weight = np.maximum(scale*(weight - 1) + 1, args.eps)
                    weight = weight / weight.sum() * replay_buffer.size
                    wandb.log({f'bc/eval/scaled_weight_std': weight.std()}, step=t+1)

            replay_buffer.reset_bc(weight)
            bc_eval_results[curr_itr] = weight
            np.save(wp, bc_eval_results)
            print(f'#iter {t+1}, saved at {file_name}')


    
    
    
