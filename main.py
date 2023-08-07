import numpy as np
import torch
import torch.nn.functional as F
import gym
import argparse
import os
import d4rl
import wandb
import utils
import TD3_BC
import time
from copy import deepcopy

def cosine_similarity(m1, m2):

    dot_product = torch.dot(m1, m2)
    m1_norm = torch.norm(m1, p=2)
    m2_norm = torch.norm(m2, p=2)

    similarity = dot_product / (m1_norm * m2_norm)
    return similarity.item()

def cosine_similarity_matrix(input_tensor):

    # Normalize the input tensor along the feature dimension (n_dim)
    normalized_input = F.normalize(input_tensor, p=2, dim=1)

    # Calculate the dot product between the normalized tensors to get the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(normalized_input, normalized_input.t()).cpu()

    # Create a diagonal mask and set the diagonal elements to zero
    mask = torch.eye(input_tensor.shape[0]).bool()

    # Apply the mask to the cosine similarity matrix
    cosine_similarity_matrix.masked_fill_(mask, 0)

    # Reshape the result to a one-dimensional tensor
    return cosine_similarity_matrix.flatten()


def compute_ntk(model, states, actions):
    N = states.size(0)

    # Function to flatten gradients of model parameters
    def flatten_grads(grads):
        return torch.cat([g.view(-1) for g in grads])

    # Compute gradients for each input
    grads = []
    for i in range(N):
        s = states[i].unsqueeze(0)
        a = actions[i].unsqueeze(0)

        # Feedforward the input through the model
        output = model.Q1(s,a)

        # Zero the model's gradients
        model.zero_grad()

        # Compute and store the gradients for each output dimension
        output.backward()
        grad = flatten_grads([p.grad for p in model.q1.parameters() if p.grad is not None])

        # Stack gradients for the current input
        grads.append(grad)

    # Stack gradients for all inputs
    # divide a constant to prevent NAN
    grads_tensor = torch.stack(grads) / (grad.shape[0])**0.5

    # Compute the NTK matrix using tensor operations
    G = torch.matmul(grads_tensor, grads_tensor.t())

    return G.detach(), grads_tensor.detach()

def compute_detect_matrix(model, states, pi, batch_grads):
    N = states.size(0)

    # Function to flatten gradients of model parameters
    def flatten_grads(grads):
        return torch.cat([g.view(-1) for g in grads])

    # Compute gradients for each input
    grads = []
    for i in range(N):
        s = states[i].unsqueeze(0)
        a = pi[i].unsqueeze(0)

        # Feedforward the input through the model
        output = model.Q1(s,a)

        # Zero the model's gradients
        model.zero_grad()

        # Compute and store the gradients for each output dimension
        output.backward()
        grad = flatten_grads([p.grad for p in model.q1.parameters() if p.grad is not None])

        # Stack gradients for the current input
        grads.append(grad)

    # Stack gradients for all inputs
    # divide a constant to prevent NAN
    pi_grads = torch.stack(grads).detach() / (grad.shape[0])**0.5
    
    detect_matrix = args.discount * torch.matmul(pi_grads, batch_grads.t()) - torch.matmul(batch_grads, batch_grads.t()) 
    # eigenvalues = torch.linalg.eigvalsh(detect_matrix)
    eigenvalues = torch.linalg.eigvals(detect_matrix)
    
    eigenvalues = eigenvalues.real
    return detect_matrix, eigenvalues, eigenvalues/(torch.pow(detect_matrix,2).sum()**0.5)
        

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

def none_or_float(value):
    if value == 'None' or value == 'none':
        return None
    return float(value)

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
    # load weight
    parser.add_argument("--bc_eval", type=int, default=1)   
    parser.add_argument("--weight_num", type=int, default=2, help='how many weights to compute avg')       
    parser.add_argument("--weight_ensemble", type=str, default='mean', help='how to aggregate weights over runnings')       
    parser.add_argument("--weight_path", type=str, help='bc adv path')       
    parser.add_argument("--iter", type=int, default=5, help='K th rebalanced behavior policy.')       
    parser.add_argument("--weight_func", default='linear', choices=['linear', 'exp', 'power'])    
    parser.add_argument("--exp_lambd", default=1.0, type=float)    
    parser.add_argument("--std", default=2.0, type=none_or_float, help="scale weights' standard deviation.")    
    parser.add_argument("--eps", default=0.1, type=none_or_float, help="")    
    parser.add_argument("--eps_max", default=None, type=none_or_float, help="")    
    # TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--normalize", default=True)
    # rebalance
    parser.add_argument("--base_prob", default=0.0, type=float)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--two_sampler", action="store_true")
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--reweight_eval", default=1, type=int)
    parser.add_argument("--reweight_improve", default=1, type=int)
    parser.add_argument("--reweight_constraint", default=1, type=int)
    parser.add_argument("--clip_constraint", default=0, type=int)  # 0: no clip; 1: hard clip; 2 soft clip
    parser.add_argument("--tag", default='', type=str)
    # new params
    parser.add_argument("--qf_layer_norm", default=0, type=int)
    parser.add_argument("--bc_coef", default=1.0, type=float)
    parser.add_argument("--reward_scale", default=1.0, type=float)
    parser.add_argument("--reward_bias", default=0, type=float)
    
    parser.add_argument("--model_freq", default=10000, type=int)
    parser.add_argument("--optimizer", default='adam', type=str)
    
    
    args = parser.parse_args()

    # resample and reweight can not been applied together
    assert not args.resample or not args.reweight

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
    _max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": _max_action,
        "discount": args.discount,
        "tau": args.tau,
        # generate weight
        "iter": args.iter,
        "bc_eval": args.bc_eval,
        "weight_ensemble": args.weight_ensemble,
        "weight_func": args.weight_func,
        "exp_lambd": args.exp_lambd,
        "std": args.std,
        "eps": args.eps,
        "eps_max": args.eps_max,
        # TD3
        "policy_noise": args.policy_noise * _max_action,
        "noise_clip": args.noise_clip * _max_action,
        "policy_freq": args.policy_freq,
        "qf_layer_norm": args.qf_layer_norm,
        # TD3 + BC
        "alpha": args.alpha,
        "bc_coef": args.bc_coef, 
        "reweight_eval": args.reweight_eval, 
        "reweight_improve": args.reweight_improve,
        "reweight_constraint": args.reweight_constraint,
        "clip_constraint": args.clip_constraint,
        "optimizer_type": args.optimizer,
    }

    wandb.init(project="TD3_BC", config={
            "env": args.env, "seed": args.seed, "tag": args.tag,
            "resample": args.resample, "two_sampler": args.two_sampler, "reweight": args.reweight, "p_base": args.base_prob,
            **kwargs
            })

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.batch_size,
        base_prob=args.base_prob, resample=args.resample, reweight=args.reweight, n_step=1, discount=args.discount)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    # save return dist
    # np.save(f'./weights/{args.env}_returns.npy', replay_buffer.returns)
    
    # if 'antmaze' in args.env:
    #     replay_buffer.reward -= 1.0
    replay_buffer.reward = replay_buffer.reward * args.reward_scale + args.reward_bias
    if args.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1

    if args.bc_eval:
        # weight loading module (filename changed)
        weight_list = []
        for seed in range(1, args.weight_num + 1):
            try:
                file_name = args.weight_path%seed
            except:
                file_name = args.weight_path # load the speificed weight
            wp =  f'./weights/{file_name}.npy'
            eval_res = np.load(wp, allow_pickle=True).item()
            num_iter, bc_eval_steps = eval_res['iter'], eval_res['eval_steps']
            assert args.iter <= num_iter
            weight_list.append(eval_res[args.iter])
            print(f'Loading weights from {wp} at {args.iter}th rebalanced behavior policy')
        if args.weight_ensemble == 'mean':
            weight = np.stack(weight_list, axis=0).mean(axis=0)
        elif args.weight_ensemble == 'median':
            weight = np.median(np.stack(weight_list, axis=0), axis=0)
        else:
            raise NotImplementedError
        replay_buffer.replace_weights(weight, args.weight_func, args.exp_lambd, args.std, args.eps, args.eps_max)

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    # time0 = time.time()
    evaluations = []
    q1_models = []
    # q2_models = []
    ntk_list = []
    random_batch_grad_list = []
    fix_batch_grad_list = []
    pi_list = []
    log_similairty = False
    fix_batch = replay_buffer.sample(uniform=True, bs=2560)
    ntk_states, ntk_actions, ntk_next_states = fix_batch[:3] 
    
    QLIMIT = replay_buffer.reward.max() / (1-args.discount) * 1000
    for t in range(int(args.max_timesteps)):
        infos, random_batch_grad = policy.train(replay_buffer, args.two_sampler)
        if (t + 1) % args.log_freq == 0:
            for k, v in infos.items():
                wandb.log({f'train/{k}': v}, step=t+1)
        # Evaluate episode
        # if (t + 1) % args.eval_freq == 0:
        #     print(f"Time steps: {t+1}")
        #     evaluations.append(eval_policy(policy, args.env, args.seed, mean, std, args.eval_episodes))
        #     wandb.log({f'eval/score': evaluations[-1]}, step=t+1)
        #     wandb.log({f'eval/avg10_score': np.mean(evaluations[-min(10, len(evaluations)):])}, step=t+1)
        #     # np.save(f"./results/{file_name}", evaluations)
        #     if args.save_model: policy.save(f"./models/{file_name}")
        
        # if not log_similairty and infos['Q1'] >  QLIMIT:
        #     log_similairty = True
        #     base_model_step = t+1
        #     base_model1 = torch.cat([param.view(-1) for param in policy.critic.q1.parameters()])
        #     base_model2 = torch.cat([param.view(-1) for param in policy.critic.q2.parameters()])
        #     wandb.log({f'q1_similarity': 1}, step=t+1)
        #     wandb.log({f'q2_similarity': 1}, step=t+1)
            
        if (t + 1) % args.model_freq == 0:
            param1 = torch.cat([param.view(-1) for param in policy.critic.q1.parameters()])
            param2 = torch.cat([param.view(-1) for param in policy.critic.q2.parameters()])
            q1_models.append(param1.cpu())
            
            ntk, batch_grads = compute_ntk(policy.critic, ntk_states, ntk_actions)
            ntk_list.append(ntk.cpu())
            
            fix_batch_grad, fix_batch_Q, fix_batch_max_Q, fix_batch_pi = policy.compute_grad(fix_batch)
            pi_sim_varying_state = cosine_similarity_matrix(fix_batch_pi).mean()
            random_batch_grad_list.append(random_batch_grad.cpu())
            fix_batch_grad_list.append(fix_batch_grad.cpu())
            pi_list.append(fix_batch_pi.cpu())
            with torch.no_grad():
                fix_batch_next_pi = policy.actor(ntk_next_states)
            detect_matrix, eigenvalues, normed_eigenvalues = compute_detect_matrix(policy.critic, ntk_next_states, fix_batch_next_pi, batch_grads)
            
            
            wandb.log({
                'stat/data_Q': fix_batch_Q.mean(),
                'stat/data_Q_std': fix_batch_Q.std(),
                'stat/max_Q': fix_batch_max_Q.mean(),
                'stat/max_Q_std': fix_batch_max_Q.std(),
                'stat/pi_sim_varying_state': pi_sim_varying_state,
                'stat/max_eigenvalues': eigenvalues.max().cpu(),
                # 'stat/eigenvalues': eigenvalues.cpu(),
                'stat/max_normed_eigenvalues': normed_eigenvalues.max().cpu(),
                # 'stat/normed_eigenvalues': normed_eigenvalues.cpu(),
                'stat/grad_norm': torch.norm(fix_batch_grad, p=2).cpu(),
                'stat/data_action_norm':ntk_actions.pow(2).sum(dim=-1).sqrt().mean().cpu(),
                'stat/pi_action_norm': fix_batch_pi.pow(2).sum(dim=-1).sqrt().mean().cpu(),
                }, step=t+1)
            
            # if fix_batch_max_Q.mean().item() > 1e10:
            #     exit()
            
        # if log_similairty and (t + 1) % args.model_freq == 0:
        #     wandb.log({f'q1_similarity': cosine_similarity(base_model1, param1)}, step=t+1)
        #     wandb.log({f'q2_similarity': cosine_similarity(base_model2, param2)}, step=t+1)
    
    print("In-distributional actions:")
    print(ntk_actions)       
    print("Policy actions:")
    print(fix_batch_pi)       
            
    # similarity between two vectors
    steps = [(t+1)*args.model_freq for t in range(len(q1_models))] 
    model_sim = [cosine_similarity(m, q1_models[-1]) for m in q1_models]
    
    # similarity between two matrix
    flat_ntks = [torch.reshape(ntk, shape=(-1,)) for ntk in ntk_list]
    ntk_sim = [cosine_similarity(flat_ntk, flat_ntks[-1]) for flat_ntk in flat_ntks]
    
    # similarity between two vectors
    random_grad_sim = [cosine_similarity(g, random_batch_grad_list[-1]) for g in random_batch_grad_list]
    fix_grad_sim = [cosine_similarity(g, fix_batch_grad_list[-1]) for g in fix_batch_grad_list]
    
    # similarity between two batch of vectors
    pi_sim_varying_step = [F.cosine_similarity(pi, pi_list[-1]).mean() for pi in pi_list]

    
    model_data = [[x, y] for (x, y) in zip(steps, model_sim)]
    table1 = wandb.Table(data=model_data, columns = ["steps", "similarity"])
    wandb.log(
{"cosine similarity of q1" : wandb.plot.line(table1, "steps", "similarity",
        title="cosine similarity of q1")})   
    
    ntk_data = [[x, y] for (x, y) in zip(steps, ntk_sim)]
    ntk_table = wandb.Table(data=ntk_data, columns = ["steps", "similarity"])
    wandb.log(
{"cosine similarity of ntk" : wandb.plot.line(ntk_table, "steps", "similarity",
        title="cosine similarity of ntk")})   
    
    random_grad_data = [[x, y] for (x, y) in zip(steps, random_grad_sim)]
    random_grad_table = wandb.Table(data=random_grad_data, columns = ["steps", "similarity"])
    wandb.log(
{"cosine similarity of random batch grad" : wandb.plot.line(random_grad_table, "steps", "similarity",
        title="cosine similarity of random batch grad")})   
    
    fix_grad_data = [[x, y] for (x, y) in zip(steps, fix_grad_sim)]
    fix_grad_table = wandb.Table(data=fix_grad_data, columns = ["steps", "similarity"])
    wandb.log(
{"cosine similarity of fix batch grad" : wandb.plot.line(fix_grad_table, "steps", "similarity",
        title="cosine similarity of fix batch grad")})   
    
    pi_sim_varying_data = [[x, y] for (x, y) in zip(steps, pi_sim_varying_step)]
    pi_sim_varying_table = wandb.Table(data=pi_sim_varying_data, columns = ["steps", "similarity"])
    wandb.log(
{"cosine similarity of pi with varying step" : wandb.plot.line(pi_sim_varying_table, "steps", "similarity",
        title="cosine similarity of pi with varying step")})   
        
    wandb.finish() 
    
    torch.save({
        'steps': steps,
        'q1_models': q1_models,
        'ntks': ntk_list,
        'random_batch_grad_list': random_batch_grad_list,
        'fix_batch_grad_list': fix_batch_grad_list,
        'pi_list': pi_list,
    }, f'results/{args.env}.pt')
        
