import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state_action):
        x = self.relu(self.fc1(state_action))
        return self.fc2(x)

def generate_trajectory(length):
    states = [np.random.rand(2)]
    actions = [np.array([np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)]) for k in range(8)]
    trajectory_actions = []

    for _ in range(length - 1):
        action = actions[np.random.randint(0, 8)]
        new_state = states[-1] + action * 0.1 + np.random.rand(2)*0.02
        states.append(new_state)
        trajectory_actions.append(action)

    return np.array(states), np.array(trajectory_actions), np.array(actions)

def compute_ntk(model, x1, x2=None):
    if x2 is None:
        x2 = copy.deepcopy(x1)

    N = x1.size(0)

    # Function to flatten gradients of model parameters
    def flatten_grads(grads):
        return torch.cat([g.view(-1) for g in grads])

    # Compute gradients for each input
    g1 = []
    for i in range(N):
        xi = x1[i].unsqueeze(0)

        # Feedforward the input through the model
        output = model(xi)

        # Zero the model's gradients
        model.zero_grad()

        # Compute the gradient of the output with respect to the model's parameters
        output.backward(torch.ones_like(output), retain_graph=True)

        # Flatten and store the gradients
        grad = flatten_grads([p.grad for p in model.parameters() if p.grad is not None])
        g1.append(grad)

    g2 = []
    for i in range(x2.size(0)):
        xi = x2[i].unsqueeze(0)

        # Feedforward the input through the model
        output = model(xi)

        # Zero the model's gradients
        model.zero_grad()

        # Compute the gradient of the output with respect to the model's parameters
        output.backward(torch.ones_like(output), retain_graph=True)

        # Flatten and store the gradients
        grad = flatten_grads([p.grad for p in model.parameters() if p.grad is not None])
        g2.append(grad)

    # Stack the gradients into a single tensor
    g1 = torch.stack(g1)
    g2 = torch.stack(g2)

    # Compute the NTK matrix using tensor operations
    G = torch.matmul(g1, g2.t())

    return G

def plot_trajectory_with_q_values(states, dataset_actions, max_actions, q_values, max_q_values, iteration):
    plt.figure(figsize=(10, 10))
    plt.scatter(states[:, 0], states[:, 1], color='blue', label='States')
    for i, (state, dataset_action, max_action, q_value, max_q_value) in enumerate(
            zip(states[:-1], dataset_actions, max_actions, q_values, max_q_values)):
        plt.arrow(state[0], state[1], dataset_action[0] * 0.03, dataset_action[1] * 0.03, color='green', width=0.002)
        #print(q_value)
        plt.annotate(f"{q_value:.2f}", (state[0] + dataset_action[0] * 0.03, state[1] + dataset_action[1] * 0.03),
                     fontsize=8, color='green')
        plt.arrow(state[0], state[1], max_action[0] * 0.03, max_action[1] * 0.03, color='red', width=0.002)
        plt.annotate(f"{max_q_value:.2f}", (state[0] + max_action[0] * 0.03, state[1] + max_action[1] * 0.03),
                     fontsize=8, color='red')
    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.savefig(f"./vis/iteration_{iteration}.png")
    plt.close()

def analyze_eigen(model, states, actions, max_actions, gamma):
    state_action_tensors = [torch.tensor(np.hstack((states[i], actions[i])), dtype=torch.float32) for i in
                            range(len(states) - 1)]
    X1 = torch.stack(state_action_tensors).cuda()

    state_action_tensors = [torch.tensor(np.hstack((states[i], max_actions[i])), dtype=torch.float32) for i in
                            range(1, len(states))]
    X2 = torch.stack(state_action_tensors).cuda()

    G = compute_ntk(model, X2, X1) * gamma - compute_ntk(model, X1, X1)
    G = G.cpu().detach().numpy()
    return np.linalg.eig(G)[0]


def q_learning(model, states, actions, action_space, iterations, learning_rate=1e-2, gamma=0.99):
    # 0.95 Adam 1e-3 for random exp
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=2e-2)
    criterion = nn.MSELoss()

    param_history = []
    grad_history = []
    q_value_history = []
    ntk_history = []

    state_action_tensors = [torch.tensor(np.hstack((states[i], actions[i])), dtype=torch.float32) for i in
                            range(len(states)-1)]
    state_action_tensors = torch.stack(state_action_tensors).cuda()

    for _ in range(iterations):
        print(f"iter {_}")
        target_values = []
        max_actions = []
        with torch.no_grad():
            model.eval()
            for i in range(len(states)):
                state = states[i]
                max_q_value = -np.inf
                max_action = None
                for action in action_space:
                    state_action = torch.tensor(np.hstack((state, action)), dtype=torch.float32).cuda()
                    q_value = model(state_action).item()
                    if q_value > max_q_value:
                        max_q_value = q_value
                        max_action = action

                target_values.append(max_q_value)
                max_actions.append(max_action)

        model.train()
        #target_values.append(0)
        q_values = model(state_action_tensors)
        targets = gamma * torch.tensor(target_values[1:]).reshape(-1, 1).cuda()
        q_value_history.append(q_values.detach().mean().item())

        print(q_values.squeeze())
        #print(f"The ratio is {targets.squeeze() / q_values.squeeze()}")
        loss = criterion(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        param_history.append([p.clone().detach().cpu() for p in model.parameters()])
        grad_history.append([p.grad.clone().detach().cpu() for p in model.parameters()])
        #ntk_history.append(compute_ntk(copy.deepcopy(model), copy.deepcopy(state_action_tensors)))
        ntk_history.append(0)

        #print(analyze_eigen(copy.deepcopy(model), states, actions, max_actions, gamma))
        #print(ntk_history[-1])

        # plot_trajectory_with_q_values(states=states, dataset_actions=actions, max_actions=max_actions,
        #                               q_values=q_values.squeeze().cpu().detach().numpy(),
        #                               max_q_values=target_values,
        #                               iteration=_)


    return param_history, grad_history, q_value_history, ntk_history

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def plot_cosine_similarity(param_history, grad_history, ntk_history):
    final_params = torch.cat([p.view(-1) for p in param_history[-1]]).numpy()
    final_grads = torch.cat([g.view(-1) for g in grad_history[-1]]).numpy()

    param_cosine_similarities = [
        cosine_similarity(torch.cat([p.view(-1) for p in params]).numpy(), final_params)
        for params in param_history
    ]

    grad_cosine_similarities = [
        cosine_similarity(torch.cat([g.view(-1) for g in grads]).numpy(), final_grads)
        for grads in grad_history
    ]

    y = torch.stack(ntk_history).reshape(len(ntk_history), -1).cpu().detach().numpy()
    final_ntk = y[-1]
    ntk_similarities = [cosine_similarity(yi, final_ntk) for yi in y]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    ax1.plot(param_cosine_similarities, label="Parameters")
    ax1.set_title("Cosine Similarity of Concatenated Parameters")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cosine Similarity")
    ax1.legend()
    ax1.grid()

    ax2.plot(grad_cosine_similarities, label="Gradients")
    ax2.set_title("Cosine Similarity of Concatenated Gradients")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cosine Similarity")
    ax2.legend()
    ax2.grid()

    ax3.plot(ntk_similarities, label="NTK")
    ax3.set_title("Cosine Similarity of NTK")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Cosine Similarity")
    #ax3.set_yrange(-)
    ax3.legend()
    ax3.grid()

    plt.savefig('vis.png')

    print(f"The cosine similarity between grad and param {cosine_similarity(final_grads, final_params)}")

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)

    np_seed = 1458
    # 1458
    torch_seed = 111
    # 111 explode
    # 233 converge

    state_dim = 2
    action_dim = 2
    hidden_dim = 64
    trajectory_length = 64
    iterations = 400

    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)

    # Two-layer MLP
    model = MLP(state_dim, action_dim, hidden_dim).cuda()
    states, actions, action_space = generate_trajectory(trajectory_length)

    param_history, grad_history, q_value_history, ntk_history = q_learning(model=model, states=states, actions=actions,
                                             action_space=action_space, iterations=iterations)
    #plot_cosine_similarity(param_history, grad_history, ntk_history)

    # np.save("q_value.npy", np.array(q_value_history))

    plt.clf()
    #plt.plot(q_value_history)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(200, len(q_value_history)), 1/np.array(q_value_history[200:]))
    #plt.title("Average Predicted Q-value Along Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("1 / Q-value-error")
    plt.grid()
    plt.savefig("Q-value.png")