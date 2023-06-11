import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class ValueModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def train(render=False, epochs=60, policy_lr=0.1, value_lr=0.01, gamma=0.99):
    env = gym.make('CartPole-v1', render_mode="human")
    D = env.observation_space.shape[0]
    # print(D)
    K = env.action_space.n
    # print(K)

    # models
    policy_model = PolicyModel(D, 32, K)    # hidden_layer_sizes
    value_model = ValueModel(D, 32, 1)

    # optimizers
    policy_optimizer = optim.Adagrad(policy_model.parameters(), lr=policy_lr)
    value_optimizer = optim.SGD(value_model.parameters(), lr=value_lr)

    for i in range(epochs):
        # play 1 game
        done = False
        obs = (env.reset())[0]
        # print(obs[0])
        total_reward = 0

        total_rewards = np.array([])

        states = []
        actions = []
        rewards = []
        iters = 0
        while not done and iters < 2000:
            if render:
                env.render()
            # choose an action
            state_tensor = torch.tensor(obs).float().unsqueeze(0)
            # print(state_tensor.shape)
            action_probs = policy_model(state_tensor)
            # print(action_probs)
            action_probs_np = action_probs.detach().numpy().flatten()
            action = np.random.choice(len(action_probs_np), p=action_probs_np)
            # print(action)
            # take action
            # print(env.step(action))
            next_obs, reward, done, _, _ = env.step(action)
            # save results
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            obs = next_obs
            # print(obs)
            total_reward += reward
            iters += 1

        total_rewards = np.append(total_rewards, total_reward)

        # calculate returns and advantages
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns).float()

        # convert states and actions to tensors
        states_tensor = torch.tensor(np.array(states)).float()
        actions_tensor = torch.tensor(actions)

        # train the value model
        value_preds = value_model(states_tensor)
        value_loss = nn.MSELoss()(value_preds, torch.unsqueeze(returns, 1))
        value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        value_optimizer.step()
        # train the policy model
        # print(returns)
        # print(value_preds)
        advantages = returns - value_preds.flatten().detach()
        # print(advantages)
        probs = torch.sum(torch.mul(nn.functional.one_hot(torch.tensor(actions), num_classes=K), 
            policy_model(states_tensor)), 1)
        # print(probs)
        # print(probs.shape)
        # print(advantages.shape)
        policy_loss = -torch.sum(torch.mul(advantages, torch.log(probs)))
        policy_optimizer.zero_grad()
        policy_loss.backward() # ?
        policy_optimizer.step()

    return policy_model, value_model, total_rewards

def main():
    pmodel, vmodel, total_rewards = train(render=True)
    print(total_rewards.mean())


if __name__ == '__main__':
    main()

   
