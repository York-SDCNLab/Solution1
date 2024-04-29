import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core.envs.qlabs_sim import QLabsSim, pure_pursuit

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)

        # Q2 architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, 256)
        self.layer_5 = nn.Linear(256, 256)
        self.layer_6 = nn.Linear(256, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)

        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            # 从回放缓冲区中随机采样批量数据
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(batch_states).to(device)
            next_state = torch.FloatTensor(batch_next_states).to(device)
            action = torch.FloatTensor(batch_actions).to(device)
            reward = torch.FloatTensor(batch_rewards).to(device)
            done = torch.FloatTensor(batch_dones).to(device)

            # 根据当前策略与噪声获取下一步的动作
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # 计算目标 Q 值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # 获取当前 Q 估计
            current_Q1, current_Q2 = self.critic(state, action)

            # 计算 critic 损失
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # 更新 critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 延迟策略更新
            if it % policy_freq == 0:
                # 计算 actor 损失
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # 更新 actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 软更新目标网络
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用示例
if __name__ == "__main__":
    env = QLabsSim()
    state_dim = 6  # 根据QLabsSim返回的状态维度
    action_dim = 2  # 速度和转向
    max_action = 1.0  # 假设最大动作值为1，这可能需要根据环境具体动作空间调整

    policy = TD3(state_dim, action_dim, max_action).to(device)
    
    episodes = 100  # 总回合数

    for episode in range(episodes):
        obs = env.reset()
        state = obs["state"]
        episode_reward = 0
        
        while True:
            action = policy.select_action(np.array(state))
            next_obs, _, done, _ = env.step(action)
            next_state = next_obs["state"]
            
            # 奖励计算逻辑需要根据环境具体情况定义
            reward = ...  # 例如基于与目标位置的距离或达成目标的情况

            policy.replay_buffer.add((state, next_state, action, reward, float(done)))
            state = next_state
            episode_reward += reward
            
            policy.train(1, 100)  # 1次迭代，每次迭代100个样本
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
