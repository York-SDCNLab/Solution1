import gym
import torch
import numpy as np
from torch.optim import Adam
from core.data.buffer import SequenceRolloutBuffer
from generator import main as generator_main  # 假设 generator.py 位于同级目录
from core.envs.qlabs_sim import QLabsSim
from core.envs.wrappers import ActionRewardResetWrapper, CollectWrapper

class TD3Agent:
    def __init__(self, env, actor_lr, critic_lr, gamma, tau, buffer_size):
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.buffer = SequenceRolloutBuffer(buffer_size, 1, (6,))

        # Initialize the actor and critic networks
        self.actor = Actor(self.env.action_size)  # Actor 类的构造函数将被初始化为接受一个动作向量，其维度为 2
        self.critic = Critic(6, self.env.action_size) # Critic 类的构造函数将被初始化为接受一个动作向量，其维度为 2

        # Initialize the target networks
        self.actor_target = Actor(self.env.action_size)
        self.critic_target = Critic(6, self.env.action_size)

        # Set up the optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

        # Copy the parameters of the target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data * (1.0 - self.tau) + target_param.data * self.tau)

    def select_action(self, state, evaluation_mode=False):
        mets = {}
        state = torch.FloatTensor(state)
        action = self.actor(state).detach().numpy()
        if not evaluation_mode:
            action += np.random.normal(0, 0.1, size=action.shape)  # Add noise for exploration
        return np.clip(action, -1, 1), mets

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) > 100:  # Minimum buffer size before learning can occur
            samples = self.buffer.sample(batch_size=64)

            states, actions, rewards, next_states, dones = samples

            # Convert samples to PyTorch tensors
            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # # Compute the target Q-values
            # target_actions = self.actor_target(next_states)
            # target_q_values = self.critic_target(next_states, target_actions)
            # target_q_values = rewards + (1 - dones) * self.gamma * target_q_values

            # 计算目标Q值
            with torch.no_grad():  # 确保在计算目标Q值时不计算梯度
                target_actions = self.actor_target(next_states)
                target_q_values = self.critic_target(next_states, target_actions)
                best_next_q_values = torch.max(target_q_values, dim=1).values  # 选择最佳的Q值
                target_q_values = rewards + (1 - dones) * self.gamma * best_next_q_values

            # Compute critic loss
            predicted_q_values = self.critic(states, actions)
            critic_loss = torch.mean((predicted_q_values - target_q_values) ** 2)

            # # Compute actor loss
            # target_actions = self.actor_target(next_states)
            # target_q_values = self.critic(next_states, target_actions)
            # actor_loss = -torch.mean(target_q_values)

            # 计算演员损失
            best_next_actions = torch.argmax(self.actor_target(next_states), dim=1)
            actor_loss = -torch.mean(self.critic(next_states, best_next_actions) * torch.ones_like(predicted_q_values))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update of the target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)


class Actor(torch.nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(6, 128)  # 假设状态维度为6
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        continuous_actions = torch.nn.functional.tanh(self.fc3(x))  # 输出的是经过 tanh 激活函数处理的连续动作值
        return continuous_actions


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_fc1 = torch.nn.Linear(state_dim, 128)
        self.state_fc2 = torch.nn.Linear(128, 128)
        self.state_fc3 = torch.nn.Linear(128, 1)

        self.action_fc1 = torch.nn.Linear(action_dim, 128)
        self.action_fc2 = torch.nn.Linear(128, 128)
        self.action_fc3 = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        state_values = torch.nn.functional.relu(self.state_fc1(state))
        state_values = torch.nn.functional.relu(self.state_fc2(state_values))
        state_values = self.state_fc3(state_values)

        action_values = torch.nn.functional.relu(self.action_fc1(action))
        action_values = torch.nn.functional.relu(self.action_fc2(action_values))
        action_values = self.action_fc3(action_values)

        q_values = state_values + action_values
        return q_values

# To use the TD3Agent, you would typically initialize it and then call select_action and store_transition
# in the generator.py file, and learn in trainer.py or another training loop.