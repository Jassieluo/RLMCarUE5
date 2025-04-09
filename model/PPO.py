import time

import numpy as np

from .model import DispResNet50ActorCritic
from torch import optim
# from torch import nn
from torch.distributions import Categorical
import torch
import torch.nn.functional as F
from tqdm import tqdm


class PPO:
    """
    PPO算法实现类
    """

    def __init__(self, action_dim, ActorCriticNet: DispResNet50ActorCritic, device='cuda:0', use_amp=True,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2, po_epochs=4, batch_size=64, entropy_coef=0.01,
                 learning_rate=3e-4, buffer_size=180):
        self.num_actions = action_dim
        self.device = device
        self.use_amp = use_amp

        # 初始化网络
        self.ActorCriticNet = ActorCriticNet(action_dim).to(device)
        self.optimizer = optim.Adam(self.ActorCriticNet.parameters(), lr=learning_rate)

        # PPO超参数
        self.gamma = gamma  # 折扣因子
        self.gae_lambda = gae_lambda  # GAE参数
        self.clip_param = clip_param  # PPO clip参数
        self.ppo_epochs = po_epochs  # PPO更新epoch数
        self.batch_size = batch_size  # 小批量大小
        self.entropy_coef = entropy_coef  # 熵正则化系数

        # 经验回放缓冲区
        self.buffer = []
        self.buffer_size = buffer_size  # 每次收集这么多经验后再更新

        self.initial_lr_ppo_train = learning_rate
        self.final_lr_ppo_train = learning_rate/10

    def get_action(self, state):
        """
        根据观察值选择动作
        obs: 预处理后的观察值
        返回: 动作, 动作对数概率, 状态值
        """
        with torch.no_grad() and torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
            action_probs, state_value = self.ActorCriticNet(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def compute_gae(self, rewards, values,  dones, next_value):
        """
        计算广义优势估计 (GAE)
        rewards: 奖励序列
        values: 状态值序列
        dones: 是否结束序列
        next_value: 最后一个状态的下一个状态值
        返回: 优势估计, 回报
        """
        gae = 0
        returns = []
        advantages = []

        # 反向计算
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return advantages, returns

    def cosine_lr(self, current_step, total_steps):
        return self.final_lr_ppo_train + 0.5 * (self.initial_lr_ppo_train - self.final_lr_ppo_train) * (1 + np.cos(np.pi * current_step / total_steps))

    def update_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr  # 更新学习率

    def update_lr_params(self, lr):
        self.initial_lr_ppo_train = lr
        self.final_lr_ppo_train = self.initial_lr_ppo_train/10

    def update_model(self, state, actions, old_log_probs, returns, advantages):
        """
        使用PPO更新模型
        """
        # 随机打乱数据
        time.sleep(0.1)
        indices = np.arange(self.buffer_size)
        # 小批量更新
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            pbar = tqdm(range(0, self.buffer_size, self.batch_size), desc=f"PPOTrain epoch: {epoch}")
            epoch_all_loss = 0
            epoch_lr = self.cosine_lr(epoch, self.ppo_epochs)
            self.update_lr_params(lr=epoch_lr)
            for start in pbar:
                end = min(start + self.batch_size, self.buffer_size)
                batch_indices = indices[start:end]

                # 获取小批量数据
                batch_state = state[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 计算新的动作概率和值
                with torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
                    new_action_probs, new_values = self.ActorCriticNet(batch_state)

                    dist = Categorical(new_action_probs)
                    new_log_probs = dist.log_prob(batch_actions)

                    # 计算概率比
                    ratio = (new_log_probs - batch_old_log_probs).exp()

                    # 计算裁剪的PPO目标
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 计算critic损失 (MSE)
                    critic_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                    # 计算熵奖励
                    entropy = dist.entropy().mean()

                    # 总损失
                    loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                    pbar.desc = f"PPOTrain epoch: {epoch}, loss: {loss:.5f}"

                    epoch_all_loss += loss.item()
                    # 梯度下降
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ActorCriticNet.parameters(), 0.5)  # 梯度裁剪
                    self.optimizer.step()
            time.sleep(0.1)
            print(f"Epoch All Loss: {epoch_all_loss:.5f}, Epoch Mean Loss: {epoch_all_loss/(self.buffer_size/self.batch_size):.5f}, Epoch Lr: {epoch_lr}\n")
            time.sleep(0.1)