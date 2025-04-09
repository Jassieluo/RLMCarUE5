import dask
from dask.distributed import Client, LocalCluster
import time
from model.DQN import DQN
import torch
from dataset_utils import UE5CarRLEnv, ReplayBuffer, PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model.model import BGRDMobileNet, DispMobileNet
import cv2


class DaskTrainer:
    def __init__(self, rl_network=DQN, Qnet=DispMobileNet, epochs=500, base_lr=0.00005,
                 lr_decay0=0.99, lr_decay1=0.9999, decay_random=1., optimizer=torch.optim.SGD,
                 momentum=0.987, weight_decay=0.01, env=UE5CarRLEnv,
                 replay_buffer=PrioritizedReplayBuffer, train_buffer_minimal_size=500,
                 batch_size=64, device='cuda:0', use_amp=True):

        # 初始化Dask本地集群
        self.cluster = LocalCluster(n_workers=2, threads_per_worker=1)
        self.client = Client(self.cluster)

        self.device = device
        self.env = env(device=device)
        self.env.create_env()

        self.epochs = epochs
        self.base_lr = base_lr
        self.lr_decay0 = lr_decay0
        self.lr_decay1 = lr_decay1
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.train_buffer_minimal_size = train_buffer_minimal_size

        # 初始化RL网络（需确保可序列化）
        self.rl_network = rl_network(
            action_dim=self.env.action_num,
            learning_rate=base_lr,
            Qnet=Qnet,
            device=device,
            use_amp=use_amp
        )
        self.rl_network.optimizer = optimizer(
            params=self.rl_network.q_net.parameters(),
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # 初始化共享状态（通过Dask Actor模式）
        self.shared_state = {
            'current_done': self.client.submit(lambda: False, actor=True).result(),
            'replay_buffer': replay_buffer(train_buffer_minimal_size * 10),
            'base_lr': base_lr
        }

        self.epochs = epochs
        self.batch_size = batch_size
        self.decay_random = decay_random
        self.writer = SummaryWriter('../runs/model0')
        self.best_reward = 0

    def update_optimizer_lr(self, lr):
        for param_group in self.rl_network.optimizer.param_groups:
            param_group["lr"] = lr

    @dask.delayed
    def collect_experience(self):
        """改为返回完整经验列表"""
        rewards = []
        state = self.env.reset()
        while not self.shared_state['current_done']:
            action, action_type, prediction = self.rl_network.take_action(state)
            next_state, reward, done = self.env.step(action)
            self.shared_state['replay_buffer'].push(state, action, reward, next_state, done)
            rewards.append((reward, action_type, prediction))
            state = next_state.clone()
            if done:
                self.shared_state['current_done'] = True
        return rewards  # 返回完整列表而非生成器

    @dask.delayed
    def update_network(self):
        """分布式网络更新任务"""
        while not self.shared_state['current_done']:
            if self.shared_state['replay_buffer'].size() >= self.train_buffer_minimal_size:
                # 从共享缓冲区采样
                if isinstance(self.shared_state['replay_buffer'], PrioritizedReplayBuffer):
                    transition_dict, data_idx, is_weights = self.shared_state['replay_buffer'].sample(self.batch_size)
                    errors = self.rl_network.update(transition_dict, True, is_weights)
                    self.shared_state['replay_buffer'].update_priorities(data_idx, errors)
                else:
                    transition_dict, _, _ = self.shared_state['replay_buffer'].sample(self.batch_size)
                    self.rl_network.update(transition_dict, True, None)

                # 学习率衰减
                new_lr = self.shared_state['base_lr'] * self.lr_decay1
                self.update_optimizer_lr(new_lr)
                self.shared_state['base_lr'] = new_lr

                self.rl_network.epsilon *= self.decay_random

    def train_epoch(self):
        """并行执行一个epoch的训练"""
        # 提交并行任务
        exp_task = self.collect_experience()
        update_task = self.update_network()

        # 收集结果
        total_reward = 0
        for result in exp_task.compute():
            reward, action_type, pred = result
            total_reward += reward
            print(f'Reward: {reward}', end='\r')
        return total_reward

    def train(self):
        print("Start Training with Dask...")
        try:
            for epoch in range(1, self.epochs + 1):
                print(f'\nEpoch {epoch}')
                epoch_reward = self.train_epoch()

                # 记录和保存模型
                self.writer.add_scalar("Epoch Reward", epoch_reward, epoch)
                if epoch_reward > self.best_reward:
                    self.best_reward = epoch_reward
                    torch.save(self.rl_network.q_net.state_dict(), "../runs/model0/model_best.pt")

                # 全局学习率衰减
                self.shared_state['base_lr'] *= self.lr_decay0
                self.update_optimizer_lr(self.shared_state['base_lr'])

        finally:
            self.client.close()
            self.cluster.close()


if __name__ == "__main__":
    trainer = DaskTrainer()
    trainer.train()