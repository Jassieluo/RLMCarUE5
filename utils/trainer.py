import copy
import time

import numpy as np

from model.DQNs import DQN, DDQN, DuelingDQN
from model.PPO import PPO
import torch
from dataset_utils import UE5CarRLEnv, ReplayBuffer, PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model.model import BGRDMobileNet, DispMobileNet, DispMobileVANet, BGRMobileNet, DispResNet50, DispResNet50ActorCritic, DispResNet101ActorCritic
import threading
import cv2


class DQNsTrainer:
    def __init__(self, rl_network=DQN, Qnet=DispMobileNet, state_type="DispMap", base_lr=0.0001, lr_decay0=0.999999,
                 lr_decay1=0.99999999,  decay_randam=0.9999,
                 optimizer=torch.optim.SGD, momentum=0.987, weight_decay=0.01, env=UE5CarRLEnv,
                 replay_buffer=PrioritizedReplayBuffer, train_buffer_minimal_size=180, batch_size=64, device='cuda:0', use_amp=True, use_thread=True,
                 project_path="../runs/DQNs/"):
        self.device = device
        self.env = env(device=device, state_type=state_type)
        self.env.create_env()
        self.rl_network = rl_network(action_dim=self.env.action_num, learning_rate=base_lr, Qnet=Qnet, device=device, use_amp=use_amp)
        self.rl_network.optimizer = optimizer(params=self.rl_network.q_net.parameters(), lr=base_lr, momentum=momentum,
                                              weight_decay=weight_decay)
        self.state_type = state_type
        self.base_lr = base_lr
        self.lr_decay0 = lr_decay0
        self.lr_decay1 = lr_decay1
        self.total_reward = 0
        self.momentum = momentum
        self.weight_decay = weight_decay
        if replay_buffer == ReplayBuffer:
            self.replay_buffer = replay_buffer(train_buffer_minimal_size * 15)
        elif replay_buffer == PrioritizedReplayBuffer:
            self.replay_buffer = replay_buffer(train_buffer_minimal_size * 10)
        self.train_buffer_minimal_size = train_buffer_minimal_size
        self.writer = SummaryWriter('../runs/DQNs')
        self.best_reward = 0
        self.best_reward_epoch = 1
        self.batch_size = batch_size
        self.decay_randam = decay_randam

        self.ues_amp = use_amp

        self.use_thread = use_thread
        self.current_done = None
        self.epoch_all_reward = []
        self.model_train_counter = 0
        # self.thread_update_replay_buffer = threading.Thread(target=self.update_replay_buffer)
        # self.thread_update_rl_network = threading.Thread(target=self.update_rl_network)
        self.thread_update_rl_network = None
        self.errors = [0.]
        self.project_path = project_path

    def update_optimizer_lr(self, lr):
        for param_group in self.rl_network.optimizer.param_groups:
            param_group["lr"] = lr  # 更新学习率

    def state_show(self, state):
        if self.state_type == "DispMap":
            normalized_disparity = cv2.normalize(state.to('cpu').squeeze().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
            cv2.imshow('state', colored_disparity)
        elif self.state_type == "BGRMap":
            bgr_state = cv2.normalize(state.to('cpu').squeeze().detach().numpy().reshape(256, 512, 3), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            cv2.imshow('state', bgr_state)
        cv2.waitKey(1)

    # def update_replay_buffer(self):
    #     self.epoch_all_reward = []
    #     state = self.env.reset()
    #     while not self.current_done:
    #         action, action_type, prediction = self.rl_network.take_action(state)
    #         self.disp_state_show(state)
    #         next_state, reward, self.current_done = self.env.step(action)
    #         self.epoch_all_reward.append(reward)
    #         self.replay_buffer.push(state, action, reward, next_state, self.current_done)
    #         state = next_state.clone()
    #         print('\r' + f'Last Train All Reward: {sum(self.epoch_all_reward)}, Last Train Step Reward: {reward}, Action Type: {action_type}, Prediction: {prediction}, Replay Buffer Size: {self.replay_buffer.size()}, Sum Errors: {sum(self.errors)}', end='', flush=True)
    #         self.writer.add_scalar("Last Train Reward", reward, len(self.epoch_all_reward))

    def update_rl_network_once(self, epoch_current_lr):
        if isinstance(self.replay_buffer, ReplayBuffer):
            transition_dict, _, _ = self.replay_buffer.sample(self.batch_size)
            self.rl_network.update(transition_dict, True, None)
            self.update_optimizer_lr(self.lr_decay1 * epoch_current_lr)
        elif isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            transition_dict, data_idx, is_weights = self.replay_buffer.sample(self.batch_size)
            errors = self.rl_network.update(transition_dict, True, is_weights)
            self.replay_buffer.update_priorities(data_idx, errors)
            self.errors = errors
            self.update_optimizer_lr(self.lr_decay1 * epoch_current_lr)

    def update_rl_network(self):
        epoch_current_lr = self.base_lr
        while not self.current_done:
            if self.replay_buffer.size() >= self.train_buffer_minimal_size:
                self.update_rl_network_once(epoch_current_lr)
                epoch_current_lr *= self.lr_decay1
                self.model_train_counter += 1
            self.rl_network.epsilon *= self.decay_randam
            cv2.waitKey(1)

    def train_epoch(self):
        self.current_done = False
        epoch_all_reward = []
        epoch_current_lr = self.base_lr
        state = self.env.reset()
        if self.use_thread:
            self.thread_update_rl_network = threading.Thread(target=self.update_rl_network)
            self.thread_update_rl_network.start()
        while not self.current_done:
            action, action_type, prediction = self.rl_network.take_action(state)
            self.state_show(state)
            next_state, reward, done = self.env.step(action)
            epoch_all_reward.append(reward)
            self.replay_buffer.push(state, action, reward, next_state, done)
            if self.replay_buffer.size() >= self.train_buffer_minimal_size and not self.use_thread:
                self.update_rl_network_once(epoch_current_lr)
                epoch_current_lr *= self.lr_decay1
                self.model_train_counter += 1
            state = next_state.clone()
            self.rl_network.epsilon *= self.decay_randam
            print('\r' + f'Last Train All Reward: {sum(epoch_all_reward):.5f}, Last Train Step Reward: {reward:.5f}, Action Type: {action_type},'
                   f' Prediction: {prediction}, Replay Buffer Size: {self.replay_buffer.size()}, Model Trained Count: {self.model_train_counter}, Sum Errors: {sum(self.errors):.5f}',
                   end='', flush=True)
            self.current_done = done
            self.writer.add_scalar("Last Train Reward", reward, len(epoch_all_reward))
        if self.use_thread:
            self.thread_update_rl_network.join()
        return epoch_all_reward

    def train(self, epochs):
        print("Start Training..., device: ", torch.cuda.get_device_name(self.device))
        for epoch in range(1, epochs + 1):
            print(f'\nEpoch {epoch}')
            epoch_all_reward = self.train_epoch()
            time.sleep(0.5)
            self.writer.add_scalar("Epoch Reward", sum(epoch_all_reward), epoch)
            if self.best_reward < sum(epoch_all_reward):
                self.best_reward = sum(epoch_all_reward)
                self.best_reward_epoch = epoch
                plt.figure(num=0)
                plt.title(f"Best Reward Epoch:{self.best_reward_epoch}")
                plt.xlabel("itels")
                plt.ylabel("Reward")
                plt.plot(range(1, len(epoch_all_reward) + 1), epoch_all_reward)
                plt.savefig(self.project_path + 'best_reward.png')
                torch.save(self.rl_network.q_net, self.project_path + "q_net_best.pt")
            self.base_lr *= self.lr_decay0
            self.update_optimizer_lr(self.base_lr)
            torch.save(self.rl_network.q_net, self.project_path + "q_net_last.pt")
        self.env.close()


class PPOTrainer:
    def __init__(self, rl_network=PPO, ActorCriticNet=DispResNet50ActorCritic, state_type="DispMap", initial_lr=3e-4, final_lr=1e-6,
                 optimizer=torch.optim.Adam, weight_decay=0.01, env=UE5CarRLEnv, collect_interval=5,
                 train_buffer_minimal_size=180, max_buff_size_rate=15, batch_size=32, device='cuda:0', use_amp=True,
                 project_path="../runs/PPO/"):
        self.device = device
        self.env = env(device=device, state_type=state_type)
        self.env.create_env()
        self.rl_network = rl_network(action_dim=self.env.action_num, learning_rate=initial_lr, ActorCriticNet=ActorCriticNet,
                                     device=device, use_amp=use_amp, batch_size=batch_size, buffer_size=train_buffer_minimal_size)
        self.rl_network.optimizer = optimizer(params=self.rl_network.ActorCriticNet.parameters(), lr=initial_lr,
                                              weight_decay=weight_decay)
        self.state_type = state_type
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.current_lr = initial_lr
        self.total_reward = 0
        self.weight_decay = weight_decay

        self.writer = SummaryWriter(project_path)
        self.batch_size = batch_size

        self.use_amp = use_amp
        self.buffer = []
        self.collect_interval = collect_interval

        self.best_epoch_reward = []
        self.last_epoch_reward = []

        self.errors = [0.]

        self.buffer_size = train_buffer_minimal_size
        self.train_buffer_minimal_size = train_buffer_minimal_size
        self.project_path = project_path
        self.max_buff_size_rate = max_buff_size_rate

    def cosine_lr(self, current_step, total_steps):
        return self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1 + np.cos(np.pi * current_step / total_steps))

    def state_show(self, state):
        if self.state_type == "DispMap":
            normalized_disparity = cv2.normalize(state.to('cpu').squeeze().detach().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
            cv2.imshow('state', colored_disparity)
        elif self.state_type == "BGRMap":
            bgr_state = cv2.normalize(state.to('cpu').squeeze().detach().numpy().reshape(256, 512, 3), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            cv2.imshow('state', bgr_state)
        cv2.waitKey(1)

    def collect_experience(self):
        """
        收集经验到缓冲区
        返回: 平均回合奖励
        """
        state = self.env.get_state()
        episode_reward = 0
        episode_count = 0

        collect_count = 0

        self.buffer = []  # 清空缓冲区

        while len(self.buffer) < self.buffer_size:
            self.state_show(state)
            # 获取动作
            action, log_prob, value = self.rl_network.get_action(state)

            # 执行动作
            next_state, reward, done = self.env.step(action)

            # 存储经验
            if collect_count % self.collect_interval == 0 or done:
                self.buffer.append({
                    'state': copy.copy(state),  # 存储原始state以节省内存
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value
                })

                episode_reward += reward
                state = next_state

                collect_count = 0

                print('\r' + f'Last Collect Step Reward: {reward:.5f}, Prediction: {action}, Replay Buffer Size: {len(self.buffer)}', end='', flush=True)

                if done:
                    time.sleep(1.5)
                    self.last_epoch_reward.append(episode_reward)
                    episode_count += 1
                    episode_reward = 0
                    state = self.env.reset()

            collect_count += 1

        self.env.stop()
        # 计算最后一个next_value
        with torch.no_grad() and torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
            _, next_value = self.rl_network.ActorCriticNet(state)
            next_value = next_value.item()

        # 准备数据
        state = torch.concatenate([transition['state'] for transition in self.buffer], dim=0)
        actions = [transition['action'] for transition in self.buffer]
        rewards = [transition['reward'] for transition in self.buffer]
        dones = [transition['done'] for transition in self.buffer]
        log_probs = [transition['log_prob'] for transition in self.buffer]
        values = [transition['value'] for transition in self.buffer]

        # 计算GAE和returns
        advantages, returns = self.rl_network.compute_gae(rewards, values, dones, next_value)

        # 标准化优势
        advantages = (np.array(advantages) - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 将数据转换为tensor
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        return state, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor

    def train(self, epochs, PPOTrainEpochs):
        """
        训练循环
        num_updates: 更新次数
        """
        print("Start Training..., device: ", torch.cuda.get_device_name(self.device))
        self.rl_network.ppo_epochs = PPOTrainEpochs
        for epoch in range(epochs):
            print(f"\nTotal Train epoch: {epoch + 1}/{epochs}, lr: {self.current_lr:.8f}")

            # 收集经验
            print("Start Collecting Experience:")
            state, actions, old_log_probs, returns, advantages = self.collect_experience()
            self.env.stop()
            if sum(self.last_epoch_reward) >= sum(self.best_epoch_reward):
                self.best_epoch_reward = self.last_epoch_reward.copy()
                plt.figure(num=1)
                plt.title(f"Best Reward Epoch:{epoch}")
                plt.xlabel("itels")
                plt.ylabel("Reward")
                plt.plot(range(1, len(self.best_epoch_reward) + 1), self.best_epoch_reward)
                plt.savefig(self.project_path + 'best_reward.png')
                torch.save(self.rl_network.ActorCriticNet, self.project_path + "ActorCriticNet_best.pt")

            self.writer.add_scalar("Epoch Reward", sum(self.last_epoch_reward), epoch)
            plt.figure(num=0)
            plt.title(f"Last Reward Epoch:{epoch}")
            plt.xlabel("itels")
            plt.ylabel("Reward")
            plt.plot(range(1, len(self.last_epoch_reward) + 1), self.last_epoch_reward)
            plt.savefig(self.project_path + 'last_reward.png')
            # torch.save(self.rl_network.ActorCriticNet, self.project_path + "ActorCriticNet_last.pt")
            # 更新模型
            print("\nStart Training PPO ActorCriticNet:")
            time.sleep(0.1)
            self.current_lr = self.cosine_lr(epoch, epochs)
            self.rl_network.update_optimizer_lr(self.current_lr)
            self.rl_network.update_lr_params(self.current_lr)
            self.rl_network.update_model(state, actions, old_log_probs, returns, advantages)
            self.buffer_size = int(self.buffer_size + 0.5 * self.train_buffer_minimal_size) if epoch < self.max_buff_size_rate else self.buffer_size
            self.rl_network.buffer_size = self.buffer_size


if __name__ == '__main__':
    # DQNs_trainer = DQNsTrainer(replay_buffer=PrioritizedReplayBuffer, use_amp=True, use_thread=True, rl_network=DDQN, Qnet=DispResNet50, state_type="DispMap")
    # DQNs_trainer.train(epochs=500)
    ppo_trainer = PPOTrainer(ActorCriticNet=DispResNet101ActorCritic, collect_interval=2, initial_lr=0.01, train_buffer_minimal_size=360)
    ppo_trainer.train(100, 15)