import collections
import copy
import math
import random

import cv2
import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from joystick_utils import GamePad
from utils_read import SharedUETrainMessageReadTools


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # SumTree的节点存储
        self.pointers = np.arange(capacity)     # 叶子节点对应的数据索引

    def update(self, data_idx, priority):
        """更新指定数据索引的优先级"""
        tree_idx = data_idx + self.capacity - 1  # 转换为树节点索引
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def _propagate(self, tree_idx, delta):
        """向上传播更新"""
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, s):
        """根据采样值获取数据和优先级"""
        idx = self._retrieve(0, s)
        data_idx = self.pointers[idx - self.capacity + 1]
        return data_idx, self.tree[idx]

    def _retrieve(self, idx, s):
        """递归查找对应的叶子节点"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0  # 初始最大优先级

        # 初始化SumTree
        self.sum_tree = SumTree(capacity)

        # 数据存储结构（使用numpy数组预分配空间）
        self.states = [0 for _ in range(capacity)]
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = [0 for _ in range(capacity)]
        self.dones = np.zeros(capacity, dtype=np.int64)

        self.write_pos = 0
        self.size_ = 0

    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        # 存储数据
        self.states[self.write_pos] = state
        self.actions[self.write_pos] = action
        self.rewards[self.write_pos] = reward
        self.next_states[self.write_pos] = next_state
        self.dones[self.write_pos] = done

        # 更新SumTree优先级（新样本赋予当前最大优先级）
        priority = self.max_priority ** self.alpha
        self.sum_tree.update(self.write_pos, priority)

        # 更新指针和缓冲区大小
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size_ = min(self.size_ + 1, self.capacity)

    def sample(self, batch_size):
        """采样一个批量的经验"""
        assert self.size_ > 0, "Buffer is empty"

        data_indices = []
        priorities = []

        # 计算分段采样区间
        segment = self.sum_tree.total_priority / batch_size

        # 采样并获取数据索引
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            a, b = min(a, b), max(a, b)
            s = np.random.uniform(a, b)
            data_idx, priority = self.sum_tree.get(s)
            data_indices.append(data_idx)
            priorities.append(priority)

        # 计算重要性采样权重
        probabilities = np.array(priorities) / self.sum_tree.total_priority
        weights = (self.size_ * probabilities) ** -self.beta
        weights /= weights.max()  # 归一化

        # 提取数据
        states = [self.states[i] for i in data_indices]
        actions = self.actions[data_indices]
        rewards = self.rewards[data_indices]
        next_states = [self.next_states[i] for i in data_indices]
        dones = self.dones[data_indices]

        return ({'states': torch.concatenate(states, dim=0),
                 'actions': actions,
                 'next_states': torch.concatenate(next_states, dim=0),
                 'rewards': rewards,
                 'dones': dones
                 }, data_indices, weights)

    def update_priorities(self, data_indices, td_errors):
        """更新样本优先级"""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for data_idx, priority in zip(data_indices, priorities):
            self.sum_tree.update(data_idx, priority)
        self.max_priority = max(priorities.max(), self.max_priority)

    def size(self):
        return self.size_


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))   # state:BGRD|D:Tensor  action:int  reward:float  next_state:BGRD|D:tensor  done:int

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return {'states': torch.concatenate(state, dim=0),
                'actions': action,
                'next_states': torch.concatenate(next_state, dim=0),
                'rewards': reward,
                'dones': done
                }, None, None

    def size(self):
        return len(self.buffer)


class UE5CarRLEnv:
    def __init__(self, device='cuda:0',
                 state_type="DispMap",
                 R_max=0.95,
                 R_min=-1.,
                 R_clip=0.8,
                 alpha=20/400,
                 beta=3/200,
                 gamma=25/1,
                 k=3.0/400,
                 delta=0.1/200
                 ):
        self.forward = 1.0
        self.turn_right = 1.0
        self.turn_left = 1.0
        self.stop_action = 0.0
        self.action_num = 3

        self.device = device
        self.state_type = state_type

        self.state = None
        self.counts = 0

        self.game_pad = GamePad()
        self.UETrainMessageReadTools = SharedUETrainMessageReadTools(device=device)

        self.last_vector_and_collision = None

        self.R_max = R_max
        self.R_min = R_min
        self.R_clip = R_clip
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.delta = delta
        self.prev_target_dist = None

    def create_env(self):
        self.UETrainMessageReadTools.create_vector_recv_map()
        if self.state_type == "DispMap":
            self.UETrainMessageReadTools.create_disp_recv_map()
        elif self.state_type == "BGRMap":
            self.UETrainMessageReadTools.create_camera_l_recv_map()

    def reset(self):
        self.game_pad.press_button(self.game_pad.START)
        self.game_pad.gamepad.update()
        self.game_pad.gamepad.reset()
        self.game_pad.gamepad.update()
        self.counts = 0
        self.last_vector_and_collision = self.UETrainMessageReadTools.get_shared_vector_and_collision_list()
        # return self.UETrainMessageReadTools.get_shared_camera_l_disp_norm_tensor_bgrd()
        if self.state_type == "DispMap":
            return self.UETrainMessageReadTools.get_shared_disp_norm_linear_tensor()
        elif self.state_type == "BGRMap":
            return self.UETrainMessageReadTools.get_shared_camera_l_norm_tensor()

    def calculate_reward(self, vector_and_collision_list:list):
        collision_status = vector_and_collision_list[-1]
        if collision_status > 0.5:
            self.prev_target_dist = None
            self.last_vector_and_collision = None
            return self.R_max * 10
        elif collision_status < -0.5:
            self.prev_target_dist = None
            self.last_vector_and_collision = None
            return self.R_min * 10

        # 初始化距离
        current_target_dist = math.hypot(vector_and_collision_list[0], vector_and_collision_list[1])
        if self.prev_target_dist is None or self.last_vector_and_collision is None:
            self.prev_target_dist = current_target_dist
            return 0.1  # 首步不计算奖励

        # 目标接近奖励（Delta Distance）
        delta_dist = (self.prev_target_dist - current_target_dist) * np.exp(-self.k * current_target_dist/4)
        self.prev_target_dist = current_target_dist

        # 方向对齐奖励（余弦相似度）
        target_vec = np.array([vector_and_collision_list[0], vector_and_collision_list[1]])
        facing_vec = np.array([vector_and_collision_list[2], vector_and_collision_list[3]])
        target_vec_norm = np.linalg.norm(target_vec)
        facing_vec_norm = np.linalg.norm(facing_vec)

        if target_vec_norm < 1e-6 or facing_vec_norm < 1e-6:
            cos_sim = 0.0
        else:
            cos_sim = np.dot(target_vec, facing_vec) / (target_vec_norm * facing_vec_norm)

        # 障碍物安全奖励（指数衰减）
        obs_distance = math.hypot(vector_and_collision_list[4], vector_and_collision_list[5])
        obs_penalty = self.gamma * np.exp(-self.k * obs_distance)

        last_obs_distance = math.hypot(self.last_vector_and_collision[4], self.last_vector_and_collision[5])
        last_obs_penalty = self.gamma * np.exp(-self.k * last_obs_distance)

        self.last_vector_and_collision = copy.copy(vector_and_collision_list)

        total_obs_penalty = obs_penalty - last_obs_penalty
        # total_obs_penalty = max(-0.15, total_obs_penalty)
        total_obs_penalty = np.clip(total_obs_penalty, -0.15, 1.0)
        # 合成稠密奖励
        dense_reward = (
                self.alpha * delta_dist +
                self.beta * cos_sim -
                total_obs_penalty -
                self.delta
        )

        # 奖励限幅
        clipped_reward = np.clip(dense_reward, -self.R_clip, self.R_clip)
        if abs(clipped_reward) > self.R_clip:
            clipped_reward = 0.0001
            print("error reward:", clipped_reward)
        return clipped_reward * 10

    def step(self, action):
        if action == 0:
            self.game_pad.left_joystick(0., self.forward)
            self.game_pad.left_trigger(0.)
            self.game_pad.right_trigger(0.)
            self.game_pad.gamepad.update()
        if action == 1:
            self.game_pad.right_trigger(self.turn_right)
            self.game_pad.left_joystick(0., 0.)
            self.game_pad.left_trigger(0)
            self.game_pad.gamepad.update()
        if action == 2:
            self.game_pad.left_trigger(self.turn_left)
            self.game_pad.left_joystick(0., 0.)
            self.game_pad.right_trigger(0)
            self.game_pad.gamepad.update()
        if self.state_type == "DispMap":
            next_state = self.UETrainMessageReadTools.get_shared_disp_norm_linear_tensor()
        elif self.state_type == "BGRMap":
            next_state = self.UETrainMessageReadTools.get_shared_camera_l_norm_tensor()
        vector_collision = self.UETrainMessageReadTools.get_shared_vector_and_collision_list()
        # print(vector_collision, end='', flush=True)
        done = 1 if vector_collision[-1] < -1 else 0
        reward = self.calculate_reward(vector_collision)
        self.last_vector_and_collision = vector_collision
        return next_state, reward, done

    def stop(self):
        self.game_pad.left_joystick(0.0001, 0.0001)
        self.game_pad.right_trigger(0.0001)
        self.game_pad.left_trigger(0.0001)
        self.game_pad.gamepad.update()

    def get_state(self):
        if self.state_type == "DispMap":
            state = self.UETrainMessageReadTools.get_shared_disp_norm_linear_tensor()
        elif self.state_type == "BGRMap":
            state = self.UETrainMessageReadTools.get_shared_camera_l_norm_tensor()
        return state

    def close(self):
        self.reset()
        self.game_pad.gamepad.reset()
        self.game_pad.gamepad.update()
        self.game_pad.gamepad.stop()
        self.game_pad.gamepad.disconnect()
        del self.game_pad.gamepad


class UE5CarRLGymEnv(gymnasium.Env):
    """ 自定义的UE5车辆控制环境，符合Gym标准接口 """

    def __init__(
            self,
            device='cuda:0',
            R_max=0.95,
            R_min=-1.,
            R_clip=0.8,
            alpha=100 / 400,
            beta=9 / 200,
            gamma=50 / 1,
            k=4.0 / 400,
            delta=0.1 / 200
    ):
        super(UE5CarRLGymEnv, self).__init__()

        # ========== 环境内部的一些参数 ==========
        self.forward = 1.0
        self.turn_right = 1.0
        self.turn_left = 1.0
        self.action_num = 3

        self.device = device

        self.state = None
        self.counts = 0

        self.game_pad = GamePad()
        self.UETrainMessageReadTools = SharedUETrainMessageReadTools(device=device)

        self.last_vector_and_collision = None

        self.R_max = R_max
        self.R_min = R_min
        self.R_clip = R_clip
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.delta = delta
        self.prev_target_dist = None

        # ========== 定义动作空间（3个离散动作） ==========
        self.action_space = spaces.Discrete(3)

        # ========== 定义观测空间（假设是 [1,1,256,512] 且取值在 [0,1] 之间） ==========
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 256, 512),
            dtype=np.float32
        )

    def create_env(self):
        """若需要在环境初始化阶段执行一些UE侧的注册/连接操作，可在外部调用这个方法"""
        # self.UETrainMessageReadTools.create_camera_l_recv_map()
        self.UETrainMessageReadTools.create_vector_recv_map()
        self.UETrainMessageReadTools.create_disp_recv_map()

    def reset(self, seed=None, options=None):
        """重置环境，返回 (obs, info)。"""
        # ========== 如果需要设置随机种子，可加下面这行 ==========
        super().reset(seed=seed)

        self.game_pad.press_button(self.game_pad.START)
        self.game_pad.gamepad.update()
        self.game_pad.gamepad.reset()
        self.game_pad.gamepad.update()
        self.counts = 0

        # 记录上一帧的向量和碰撞信息
        self.last_vector_and_collision = self.UETrainMessageReadTools.get_shared_vector_and_collision_list()

        # 从UE获取初始观测
        # 这里示例用的是 "深度图/视差图"；你也可以改成别的方法
        self.state = self.UETrainMessageReadTools.get_shared_disp_np_linear_norm()

        # Gym新版API要求reset返回 (obs, info)
        info = {}
        return self.state, info

    def step(self, action):
        """执行一次动作，返回 (obs, reward, done, truncated, info)。"""
        # 根据离散动作的id执行不同操作
        if action == 0:
            self.game_pad.left_joystick(0., self.forward)
            self.game_pad.left_trigger(0.)
            self.game_pad.right_trigger(0.)
        elif action == 1:
            self.game_pad.right_trigger(self.turn_right)
            self.game_pad.left_joystick(0., 0.)
            self.game_pad.left_trigger(0)
        elif action == 2:
            self.game_pad.left_trigger(self.turn_left)
            self.game_pad.left_joystick(0., 0.)
            self.game_pad.right_trigger(0)

        # 推送动作到手柄
        self.game_pad.gamepad.update()

        # 获取下一帧观测
        next_state = self.UETrainMessageReadTools.get_shared_disp_np_linear_norm()
        # 获取环境中的向量 & 碰撞信息
        vector_collision = self.UETrainMessageReadTools.get_shared_vector_and_collision_list()

        # 根据碰撞信息判断是否结束
        # 这里你用 collision<-1 作为结束条件
        done = bool(vector_collision[-1] < -1)

        # 计算奖励
        reward = self.calculate_reward(vector_collision)

        # 更新内部状态
        self.last_vector_and_collision = vector_collision
        self.state = next_state

        # Gym新版API的第4个返回值表示 truncated，这里根据任务需要
        truncated = False

        # info 字典可以存一些调试数据
        info = {}

        return next_state, reward, done, truncated, info

    def calculate_reward(self, vector_and_collision_list: list):
        """根据向量与碰撞信息计算并返回一个奖励值。"""
        collision_status = vector_and_collision_list[-1]
        # 如果是正碰撞（collision_status>0.5）则给一个最大奖励 R_max
        if collision_status > 0.5:
            self.prev_target_dist = None
            self.last_vector_and_collision = None
            return self.R_max
        # 如果是负碰撞（collision_status<-0.5）则给一个最小奖励 R_min
        elif collision_status < -0.5:
            self.prev_target_dist = None
            self.last_vector_and_collision = None
            return self.R_min

        # ========== 以下是你原先的奖励计算逻辑，保持不变即可 ==========
        current_target_dist = math.hypot(vector_and_collision_list[0], vector_and_collision_list[1])
        if self.prev_target_dist is None or self.last_vector_and_collision is None:
            self.prev_target_dist = current_target_dist
            return 0.1

        delta_dist = (self.prev_target_dist - current_target_dist) * np.exp(-self.k * current_target_dist / 4)
        self.prev_target_dist = current_target_dist

        target_vec = np.array([vector_and_collision_list[0], vector_and_collision_list[1]])
        facing_vec = np.array([vector_and_collision_list[2], vector_and_collision_list[3]])
        target_vec_norm = np.linalg.norm(target_vec)
        facing_vec_norm = np.linalg.norm(facing_vec)
        if target_vec_norm < 1e-6 or facing_vec_norm < 1e-6:
            cos_sim = 0.0
        else:
            cos_sim = np.dot(target_vec, facing_vec) / (target_vec_norm * facing_vec_norm)

        obs_distance = math.hypot(vector_and_collision_list[4], vector_and_collision_list[5])
        obs_penalty = self.gamma * np.exp(-self.k * obs_distance)

        last_obs_distance = math.hypot(self.last_vector_and_collision[4], self.last_vector_and_collision[5])
        last_obs_penalty = self.gamma * np.exp(-self.k * last_obs_distance)

        self.last_vector_and_collision = copy.copy(vector_and_collision_list)

        total_obs_penalty = obs_penalty - last_obs_penalty
        total_obs_penalty = np.clip(total_obs_penalty, -0.15, 1.0)

        dense_reward = self.alpha * delta_dist + self.beta * cos_sim - total_obs_penalty - self.delta
        clipped_reward = np.clip(dense_reward, -self.R_clip, self.R_clip)

        if abs(clipped_reward) > self.R_clip:
            clipped_reward = 0.0001
            print("error reward:", clipped_reward)

        return clipped_reward

    def render(self, mode='human'):
        """在窗口中可视化观测数据等（可选）。"""
        if self.state is not None:
            # 你原先的可视化逻辑
            # 注意：Gym一般要求返回 numpy 数组，而不是 torch.Tensor；
            # 如果 self.state 是 torch.Tensor，请先 .cpu().numpy() 再可视化
            if hasattr(self.state, 'detach'):
                disp_np = self.state.detach().cpu().numpy()
            else:
                disp_np = self.state

            # 以下是你用 cv2 展示图像的示例
            # 根据你自己的 shape 调整 squeeze/transpose 等操作
            normalized_disparity = cv2.normalize(
                disp_np.squeeze(), None, 0, 255,
                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
            )
            cv2.imshow('state', normalized_disparity)
            cv2.waitKey(1)

    def close(self):
        """环境关闭时，释放资源。"""
        # 如果需要在close时重置或断开连接，则保留下面这些操作
        self.reset()
        self.game_pad.gamepad.reset()
        self.game_pad.gamepad.update()
        self.game_pad.gamepad.stop()
        self.game_pad.gamepad.disconnect()
        del self.game_pad.gamepad


if __name__ == '__main__':
    # env = UE5CarRLEnv()
    # env.create_env()
    env = UE5CarRLGymEnv()
    env.create_env()
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import DQN
    check_env(env)
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=5000, learning_starts=10000,
                batch_size=128, target_update_interval=1000, train_freq=4, gamma=0.99, tensorboard_log="logs", exploration_final_eps=1e-6,
                policy_kwargs={'normalize_images': False})#0.995?
    model.learn(total_timesteps=100000000)

