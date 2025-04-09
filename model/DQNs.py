import torch
import numpy as np


class DQN:
    def __init__(self, Qnet, action_dim=3, learning_rate=0.00001, gamma=0.5, epsilon=0.5, target_update=10, device='cuda:0' if torch.cuda.is_available() else 'cpu',
                  loss_func=torch.nn.HuberLoss, use_amp=True):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

        self.q_net = Qnet(self.action_dim).to(self.device)
        self.target_q_net = Qnet(self.action_dim).to(device)

        self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=learning_rate)
        self.count = 0

        self.loss_func = loss_func()

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(enabled=use_amp)

    def take_action(self, state:torch.Tensor):
        self.q_net.eval()
        with torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
            prediction = self.q_net(state)
            if isinstance(prediction, tuple):
                prediction = prediction[1]
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
            action_type = 'random'
        else:
            action = prediction.argmax().item()
            action_type = 'predict'
        return action, action_type, prediction.to('cpu').squeeze().detach().numpy()

    def update(self, transition_dict, is_update=True, is_weights=None):
        if is_update:
            self.q_net.train()
        else:
            self.q_net.eval()
        states = transition_dict['states'].to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = transition_dict['next_states'].to(self.device)
        dones = torch.tensor(transition_dict['dones']).view(-1, 1).to(self.device)

        with torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
            q_values = self.q_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

            if is_update:
                dqn_loss = (self.loss_func(q_values, q_targets)).mean() if is_weights is None else (
                            self.loss_func(q_values, q_targets) * torch.tensor(is_weights).to(self.device)).mean()
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(dqn_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    dqn_loss.backward()
                    self.optimizer.step()

                if self.count % self.target_update == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

                self.count += 1
        return torch.abs(q_values - q_targets).to('cpu').detach().numpy().squeeze()


class DDQN(DQN):
    def __init__(self, Qnet, action_dim=3, learning_rate=0.00001, gamma=0.5, epsilon=0.6, target_update=10, device='cuda:0' if torch.cuda.is_available() else 'cpu',
                 loss_func=torch.nn.HuberLoss, use_amp=True):
        super().__init__(Qnet, action_dim, learning_rate, gamma, epsilon, target_update, device, loss_func, use_amp)

    def update(self, transition_dict, is_update=True, is_weights=None):
        if is_update:
            self.q_net.train()
        else:
            self.q_net.eval()

        states = transition_dict['states'].to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = transition_dict['next_states'].to(self.device)
        dones = torch.tensor(transition_dict['dones']).view(-1, 1).to(self.device)

        with torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
            q_values = self.q_net(states).gather(1, actions)

            with torch.no_grad():
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)

            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

            if is_update:
                dqn_loss = (self.loss_func(q_values, q_targets)).mean() if is_weights is None else (
                    self.loss_func(q_values, q_targets) * torch.tensor(is_weights).to(self.device)).mean()

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(dqn_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    dqn_loss.backward()
                    self.optimizer.step()

                # 更新目标网络
                if self.count % self.target_update == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

                self.count += 1

        return torch.abs(q_values - q_targets).to('cpu').detach().numpy().squeeze()


class DuelingDQN(DQN):
    def __init__(self, Qnet, action_dim=3, learning_rate=0.00001, gamma=0.5, epsilon=0.5, target_update=10, device='cuda:0' if torch.cuda.is_available() else 'cpu',
                 loss_func=torch.nn.HuberLoss, use_amp=True):
        super().__init__(Qnet, action_dim, learning_rate, gamma, epsilon, target_update, device, loss_func, use_amp)

    def update(self, transition_dict, is_update=True, is_weights=None):
        if is_update:
            self.q_net.train()
        else:
            self.q_net.eval()

        states = transition_dict['states'].to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = transition_dict['next_states'].to(self.device)
        dones = torch.tensor(transition_dict['dones']).view(-1, 1).to(self.device)

        with torch.amp.autocast(enabled=self.use_amp, device_type=self.device):
            v_values, a_values = self.q_net(states)
            q_values = (v_values + a_values - torch.mean(a_values, dim=-1, keepdim=True)).gather(1, actions)

            with torch.no_grad():
                v_values_target, a_values_target = self.target_q_net(next_states)
                q_values_target = v_values_target + a_values_target - torch.mean(a_values_target, dim=-1, keepdim=True)

            q_targets = rewards + self.gamma * torch.max(q_values_target, dim=-1, keepdim=True)[0] * (1 - dones)
            if is_update:
                dqn_loss = (self.loss_func(q_values, q_targets)).mean() if is_weights is None else (
                    self.loss_func(q_values, q_targets) * torch.tensor(is_weights).to(self.device)).mean()

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(dqn_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    dqn_loss.backward()
                    self.optimizer.step()

                # 更新目标网络
                if self.count % self.target_update == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

                self.count += 1

        return torch.abs(q_values - q_targets).to('cpu').detach().numpy().squeeze()
