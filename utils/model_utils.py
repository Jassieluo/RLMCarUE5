from utils_read import Shared_vectorAndcollision_list, SharedDepthTensor, SharedDispTensor
import math
import torch
from joystick_utils import Tensor3Controll, gamepad, List3Controll
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# import time

agents = [[0.5, 0., 0.], [0., 0., 0.3], [0., 0., -0.3]]

def Sign(x):
    return 1. if x > 0 else -1.

def DistanceNorm2Squire(distanceVector: list):
    rate_x = 1 if abs(distanceVector[0])>=1 or abs(distanceVector[0])<=0.1 else abs(1/distanceVector[0])
    rate_y = 1 if abs(distanceVector[1])>=1 or abs(distanceVector[1])<=0.1 else abs(1/distanceVector[1])
    x = distanceVector[0] * rate_x * rate_y
    y = distanceVector[1] * rate_y * rate_x
    return [Sign(x) if abs(x) >= abs(y) else x / abs(y), Sign(y) if abs(y) >= abs(x) else y / abs(x)]


def DistanceXForward(distanceVector: list, forwardVector: list):
    disX, disY = DistanceNorm2Squire(distanceVector)
    forX, forY = forwardVector
    return disX * forY - disY * forX


def DistanceAngleCosForward(distanceVector: list, forwardVector: list):
    disX, disY = distanceVector
    forX, forY = forwardVector
    return (disX * forX + disY * forY) / math.sqrt((disX ** 2 + disY ** 2) * (forX ** 2 + forY ** 2))


def DistanceAngleCosForwardNorm(distanceVector: list, forwardVector: list):
    disX, disY = DistanceNorm2Squire(distanceVector)
    forX, forY = forwardVector
    return (disX * forX + disY * forY) / math.sqrt((disX ** 2 + disY ** 2) * (forX ** 2 + forY ** 2))


def DistanceAngleForward(distanceVector: list, forwardVector: list):
    disX, disY = distanceVector
    forX, forY = forwardVector
    return math.acos((disX * forX + disY * forY) / math.sqrt((disX ** 2 + disY ** 2) * (forX ** 2 + forY ** 2)))


def DistanceAngleForwardNorm(distanceVector: list, forwardVector: list):
    disX, disY = DistanceNorm2Squire(distanceVector)
    forX, forY = forwardVector
    return math.acos((disX * forX + disY * forY) / math.sqrt((disX ** 2 + disY ** 2) * (forX ** 2 + forY ** 2)))

def NormDistance2ForwardVector(distanceVector: list, forwardVector: list):
    Angle = DistanceAngleForward(distanceVector, forwardVector)
    distance = (
            distanceVector[0]**2 + distanceVector[1]**2)
    disForwardY = distance*math.cos(Angle)
    disForwardX = -distance*math.sin(Angle)*Sign(DistanceXForward(distanceVector, forwardVector))
    return DistanceNorm2Squire([disForwardX, disForwardY])
    # return [disForwardX, disForwardY]

class Trainer:

    def __init__(self, model, epochs=500, base_lr=0.0001, lr_decay0=0.999, lr_decay1=0.9999999,
                 collision_negtive_reward=-50, collision_positive_reward=2, collision_null_reward=0,
                 distance_reward=0.01, ratattion_reward=1, optimizer=torch.optim.SGD,
                 momentum=0.987, weight_decay=0.01):
        self.model = model
        self.epochs = epochs
        self.base_lr = base_lr
        self.lr_decay0 = lr_decay0
        self.lr_decay1 = lr_decay1
        self.total_reward = 0
        self.collision_negtive_reward = collision_negtive_reward
        self.collision_positive_reward = collision_positive_reward
        self.collision_null_reward = collision_null_reward
        self.distance_reward = distance_reward
        self.ratattion_reward = ratattion_reward
        self.optimizer = optimizer(params=model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        self.loss_fun0 = torch.nn.CrossEntropyLoss(reduction='mean')
        self.writer = SummaryWriter('../runs/model0')
        self.best_reward = 0
        self.best_reward_epoch = 1

    def cal_rewards(self, stateList: list):
        collision_reward = self.collision_null_reward
        if stateList[4] > 1:
            collision_reward = self.collision_positive_reward
        elif stateList[4] < -1:
            collision_reward = self.collision_negtive_reward
        return [self.distance_reward / (math.sqrt(stateList[0] ** 2 + stateList[1] ** 2) + self.distance_reward / 2),
                self.ratattion_reward * (DistanceAngleCosForwardNorm(stateList[:2], stateList[2:4])),
                collision_reward]

    def lossAndReward(self, prediction: torch.Tensor, stateList: list):
        NormDistance = DistanceNorm2Squire(stateList[:2])
        rewards = self.cal_rewards(stateList)
        target_state = [1.0, 0.0, 0.0]
        # disForwardX, disForwardY = NormDistance2ForwardVector(NormDistance, stateList[2:4])
        # print(disForwardY, disForwardX, -Sign(DistanceXForward(stateList[:2], stateList[2:4])) if DistanceAngleForwardNorm(stateList[:2], stateList[
        #                                                                                                   2:4]) > 1 else -Sign(
        #     DistanceXForward(stateList[:2], stateList[2:4])) * DistanceAngleForwardNorm(stateList[:2], stateList[2:4]), prediction.squeeze(dim=0)[2])

        # state_loss = (self.loss_fun0(prediction.squeeze(dim=0)[0], torch.tensor(disForwardX)) +
        #               self.loss_fun0(prediction.squeeze(dim=0)[1], torch.tensor(disForwardY))) * (1 / rewards[0]) + \
        # (self.loss_fun0(prediction.squeeze(dim=0)[2], torch.tensor(-Sign(DistanceXForward(stateList[:2], stateList[2:4])) if DistanceAngleForwardNorm(stateList[:2], stateList[
        #                                                                                                   2:4]) > 1 else -Sign(
        #     DistanceXForward(stateList[:2], stateList[2:4])) * DistanceAngleForwardNorm(stateList[:2], stateList[2:4]))) * (1/rewards[1]))
        angle = -DistanceAngleForward(NormDistance, stateList[2:4]) * Sign(DistanceXForward(NormDistance, stateList[2:4]))
        print(angle)
        if angle > math.pi / 16:
            target_state = [-float(prediction[0, 0]), 2.0, -float(prediction[0, 2])]
        elif angle < -math.pi / 16:
            target_state = [-float(prediction[0, 0]), -float(prediction[0, 1]), 2.0]
        state_loss = self.loss_fun0(prediction, prediction + torch.tensor([target_state], dtype=torch.float32) * (1/rewards[1]) * (1/rewards[0]))
        state_loss /= (rewards[2]-self.collision_negtive_reward+0.01)
        return state_loss, rewards

    def update_optimier_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr  # 更新学习率

    def train_epoch(self):
        keep_going = True
        epoch_all_loss = []
        epoch_all_reward = []
        epoch_current_lr = self.base_lr
        while keep_going:
            # Start = time.time()
            self.optimizer.zero_grad()
            stateNow_vector = Shared_vectorAndcollision_list()
            stateNow_depth = SharedDispTensor()
            prediction = self.model(stateNow_depth)
            prediction_index = torch.argmax(prediction, dim=1)
            List3Controll(agents[prediction_index])
            gamepad.update()
            loss, reward = self.lossAndReward(prediction, stateNow_vector)
            epoch_all_loss.append(loss.item())
            epoch_all_reward.append(sum(reward))
            loss.backward()
            self.optimizer.step()
            keep_going = True if reward[2] > (self.collision_negtive_reward/2) else False
            self.update_optimier_lr(self.lr_decay1 * epoch_current_lr)
            epoch_current_lr *= self.lr_decay1
            # End = time.time()
            # print('\r' + '程序运行时间:%s毫秒' % ((End - Start) * 1000), end='', flush=True)
            #
            # print('\r' + f'Last Train Reward: {sum(epoch_all_reward)} ,'
            #              f'   Last Train Loss: {sum(epoch_all_loss)}', end='', flush=True)
            self.writer.add_scalar("Last Train Reward", sum(reward), len(epoch_all_reward))
            self.writer.add_scalar("Last Train Loss", loss.item(), len(epoch_all_loss))
        return epoch_all_loss, epoch_all_reward

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}')
            epoch_all_loss, epoch_all_reward = self.train_epoch()
            self.writer.add_scalar("Epoch Loss", sum(epoch_all_loss), epoch)
            self.writer.add_scalar("Epoch Reward", sum(epoch_all_reward), epoch)
            if self.best_reward < sum(epoch_all_reward):
                self.best_reward = sum(epoch_all_reward)
                self.best_reward_epoch = epoch
                plt.figure(num=0)
                plt.title(f"Best Loss Epoch:{self.best_reward_epoch}")
                plt.xlabel("itels")
                plt.ylabel("Loss")
                plt.plot(range(1, len(epoch_all_loss)+1), epoch_all_loss)
                plt.savefig('runs/model0/best_loss.png')
                plt.figure(num=1)
                plt.title(f"Best Reward Epoch:{self.best_reward_epoch}")
                plt.xlabel("itels")
                plt.ylabel("Reward")
                plt.plot(range(1, len(epoch_all_reward)+1), epoch_all_reward)
                plt.savefig('runs/model0/best_reward.png')
                torch.save(self.model, "runs/model0/model_best.pt")
            self.base_lr *= self.lr_decay0
            self.update_optimier_lr(self.base_lr)
            torch.save(self.model, "runs/model0/model_last.pt")
        gamepad.reset()
        gamepad.update()
        del gamepad