from torchvision import models
import torch.nn as nn
# import gymnasium as gym


class DispMobileNet(nn.Module):
    def __init__(self, action_dim=3, pretrained=True):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)

        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # 修改分类头（根据任务调整）
        self.mobilenet.classifier[3] = nn.Linear(1280, action_dim)  # 输出3维动作

        nn.init.uniform_(self.mobilenet.classifier[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.mobilenet.classifier[-1].bias)

    def forward(self, x):
        return self.mobilenet(x)


class DispMobileVANet(nn.Module):
    def __init__(self, action_dim=3, pretrained=True):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)

        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # 修改分类头（根据任务调整）
        self.mobilenet.classifier[3] = nn.Identity()  # 输出3维动作

        self.V_layer = nn.Linear(1280, 1)
        self.A_layer = nn.Linear(1280, action_dim)

    def forward(self, x):
        x = self.mobilenet(x)
        v = self.V_layer(x)
        a = self.A_layer(x)
        return v, a


class BGRDMobileNet(nn.Module):
    def __init__(self, action_dim=3, pretrained=True):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # 修改分类头（根据任务调整）
        self.mobilenet.classifier[3] = nn.Linear(1024, action_dim)  # 输出3维动作

    def forward(self, x):
        return self.mobilenet(x)


class BGRMobileNet(nn.Module):
    def __init__(self, action_dim=3, pretrained=True):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)

        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            3,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # 修改分类头（根据任务调整）
        self.mobilenet.classifier[3] = nn.Linear(1280, action_dim)  # 输出3维动作

    def forward(self, x):
        return self.mobilenet(x)


class DispResNet50(nn.Module):
    def __init__(self, action_dim=3, pretrained=False):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # 修改分类头（根据任务调整）
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, action_dim)  # 输出3维动作

        nn.init.uniform_(self.resnet.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.resnet.fc.bias)

    def forward(self, x):
        return self.resnet(x)


class DispResNet101(nn.Module):
    def __init__(self, action_dim=3, pretrained=False):
        super().__init__()
        self.resnet = models.resnet101(pretrained=pretrained)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # 修改分类头（根据任务调整）
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, action_dim)  # 输出3维动作

        nn.init.uniform_(self.resnet.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.resnet.fc.bias)

    def forward(self, x):
        return self.resnet(x)


class DispResNet50ActorCritic(nn.Module):
    def __init__(self, action_dim=3, pretrained=False):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        self.actor = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
        )

        self.resnet.fc = nn.Identity()  # 输出3维动作

    def forward(self, x):
        x = self.resnet(x)
        action_probs, state_value = self.actor(x), self.critic(x)
        return action_probs, state_value


class DispResNet101ActorCritic(nn.Module):
    def __init__(self, action_dim=3, pretrained=False):
        super().__init__()
        self.resnet = models.resnet101(pretrained=pretrained)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        self.actor = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
        )

        self.resnet.fc = nn.Identity()  # 输出3维动作

    def forward(self, x):
        x = self.resnet(x)
        action_probs, state_value = self.actor(x), self.critic(x)
        return action_probs, state_value