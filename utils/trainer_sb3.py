from dataset_utils import UE5CarRLGymEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from model.model import DispMobileNet

env = UE5CarRLGymEnv()
env.create_env()
policy_kwargs = dict(
    normalize_images=False,
    features_extractor_class=DispMobileNet,
    features_extractor_kwargs=dict(action_dim=3, pretrained=False),
)
model_ppo = PPO("CnnPolicy", env, verbose=1,
                batch_size=64, gamma=0.99, tensorboard_log="runs/logs_ppo",
                policy_kwargs={'normalize_images': False}, device="cuda:0", learning_rate=3e-4)

model_ppo.learn(total_timesteps=10000000)

# model_dqn = DQN(
#         "CnnPolicy",
#         env,
#         verbose=1,
#         buffer_size=5000,
#         learning_starts=1000,
#         batch_size=64,
#         target_update_interval=1000,
#         train_freq=4,
#         gamma=0.99,
#         tensorboard_log="logs",
#         exploration_final_eps=1e-6,
#         policy_kwargs=policy_kwargs,
#         learning_rate=1e-4,  # 根据需要调整
#     )
#
# model_dqn.learn(total_timesteps=10000000)