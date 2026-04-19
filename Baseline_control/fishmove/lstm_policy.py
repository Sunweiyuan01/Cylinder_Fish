import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, List, Tuple, Type, Union, Any, Optional, Callable

class LstmExtractor(nn.Module):
    """
    特征提取器，使用LSTM处理序列信息
    """
    def __init__(
        self,
        feature_dim: int,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        
        # 初始化隐藏状态
        self.lstm_hidden_states = None
    
    def forward(self, features: th.Tensor) -> th.Tensor:
        # 重塑输入以匹配LSTM期望的形状 [batch_size, seq_len, feature_dim]
        batch_size = features.shape[0]
        seq_features = features.reshape(batch_size, 1, -1)
        
        # 如果没有隐藏状态或批次大小变化，重新初始化
        if (
            self.lstm_hidden_states is None
            or self.lstm_hidden_states[0].shape[1] != batch_size
        ):
            self.lstm_hidden_states = (
                th.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=features.device),
                th.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=features.device),
            )
        
        # 前向传播
        lstm_out, self.lstm_hidden_states = self.lstm(
            seq_features, self.lstm_hidden_states
        )
        
        # 提取最后一个时间步的输出
        features = lstm_out[:, -1, :]
        return features
    
    def reset_states(self, batch_size: int = 1, device: th.device = th.device("cpu")):
        """重置LSTM隐藏状态"""
        self.lstm_hidden_states = (
            th.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device),
            th.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device),
        )

class MlpLstmPolicy(ActorCriticPolicy):
    """
    结合MLP和LSTM的策略
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        enable_critic_lstm: bool = True,
        **kwargs,
    ):
        # 保存LSTM参数
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.enable_critic_lstm = enable_critic_lstm
        
        # 初始化父类
        super(MlpLstmPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
        # 创建actor和critic的LSTM特征提取器
        self.actor_lstm = LstmExtractor(
            self.features_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
        )
        
        if enable_critic_lstm:
            self.critic_lstm = LstmExtractor(
                self.features_dim,
                lstm_hidden_size=lstm_hidden_size,
                lstm_layers=lstm_layers,
            )
        else:
            self.critic_lstm = None
    
    def forward(self, obs, deterministic=False):
        """前向传播，输出动作和值函数"""
        features = self.extract_features(obs)
        
        # 通过LSTM处理特征
        actor_features = self.actor_lstm(features)
        
        if self.enable_critic_lstm:
            critic_features = self.critic_lstm(features)
        else:
            critic_features = features
        
        # 获取动作分布
        latent_pi = self.mlp_extractor.policy_net(actor_features)
        action_distribution = self._get_action_dist_from_latent(latent_pi)
        
        # 获取值函数估计
        latent_vf = self.mlp_extractor.value_net(critic_features)
        values = self.value_net(latent_vf)
        
        # 采样动作
        actions = action_distribution.get_actions(deterministic=deterministic)
        log_probs = action_distribution.log_prob(actions)
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        """评估给定动作的价值和概率"""
        features = self.extract_features(obs)
        
        # 通过LSTM处理特征
        actor_features = self.actor_lstm(features)
        
        if self.enable_critic_lstm:
            critic_features = self.critic_lstm(features)
        else:
            critic_features = features
        
        # 获取动作分布
        latent_pi = self.mlp_extractor.policy_net(actor_features)
        action_distribution = self._get_action_dist_from_latent(latent_pi)
        
        # 获取值函数估计
        latent_vf = self.mlp_extractor.value_net(critic_features)
        values = self.value_net(latent_vf)
        
        # 计算动作的log概率和熵
        log_probs = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        
        return values, log_probs, entropy
    
    def reset_lstm_states(self):
        """重置LSTM隐藏状态"""
        device = next(self.parameters()).device
        self.actor_lstm.reset_states(device=device)
        if self.enable_critic_lstm:
            self.critic_lstm.reset_states(device=device)

# 注册自定义策略
from stable_baselines3.common.policies import register_policy

register_policy("MlpLstmPolicy", MlpLstmPolicy)