import os
import argparse
import torch
import numpy as np
import multiprocessing as mp
from training import SharedTrainingManager, train_with_rank
from lstm_policy import MlpLstmPolicy  # 导入LSTM策略

def parse_args():
    parser = argparse.ArgumentParser(description='Parallel CFD Training with RL')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--timesteps', type=int, default=20000, help='Total timesteps per worker')
    parser.add_argument('--use-lstm', action='store_true', help='Use LSTM policy')
    parser.add_argument('--lateral-line', action='store_true', help='Use lateral line sensors')
    parser.add_argument('--flow-field', type=str, default='random', 
                        choices=['no_obstacle', 'single_obstacle', 'multiple_obstacles', 'random'],
                        help='Flow field type')
    parser.add_argument('--fusion', action='store_true', help='Perform model fusion after training')
    return parser.parse_args()

def model_fusion(model_paths, output_path="./saved_models/fused_model.zip"):
    """
    简单的模型融合方法 - 参数平均
    """
    from stable_baselines3 import PPO
    
    # 加载所有模型
    models = [PPO.load(path) for path in model_paths if os.path.exists(path)]
    
    if not models:
        print("No valid models found for fusion")
        return
    
    # 创建一个新模型，与第一个模型相同架构
    fused_model = PPO(
        policy=models[0].__class__.__name__,
        env=models[0].env,
        policy_kwargs=models[0].policy_kwargs,
        verbose=1
    )
    
    # 获取所有模型的状态字典
    state_dicts = [model.policy.state_dict() for model in models]
    
    # 平均模型参数
    fused_state_dict = fused_model.policy.state_dict()
    for key in fused_state_dict:
        # 确保所有模型都有这个参数
        if all(key in sd for sd in state_dicts):
            # 计算平均值
            fused_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    
    # 加载融合的参数
    fused_model.policy.load_state_dict(fused_state_dict)
    
    # 保存融合模型
    fused_model.save(output_path)
    print(f"Fused model saved to {output_path}")

def main():
    args = parse_args()
    
    print(f"Starting parallel training with {args.workers} workers")
    print(f"Configuration: LSTM={args.use_lstm}, "
          f"Lateral Line={args.lateral_line}, "
          f"Flow Field={args.flow_field}")
    
    # 创建共享管理器
    manager = SharedTrainingManager(args.workers)
    
    # 创建进程
    processes = []
    for rank in range(args.workers):
        p = mp.Process(
            target=train_with_rank, 
            args=(rank, args.workers, manager, args.timesteps, args.use_lstm)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 获取最佳模型
    best_model_path = manager.get_best_model_path()
    best_reward = manager.get_best_reward()
    print(f"\nTraining completed. Best model: {best_model_path} with reward: {best_reward:.2f}")
    
    # 复制最佳模型
    if best_model_path:
        import shutil
        final_best_path = "./saved_models/best_model.zip"
        shutil.copy(best_model_path, final_best_path)
        print(f"Best model copied to {final_best_path}")
    
    # 如果启用了模型融合
    if args.fusion:
        print("Performing model fusion...")
        # 收集每个worker的最终模型
        model_paths = [f"./saved_models/worker_{rank}/final_model.zip" for rank in range(args.workers)]
        model_fusion(model_paths, "./saved_models/fused_model.zip")

if __name__ == "__main__":
    # 设置多处理方法
    mp.set_start_method('spawn', force=True)
    main()