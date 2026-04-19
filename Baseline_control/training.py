import os
import numpy as np
import torch
import csv
import multiprocessing as mp
import time
from EnvFluent import FluentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# 共享信息管理器
class SharedTrainingManager:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.reward_queue = mp.Queue()
        self.model_queue = mp.Queue()
        self.best_reward = mp.Value('d', -float('inf'))
        self.best_model_path = mp.Array('c', b'\x00' * 500)
        self.lock = mp.Lock()
        self.worker_status = mp.Array('i', [0] * num_workers)  # 0: 未启动, 1: 运行中, 2: 错误, 3: 完成
        
    def update_worker_status(self, rank, status):
        self.worker_status[rank] = status
        
    def update_best_model(self, rank, reward, model_path):
        with self.lock:
            if reward > self.best_reward.value:
                self.best_reward.value = reward
                path_bytes = model_path.encode('utf-8')[:499]
                self.best_model_path.value = path_bytes + b'\x00' * (500 - len(path_bytes))
                return True
        return False
    
    def get_best_model_path(self):
        with self.lock:
            return self.best_model_path.value.decode('utf-8').rstrip('\x00')
    
    def get_best_reward(self):
        with self.lock:
            return self.best_reward.value

# 增强的回调函数（自动保存本地/全局best + 保存VecNormalize）
class EnhancedCallback(BaseCallback):
    def __init__(self, save_path, rank, manager, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.rank = rank
        self.manager = manager
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.best_mean_reward = -np.inf
        self.consecutive_failures = 0  # 连续失败计数
        self.max_consecutive_failures = 5  # 最大连续失败次数
        
        # 日志路径
        self.reward_log_path = os.path.join(save_path, f"rewards_rank{rank}.csv")
        self.action_log_folder = os.path.join(save_path, f"actions_rank{rank}")
        self.performance_log_path = os.path.join(save_path, f"performance_rank{rank}.csv")
        os.makedirs(self.action_log_folder, exist_ok=True)
        
        # 初始化日志文件
        with open(self.reward_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Episode", "Reward", "Mean_Reward", "Best_Global_Reward", "Success", "Failure_Reason", "Episode_Length"])
        
        with open(self.performance_log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                "Episode", "Final_Target_Distance", "Min_Obstacle_Distance", "Simulation_Time", 
                "Success_Rate", "Avg_Turning_Action", "Avg_Period_Action", "Consecutive_Failures"
            ])
            
        self.step_counter = 0
        self.episode_file = None
        self.episode_writer = None
        
        # 性能统计
        self.episode_turning_actions = []
        self.episode_period_actions = []
        self.min_obstacle_distance = float('inf')
        self.success_count = 0

        # 本地/全局最佳模型固定路径（用于断点续训）
        self.local_saved_model = os.path.join(self.save_path, "saved_model.zip")
        self.local_saved_vecnorm = os.path.join(self.save_path, "saved_vecnormalize.pkl")
        self.global_saved_model = os.path.join("./saved_models", "saved_model.zip")
        self.global_saved_vecnorm = os.path.join("./saved_models", "saved_vecnormalize.pkl")

        os.makedirs("./saved_models", exist_ok=True)

    def _save_checkpoint(self, model_path, vecnorm_path):
        # 保存模型
        self.model.save(model_path)
        # 保存VecNormalize
        try:
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vecnorm.save(vecnorm_path)
        except Exception as e:
            print(f"[Rank {self.rank}] Warning: failed to save VecNormalize: {e}")

    def _on_step(self):
        try:
            reward = self.locals.get('rewards', [0])[0]
            done = self.locals.get('dones', [False])[0]
            
            # 记录步骤信息
            if self.episode_file is None:
                episode_filename = os.path.join(
                    self.action_log_folder, f"episode_{self.episode_count + 1}_actions.csv")
                self.episode_file = open(episode_filename, 'w', newline='', buffering=1)
                self.episode_writer = csv.DictWriter(
                    self.episode_file, 
                    fieldnames=[
                        "step", "simulation_time", "fish_x", "fish_y", "fish_theta", 
                        "turning_action", "period_action", "obstacle_distance", "target_distance",
                        "reward", "success", "failed", "failure_reason"
                    ]
                )
                self.episode_writer.writeheader()
                
                # 重置episode统计
                self.episode_turning_actions = []
                self.episode_period_actions = []
                self.min_obstacle_distance = float('inf')
            
            # 记录动作信息
            info = self.locals.get('infos', [{}])[0]
            actions = self.locals.get('actions', [0])
            action = actions[0] if len(actions) > 0 else 0
            
            # 收集统计信息
            turning_action = info.get("turning_action", 0)
            period_action = info.get("period_action", 1)
            obstacle_distance = info.get("obstacle_distance", float('inf'))
            
            self.episode_turning_actions.append(turning_action)
            self.episode_period_actions.append(period_action)
            self.min_obstacle_distance = min(self.min_obstacle_distance, obstacle_distance)
            
            self.step_counter += 1
            self.episode_writer.writerow({
                "step": self.step_counter,
                "simulation_time": info.get("simulation_time", 0),
                "fish_x": info.get("simulation_time", 0) and info.get("fish_position", [0, 0])[0] or info.get("fish_position", [0, 0])[0],
                "fish_y": info.get("fish_position", [0, 0])[1],
                "fish_theta": info.get("fish_orientation", 0),
                "turning_action": turning_action,
                "period_action": period_action,
                "obstacle_distance": obstacle_distance,
                "target_distance": info.get("target_distance", float('inf')),
                "reward": float(reward),
                "success": info.get("success", False),
                "failed": info.get("failed", False),
                "failure_reason": info.get("failure_reason", "")
            })
            
            self.current_episode_reward += reward
            
            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                
                # 检查连续失败
                failure_reason = info.get("failure_reason", "")
                if failure_reason in ["fluent_connection_lost", "fluent_exception", "fluent_step_exception"]:
                    self.consecutive_failures += 1
                    print(f"[Rank {self.rank}] Fluent连接失败 ({self.consecutive_failures}/{self.max_consecutive_failures})")
                else:
                    self.consecutive_failures = 0
                
                # 如果连续失败太多次，标记worker为错误状态
                if self.consecutive_failures >= self.max_consecutive_failures:
                    print(f"[Rank {self.rank}] 连续失败次数过多，标记为错误状态")
                    self.manager.update_worker_status(self.rank, 2)  # 错误状态
                    return False  # 停止训练
                
                # 更新成功统计
                if info.get("success", False):
                    self.success_count += 1
                
                # 计算平均奖励
                if len(self.episode_rewards) >= 10:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                else:
                    mean_reward = np.mean(self.episode_rewards)
                
                # 计算成功率
                success_rate = self.success_count / self.episode_count
                
                # 每个回合保存一个快照（原始命名）
                model_path = os.path.join(self.save_path, f"model_rank{self.rank}_ep{self.episode_count}.zip")
                self.model.save(model_path)
                
                # 如果是当前进程的最佳模型，更新“本地最优 saved_model.zip”
                best_status = ""
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # 保存本地best（用于断点续训）
                    self._save_checkpoint(self.local_saved_model, self.local_saved_vecnorm)

                    # 尝试刷新全局最优
                    is_global_best = self.manager.update_best_model(self.rank, mean_reward, model_path)
                    if is_global_best:
                        best_status = "GLOBAL BEST!"
                        # 同步全局best到固定路径，便于跨worker续训
                        try:
                            self._save_checkpoint(self.global_saved_model, self.global_saved_vecnorm)
                        except Exception as e:
                            print(f"[Rank {self.rank}] Warning: failed to update global saved_model: {e}")
                    else:
                        best_status = "Local Best"                
                # 获取全局最佳奖励
                global_best = self.manager.get_best_reward()                
                # 记录奖励日志
                with open(self.reward_log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        self.episode_count, 
                        self.current_episode_reward, 
                        mean_reward,
                        global_best,
                        info.get("success", False),
                        info.get("failure_reason", ""),
                        self.step_counter
                    ])
                
                # 记录性能日志
                avg_turning = np.mean(self.episode_turning_actions) if self.episode_turning_actions else 0
                avg_period = np.mean(self.episode_period_actions) if self.episode_period_actions else 1
                
                with open(self.performance_log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        self.episode_count,
                        info.get("target_distance", float('inf')),
                        self.min_obstacle_distance,
                        info.get("simulation_time", 0),
                        success_rate,
                        avg_turning,
                        avg_period,
                        self.consecutive_failures
                    ])
                    
                print(f"[Rank {self.rank:02d}|Ep {self.episode_count:03d}] "
                      f"Reward: {self.current_episode_reward:.2f} | "
                      f"Mean10: {mean_reward:.2f} | "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Target Dist: {info.get('target_distance', float('inf')):.2f} | "
                      f"Global Best: {global_best:.2f} {best_status}")
                
                # 重置episode统计
                if self.episode_file:
                    self.episode_file.close()
                self.episode_file = None
                self.episode_writer = None
                self.step_counter = 0
                self.current_episode_reward = 0
                
        except Exception as e:
            print(f"Error in callback for rank {self.rank}: {str(e)}")
            return False
            
        return True

def _build_env_with_optional_resume(rank, log_path, norm_obs=True, norm_reward=True, clip_obs=10.0, local_saved_vecnorm=None, global_saved_vecnorm=None):
    """
    创建环境并尝试恢复 VecNormalize。
    优先加载本地 worker 的 saved_vecnormalize.pkl；若不存在，尝试全局的。
    """
    def make_env():
        def _init():
            env = FluentEnv(max_steps=800, simu_name=f"CFD_{rank}")
            return Monitor(env, os.path.join(log_path, f"monitor"))
        return _init

    base_env = DummyVecEnv([make_env()])
    # 恢复 VecNormalize（优先本地，再全局），若都无则新建
    if local_saved_vecnorm and os.path.exists(local_saved_vecnorm):
        print(f"Worker {rank}: Loading VecNormalize from {local_saved_vecnorm}")
        env = VecNormalize.load(local_saved_vecnorm, base_env)
        env.training = True
    elif global_saved_vecnorm and os.path.exists(global_saved_vecnorm):
        print(f"Worker {rank}: Loading GLOBAL VecNormalize from {global_saved_vecnorm}")
        env = VecNormalize.load(global_saved_vecnorm, base_env)
        env.training = True
    else:
        env = VecNormalize(base_env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs)
    return env

def train_with_rank(rank, num_workers, manager, total_timesteps, use_lstm=False):
    try:
        # 分阶段启动，避免同时启动造成资源冲突
        startup_delay = rank * 30  # 每个worker间隔30秒启动
        print(f"Worker {rank}: 等待{startup_delay}秒后启动...")
        time.sleep(startup_delay)
        
        # 更新worker状态
        manager.update_worker_status(rank, 1)  # 运行中
        
        # 设置随机种子
        torch.manual_seed(rank * 42)
        np.random.seed(rank * 42)
        
        # 设置设备和路径
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_path = os.path.join("./saved_models", f"worker_{rank}")
        log_path = os.path.join("./logs", f"worker_{rank}")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # 固定的断点续训路径（本地与全局）
        local_saved_model = os.path.join(save_path, "saved_model.zip")
        local_saved_vecnorm = os.path.join(save_path, "saved_vecnormalize.pkl")
        global_saved_model = os.path.join("./saved_models", "saved_model.zip")
        global_saved_vecnorm = os.path.join("./saved_models", "saved_vecnormalize.pkl")
        
        # 创建环境（并尝试恢复VecNormalize）
        env = _build_env_with_optional_resume(
            rank,
            log_path,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            local_saved_vecnorm=local_saved_vecnorm,
            global_saved_vecnorm=global_saved_vecnorm
        )
        
        # 选择网络架构
        if use_lstm:
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 512, 256], vf=[512, 256, 128]),
                lstm_hidden_size=256,
                enable_critic_lstm=True,
                lstm_layers=2
            )
            policy_type = "MlpLstmPolicy"
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 512, 256], vf=[512, 256, 128]),
                activation_fn=torch.nn.ReLU
            )
            policy_type = "MlpPolicy"
        
        # 如果存在 saved_model.zip，则加载继续训练；否则新建模型
        model = None
        if os.path.exists(local_saved_model):
            print(f"Worker {rank}: Resuming from {local_saved_model}")
            model = PPO.load(local_saved_model, env=env, device=device)
        elif os.path.exists(global_saved_model):
            print(f"Worker {rank}: Resuming from GLOBAL {global_saved_model}")
            model = PPO.load(global_saved_model, env=env, device=device)
        else:
            model = PPO(
                policy_type, 
                env, 
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=32,     # 减少steps以降低内存使用
                batch_size=16,    # 减少batch size
                n_epochs=10, 
                gamma=0.995,
                gae_lambda=0.98, 
                clip_range=0.2, 
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0,
                tensorboard_log=log_path, 
                device=device
            )
        
        # 创建回调（会自动维护本地/全局 best 的 saved_model）
        callback = EnhancedCallback(save_path, rank, manager, verbose=1)        
        print(f"Worker {rank}: Starting training for {total_timesteps} steps")
        print(f"Worker {rank}: Environment observation space: {env.observation_space}")
        print(f"Worker {rank}: Environment action space: {env.action_space}")        
        # 开始训练
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)        
        # 保存最终模型 & 当前VecNormalize
        final_path = os.path.join(save_path, f"final_model.zip")
        model.save(final_path)
        try:
            env.save(os.path.join(save_path, f"vec_normalize.pkl"))
        except Exception as e:
            print(f"Worker {rank}: Warning saving vec_normalize.pkl: {e}")        
        # 更新状态为完成
        manager.update_worker_status(rank, 3)
        print(f"Worker {rank}: Training completed successfully")        
    except KeyboardInterrupt:
        print(f"Worker {rank}: Interrupted. Saving backup...")
        try:
            backup_path = os.path.join(save_path, f"interrupted_model.zip")
            model.save(backup_path)
        except:
            pass
        manager.update_worker_status(rank, 2)
        
    except Exception as e:
        print(f"Worker {rank}: Error: {str(e)}")
        import traceback
        traceback.print_exc()
        manager.update_worker_status(rank, 2)
        
    finally:
        try:
            env.close()
            print(f"Worker {rank}: Environment closed")
        except Exception as e:
            print(f"Worker {rank}: Error closing environment: {e}")

def monitor_workers(manager, num_workers, check_interval=60):
    """监控worker状态"""
    while True:
        time.sleep(check_interval)
        
        status_counts = [0, 0, 0, 0]  # [未启动, 运行中, 错误, 完成]
        for i in range(num_workers):
            status_counts[manager.worker_status[i]] += 1
        
        print(f"\n=== Worker状态监控 ===")
        print(f"未启动: {status_counts[0]}, 运行中: {status_counts[1]}, 错误: {status_counts[2]}, 完成: {status_counts[3]}")
        print(f"全局最佳奖励: {manager.get_best_reward():.2f}")
        
        # 如果所有worker都完成或出错，退出监控
        if status_counts[1] == 0:  # 没有运行中的worker
            print("所有worker已停止运行")
            break

def main():
    # 设置多进程启动方法
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    else:  # Unix/Linux
        mp.set_start_method('fork', force=True)
    
    # 训练参数 - 保守设置以提高稳定性
    num_workers = 1       
    total_timesteps = 20000   # 适当减少步数
    use_lstm = False
    
    print("="*50)
    print("FISH OBSTACLE AVOIDANCE TRAINING (稳定版+断点续训)")
    print("="*50)
    print(f"Number of workers: {num_workers}")
    print(f"Total timesteps per worker: {total_timesteps}")
    print(f"Using LSTM: {use_lstm}")
    print(f"Task: Navigate fish to target (-5,0) while avoiding obstacle at (-3.5,0)")
    print(f"启动策略: 分阶段启动，每个worker间隔30秒")
    print("="*50)
    
    # 创建共享管理器
    manager = SharedTrainingManager(num_workers)    
    # 启动监控进程
    monitor_process = mp.Process(target=monitor_workers, args=(manager, num_workers))
    monitor_process.start()    
    # 创建训练进程
    processes = []
    for rank in range(num_workers):
        p = mp.Process(
            target=train_with_rank, 
            args=(rank, num_workers, manager, total_timesteps, use_lstm)
        )
        p.start()
        processes.append(p)
        print(f"Worker {rank} 进程已启动")
        
    try:
        # 等待所有进程完成
        for p in processes:
            p.join()        
        # 停止监控进程
        monitor_process.terminate()
        monitor_process.join()        
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating processes...")
        for p in processes:
            p.terminate()
        monitor_process.terminate()        
        for p in processes:
            p.join()
        monitor_process.join()
    
    # 获取最佳模型
    best_model_path = manager.get_best_model_path()
    best_reward = manager.get_best_reward()
    print(f"\nTraining completed. Best model: {best_model_path} with reward: {best_reward:.2f}")
    
    # 将最佳模型复制到标准位置
    if best_model_path and os.path.exists(best_model_path):
        import shutil
        final_best_path = "./saved_models/best_obstacle_avoidance_model.zip"
        try:
            shutil.copy(best_model_path, final_best_path)
            print(f"Best model copied to {final_best_path}")
        except Exception as e:
            print(f"Failed to copy best model: {e}")
    else:
        print("No valid best model found")

if __name__ == "__main__":
    main()

