# === EnvFluent.py (drop-in replacement with curriculum + dual-path waypoints) ===
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import ansys.fluent.core as pyfluent
import csv
import time

class FluentEnv(gym.Env):
    def __init__(
        self,
        max_steps=800,
        reward_function='obstacle_avoidance',
        simu_name="CFD_0",
        preferred_side="auto",            # 'upper' | 'lower' | 'auto'
        waypoint_enable_step=200         # === 课程式开关：达到该步数后启用“中间航点/双路径引导”
    ):
        super().__init__()
        print(f"--- Initializing FluentEnv: {simu_name} ---")

        # === 参数初始化 ===
        self.simu_name = simu_name
        self.max_steps = max_steps
        self.reward_function = reward_function
        self.log_file = f"log_{simu_name}.csv"
        self.device = None
        self.action_summary_file = "action_summary.txt"
        # === 物理/动作离散化参数（可按需调整） ===
        self.period_options = [2]
        self.turning_options = [-0.1, 0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1]
        self.delta_options = [-1, 0, 1]  # [at_delta, tc_delta]
        # === 环境常量 ===
        self.obstacle_position = np.array([-3.5, 0.0])  # 障碍物位置
        self.target_position   = np.array([-5.0, 0.0])  # 目标位置
        self.obstacle_diameter = 1.0                    # 障碍物直径
        self.flow_velocity     = 0.2                    # 来流速度（仅作为观测的一部分）
        # 流域边界范围 (x: 3到-7, y: -3到3)
        self.flow_domain_x_min = -7.0
        self.flow_domain_x_max = 3.0
        self.flow_domain_y_min = -3.0
        self.flow_domain_y_max = 3.0
        # === 路径引导参数 ===
        self.preferred_side = preferred_side            # 'upper' | 'lower' | 'auto'
        self.corridor_margin = 0.30                     # 路径走廊半宽
        self.waypoint_radius = 0.30                     # 航点命中半径
        # === 课程式训练控制 ===
        self.waypoint_enable_step = int(waypoint_enable_step)
        self.waypoint_guidance_active = False           # 会在 step() 中根据 current_step 自动更新
        # === 双路径构建（上/下） ===
        self._build_dual_paths()
        # 进度/状态缓存
        self.prev_target_distance = None
        self._current_wp_index = 0  # 0=绕障点，1=终点（两条路径同层级）

        # === Gym 空间定义 ===
        self.action_space = spaces.MultiDiscrete([3, 3])  # [at_delta_index, tc_delta_index]
        # 状态空间：6维 [x, y, theta, obstacle_distance, obstacle_diameter, flow_velocity]
        self.observation_space = spaces.Box(
            low=np.array([self.flow_domain_x_min, self.flow_domain_y_min, -np.pi, 0.0, 0.0, 0.0]),
            high=np.array([self.flow_domain_x_max, self.flow_domain_y_max,  np.pi, 20.0, 5.0, 1.0]),
            dtype=np.float64
        )

        # === 状态变量 ===
        self.episode_number  = 0
        self.current_step    = 0
        self.simulation_time = 0.0
        self.time_step       = 0.05  # Fluent 仿真时间步长
        # 动作索引/当前动作值
        self.last_period_index   = 0
        self.last_turning_index  = 5  # 初始为0转弯
        self.current_turning_value = self.turning_options[self.last_turning_index]
        self.current_period_value  = self.period_options[self.last_period_index]
        # 动作执行状态（留作扩展）
        self.action_start_time = 0.0
        self.action_duration   = self.current_period_value
        self.is_action_executing = False
        # 鱼的位姿
        self.fish_position   = np.array([0.0, 0.0])
        self.fish_orientation = 0.0
        self.state = np.zeros(6, dtype=np.float64)
        # === 工作目录设置（按需修改/屏蔽） ===
        self.env_dir = os.path.join("fishmove", f"{self.simu_name}")
        os.makedirs(self.env_dir, exist_ok=True)
        os.chdir(self.env_dir)
        # === 控制台日志（transcript）路径 ===
        self.console_log = "fluent_console.log"
        self._transcript_active = False  # <- 新增：进程内标志
        # === 启动 Fluent ===
        self.solver = pyfluent.launch_fluent(
            precision="double",
            processor_count=6,
            dimension=2,
            ui_mode="gui",  # gui | no_gui_or_graphics | no_gui
        )
        self.start_class(complete_reset=True)
        print(f"--- FluentEnv {self.simu_name} initialized ---")
        
    # === 新增：统一的 TUI 执行包装，确保一条命令一行发出去 ===
    def _tui(self, cmd: str):
        # 防止意外换行或多余空白导致的分行错误
        cmd = " ".join(str(cmd).strip().split())
        return self.solver.execute_tui(cmd)

    # === 新增：安全的 transcript 启停 ===
    def _stop_transcript_safe(self):
        try:
            self._tui('/file/stop-transcript')
        except Exception:
            pass
        self._transcript_active = False

    def _start_transcript_safe(self, new_file: str = None, overwrite=True):
        # 路径与文件处理
        log_path = (new_file or self.console_log)
        if overwrite:
            try:
                if os.path.exists(log_path):
                    os.remove(log_path)
            except Exception:
                pass
        # 先尝试停，再启（幂等）
        self._stop_transcript_safe()
        # 用绝对路径更稳妥（Windows 下反斜杠也 OK，但尽量用正斜杠）
        abs_path = os.path.abspath(log_path).replace("\\", "/")
        try:
            self._tui(f'/file/start-transcript "{abs_path}"')
            self._transcript_active = True
            print(f"[{self.simu_name}] Transcript started -> {abs_path}")
        except Exception as e:
            print(f"[{self.simu_name}] Warning: cannot start transcript: {e}")
            self._transcript_active = False
    # -------------------- 路径引导工具 --------------------
    def _build_dual_paths(self):
        """
        构造两条“分段路径”：上侧/下侧
        index=0 为绕障航点；index=1 为终点（相同）
        """
        obs = self.obstacle_position
        y_offset = (self.obstacle_diameter / 2.0) + 0.60
        self.start_position = np.array([0.0, 0.0])
        wp_upper = np.array([obs[0], obs[1] + y_offset])
        wp_lower = np.array([obs[0], obs[1] - y_offset])
        self.path_upper = [wp_upper, self.target_position.copy()]
        self.path_lower = [wp_lower, self.target_position.copy()]
        # 强制单侧：另一侧置 None 减少计算
        if self.preferred_side == 'upper':
            self.path_lower = None
        elif self.preferred_side == 'lower':
            self.path_upper = None
        # 'auto' 则两条都保留

    @staticmethod
    def _segment_progress_and_lateral(p, a, b):
        """
        点 p 到线段 a->b：
        返回 (t, lateral_dist, proj_point)
        - t ∈ [0,1] 为投影在段上的归一化进度
        - lateral_dist 为垂直偏移距离
        """
        v = b - a
        vv = np.dot(v, v)
        if vv <= 1e-12:
            return 0.0, np.linalg.norm(p - a), a.copy()
        t = np.clip(np.dot(p - a, v) / vv, 0.0, 1.0)
        proj = a + t * v
        lateral = np.linalg.norm(p - proj)
        return t, lateral, proj

    def _current_path_segment(self, use_upper: bool):
        """
        返回“上/下路径”的当前段 (a, b)
        """
        path = self.path_upper if use_upper else self.path_lower
        # 仅在存在该侧路径时调用本函数
        if self._current_wp_index == 0:
            a = self.start_position
        else:
            a = path[self._current_wp_index - 1]
        b = path[min(self._current_wp_index, 1)]
        return a, b

    # -------------------- 基础流程 --------------------
    def start_class(self, complete_reset=True):
        self.episode_number = 0
        self.current_step = 0
        self.initialize_flow(complete_reset)

    def initialize_flow(self, complete_reset=True):
        if complete_reset:
            print(f"[{self.simu_name}] Resetting Fluent case...")
            self.solver.file.read_case_data(file_name="fishmove1.cas.h5")
            self.solver.execute_tui('/define/user-defined/execute-on-demand "load_fish_state::libudf"')
            # 重置状态
            self.simulation_time = 0.0
            self.last_turning_index  = 5  # 初始为0转弯
            self.last_period_index   = 0
            self.current_turning_value = self.turning_options[self.last_turning_index]
            self.current_period_value  = self.period_options[self.last_period_index]
            self.action_start_time = 0.0
            self.action_duration   = self.current_period_value
            self.is_action_executing = False
            for filename in ["variable_record.txt", self.action_summary_file, "Output.txt", self.console_log, "vc-rfile.out"]:
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except Exception:
                    pass
            # 关键：改成安全启用
            self._start_transcript_safe(self.console_log, overwrite=True)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.initialize_flow(complete_reset=True)
        self.current_step    = 0
        self.episode_number += 1
        self.simulation_time = 0.0
        self.last_turning_index  = 5  # 初始为0转弯
        self.last_period_index   = 0
        # 位置和朝向
        self.fish_position    = np.array([0.0, 0.0])
        self.fish_orientation = 0.0
        self.state = np.zeros(6, dtype=np.float64)
        # 路径/进度重置
        self._current_wp_index = 0
        self.prev_target_distance = np.linalg.norm(self.fish_position - self.target_position)
        # 重新生成双路径（以防参数修改）
        self._build_dual_paths()
        # 课程开关重置
        self.waypoint_guidance_active = False
        print(f"After reset, fish_position={self.fish_position}, simulation_time={self.simulation_time}")
        for filename in ["variable_record.txt"]:
            if os.path.exists(filename):
                os.remove(filename)
        return self.state, {}

    # -------------------- 观测读取 --------------------
    def _get_obs(self):
        x_disp, y_disp, theta_disp, _, _, _ = self._read_output_file()
        obstacle_distance = self._calculate_obstacle_distance()
        state = np.array([
            x_disp, y_disp, theta_disp,
            obstacle_distance, self.obstacle_diameter, self.flow_velocity
        ], dtype=np.float64)
        return state

    def _read_output_file(self):
        x_disp = y_disp = theta_disp = force_x = force_y = moment_z = 0.0
        output_file = "Output.txt"
        try:
            data = np.loadtxt(output_file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[0] > 0:
                last_row = data[-1]
                x_disp     = last_row[1]
                y_disp     = last_row[2]
                theta_disp = last_row[3]
                force_x    = last_row[4]
                force_y    = last_row[5]
                moment_z   = last_row[6]
        except Exception as e:
            print(f"[{self.simu_name}] Error reading Output.txt: {e}")
        return x_disp, y_disp, theta_disp, force_x, force_y, moment_z

    def _negative_mesh_detected_from_console(self):
        if not hasattr(self, "console_log") or not os.path.exists(self.console_log):
            return False
        try:
            with open(self.console_log, "rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                read_size = 64 * 1024
                f.seek(max(file_size - read_size, 0))
                tail_bytes = f.read()
            tail_text = tail_bytes.decode(errors="ignore")
            key_err = "Error at host: Update-Dynamic-Mesh failed. Negative cell volume detected."
            if key_err in tail_text:
                return True
            return ("Update-Dynamic-Mesh failed" in tail_text and
                    "Negative cell volume detected" in tail_text)
        except Exception:
            return False

    def _calculate_obstacle_distance(self):
        return np.linalg.norm(self.fish_position - self.obstacle_position)

    # -------------------- 交互步进 --------------------
    def step(self, action):
        # 1) 解码离散动作为索引增量
        at_delta = self.delta_options[action[0]]
        tc_delta = self.delta_options[action[1]]
        # 2) 更新离散索引并读取动作值
        new_turning_index = int(np.clip(self.last_turning_index + at_delta, 0, len(self.turning_options) - 1))
        new_period_index  = int(np.clip(self.last_period_index  + tc_delta, 0, len(self.period_options) - 1))
        self.current_turning_value = self.turning_options[new_turning_index]
        self.current_period_value  = self.period_options[new_period_index]
        steps_to_execute = int(self.current_period_value / self.time_step)
        # 3) 标志位
        failed = False
        success = False
        failure_reason = ""
        terminated = False
        truncated  = False
        step_i = -1
        # 4) 求解器参数同步
        self.solver.execute_tui(f"/solve/set/time-step {self.time_step}")
        self.solver.execute_tui(f"(rpsetvar 'tc {self.current_period_value})")
        self.solver.execute_tui(f"(rpsetvar 'at {self.current_turning_value})")
        # 5) 在 Fluent 中推进
        for step_i in range(steps_to_execute):
            if step_i == 0:
                self.solver.execute_tui('/define/user-defined/execute-on-demand "add_action_from_console::libudf"')

            if self.current_step >= self.max_steps:
                truncated = True
                break
            try:
                self.simulation_time += self.time_step
                self.current_step += 1
                self.solver.execute_tui("/solve/dual-time-iterate 1 10")
                # 负网格
                if self._negative_mesh_detected_from_console():
                    failed = True
                    failure_reason = "negative_mesh_detected"
                    print("negative_mesh_detected (from console)")
                    terminated = True
                    break
                # 更新位姿
                xdisp, ydisp, thetadisp, _, _, _ = self._read_output_file()
                self.fish_position[0] = xdisp
                self.fish_position[1] = ydisp
                self.fish_orientation = thetadisp
                # 碰撞/越界
                obstacle_distance = self._calculate_obstacle_distance()
                if obstacle_distance < (self.obstacle_diameter / 2 + 0.02):
                    failed = True
                    failure_reason = "collision_with_obstacle"
                    terminated = True
                    break
                if (self.fish_position[0] > self.flow_domain_x_max or
                    self.fish_position[0] < self.flow_domain_x_min or
                    self.fish_position[1] > self.flow_domain_y_max or
                    self.fish_position[1] < self.flow_domain_y_min):
                    failed = True
                    failure_reason = "out_of_flow_domain"
                    terminated = True
                    break

            except Exception as e:
                print(f"[{self.simu_name}] Error in simulation step {step_i}: {e}")
                failed = True
                failure_reason = "fluent_exception"
                terminated = True
                break
        # 6) 更新索引记录
        self.last_turning_index = new_turning_index
        self.last_period_index  = new_period_index
        # 7) 观测与是否成功
        self.state = self._get_obs()
        target_distance   = np.linalg.norm(self.fish_position - self.target_position)
        obstacle_distance = self._calculate_obstacle_distance()
        success = (target_distance < 0.2)
        if success:
            terminated = True
        # 8) 奖励计算（课程式航点引导）
        if self.prev_target_distance is None:
            self.prev_target_distance = target_distance
        progress = self.prev_target_distance - target_distance      # >0 表示前进
        progress_reward = 50.0 * progress
        self.prev_target_distance = target_distance

        # 是否启用航点/双路径引导
        self.waypoint_guidance_active = (self.current_step >= self.waypoint_enable_step)
        path_reward = 0.0
        waypoint_bonus = 0.0
        info_active_side = None
        if self.waypoint_guidance_active:
            # 启用后：两侧候选，选“当前目标航点更近”的那一侧用于奖励计算
            candidates = []
            if self.path_upper is not None:
                candidates.append(("upper", self.path_upper))
            if self.path_lower is not None:
                candidates.append(("lower", self.path_lower))

            active_choice = None
            best_wp_dist = np.inf
            for name, path in candidates:
                next_wp = path[self._current_wp_index]
                dist = np.linalg.norm(self.fish_position - next_wp)
                if dist < best_wp_dist:
                    best_wp_dist = dist
                    active_choice = (name, path, next_wp)

            if active_choice is not None:
                name, path, next_wp = active_choice
                info_active_side = name

                # 命中任意一侧的当前航点即可推进段索引
                hit_any = False
                for _name, _path in candidates:
                    _wp = _path[self._current_wp_index]
                    if np.linalg.norm(self.fish_position - _wp) <= self.waypoint_radius:
                        hit_any = True
                        break
                if hit_any:
                    waypoint_bonus += 200.0
                    self._current_wp_index = min(self._current_wp_index + 1, 1)  # 两段路径：0->1

                # 段内“侧向偏离 + 段进度”
                use_upper = (name == "upper")
                a, b = self._current_path_segment(use_upper=use_upper)
                t_seg, lateral, _ = self._segment_progress_and_lateral(self.fish_position, a, b)

                excess = max(0.0, lateral - self.corridor_margin)
                deviation_penalty = -3.0 * (lateral + 2.0 * excess)
                segment_progress_reward = 5.0 * (t_seg)
                path_reward = deviation_penalty + segment_progress_reward
        success_reward    = 1000 if success else 0
        distance_reward   = -target_distance * 10.0
        reward = (
            progress_reward + distance_reward + success_reward
        )

        if failed:
            if failure_reason == "collision_with_obstacle":
                reward -= 500
            elif failure_reason == "negative_mesh_detected":
                reward -= 800
            elif failure_reason == "out_of_flow_domain":
                reward -= 400
            elif failure_reason == "fluent_exception":
                reward -= 1000

        info = {
            "simulation_time": self.simulation_time,
            "turning_action": self.current_turning_value,
            "period_action": self.current_period_value,
            "fish_position": self.fish_position.copy(),
            "fish_orientation": self.fish_orientation,
            "obstacle_distance": obstacle_distance,
            "target_distance": target_distance,
            "success": success,
            "failed": failed,
            "failure_reason": failure_reason,
            "timeout": truncated,
            "turning_index": new_turning_index,
            "period_index": new_period_index,
            "steps_executed": (step_i + 1) if step_i >= 0 else 0,
            "at_delta": at_delta,
            "tc_delta": tc_delta,
            "action": action.copy() if hasattr(action, 'copy') else list(action),
            "active_path_side": info_active_side,                 # None 表示未启用航点引导
            "wp_index": self._current_wp_index,
            "waypoint_guidance_active": self.waypoint_guidance_active,
            "waypoint_enable_step": self.waypoint_enable_step,
        }

        self._log_variables()
        self._log_action_summary(self.state, self.current_turning_value, self.current_period_value)

        return self.state.copy(), reward, terminated, truncated, info

    # -------------------- 关闭 --------------------
    def close(self):
        if hasattr(self, 'solver') and self.solver is not None:
            try:
                self._stop_transcript_safe()
                self.solver.exit()
            except Exception as e:
                print(f"[{self.simu_name}] Error closing Fluent: {e}")
    # -------------------- 日志 --------------------
    def _log_variables(self):
        filename = "variable_record.txt"
        mode = "a" if os.path.exists(filename) else "w"
        with open(filename, mode, encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',', lineterminator="\n")
            if mode == "w":
                header = ["simulation_time", "x_disp", "y_disp", "theta_disp",
                          "turning_action", "period_action", "obstacle_distance", "target_distance"]
                writer.writerow(header)
            writer.writerow([
                self.simulation_time,
                self.fish_position[0],
                self.fish_position[1],
                self.fish_orientation,
                self.current_turning_value,
                self.current_period_value,
                self._calculate_obstacle_distance(),
                np.linalg.norm(self.fish_position - self.target_position)
            ])

    def _log_action_summary(self, state, at_value, tc_value):
        x_disp, y_disp, theta_disp = state[0], state[1], state[2]
        obstacle_distance, obstacle_diameter, flow_velocity = state[3], state[4], state[5]
        target_distance = np.linalg.norm(self.fish_position - self.target_position)

        line = [
            f"time={self.simulation_time:.3f}",
            f"at={at_value}", f"tc={tc_value}",
            f"x={x_disp:.4f}", f"y={y_disp:.4f}", f"theta={theta_disp:.4f}",
            f"obs_dist={obstacle_distance:.4f}",
            f"obs_diam={obstacle_diameter:.4f}",
            f"flow_vel={flow_velocity:.4f}",
            f"target_dist={target_distance:.4f}",
            f"turn_idx={self.last_turning_index}",
            f"period_idx={self.last_period_index}",
            f"step={getattr(self, 'current_step', 0)}",
        ]
        with open(self.action_summary_file, "a", encoding="utf-8") as f:
            f.write(" ".join(line) + "\n")
