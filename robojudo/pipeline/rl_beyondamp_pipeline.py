import logging
from enum import Enum, auto
from pathlib import Path

import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlBeyondAMPPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger(__name__)


@pipeline_registry.register
class RlBeyondAMPPipeline(RlPipeline):
    """BeyondAMP 专用 pipeline：
    1) 可选 q_xml -> q_safe 插值；
    2) q_safe(或 q_xml) -> q_ref0(参考动作首帧) 插值；
    3) 插值结束后切换到 actor 正常接管。
    """

    cfg: RlBeyondAMPPipelineCfg

    class StartupStage(Enum):
        XML_TO_SAFE = auto()
        SAFE_TO_REF = auto()
        DONE = auto()

    def __init__(self, cfg: RlBeyondAMPPipelineCfg):
        # super().__init__ 过程中会执行 self_check/reset，此时本类启动状态尚未加载，
        # 先关掉启动状态机，待 super 完成后再初始化。
        self._startup_runtime_ready = False
        super().__init__(cfg=cfg)

        self._load_startup_targets()
        self._startup_runtime_ready = True
        if self.cfg.startup.reset_to_ref_random:
            self._reset_env_to_random_motion_frame()
            self.policy.reset()
        self._reset_startup_state()

    def reset(self):
        super().reset()
        if self._startup_runtime_ready:
            if self.cfg.startup.reset_to_ref_random:
                self._reset_env_to_random_motion_frame()
                # 环境状态被强制写入后，清理策略内部状态，避免沿用旧 last_action。
                self.policy.reset()
            self._reset_startup_state()

    def _load_startup_targets(self):
        startup_cfg = self.cfg.startup

        # q_safe: 默认取 env config 的默认姿态（非 AMP q0），也支持显式覆写。
        if startup_cfg.safe_dof_pos is not None:
            q_safe = np.asarray(startup_cfg.safe_dof_pos, dtype=np.float32)
        else:
            q_safe = np.asarray(self.cfg.env.dof.default_pos, dtype=np.float32)

        # q_ref0: 来自参考动作 npz 的指定帧（默认第一帧）。
        motion_file = Path(startup_cfg.ref_motion_file).expanduser()
        if not motion_file.is_file():
            raise FileNotFoundError(
                f"BeyondAMP startup motion file not found: {motion_file}. "
                "Please set `startup.ref_motion_file` to a valid npz path."
            )
        with np.load(motion_file) as data:
            if "joint_pos" not in data:
                raise ValueError(f"`joint_pos` not found in motion file: {motion_file}")
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32) if "joint_vel" in data else None
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32) if "body_pos_w" in data else None
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32) if "body_quat_w" in data else None
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32) if "body_lin_vel_w" in data else None
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32) if "body_ang_vel_w" in data else None

        if joint_pos.ndim != 2:
            raise ValueError(f"`joint_pos` should be 2-D [T, D], got shape {joint_pos.shape}")
        if joint_pos.shape[0] == 0:
            raise ValueError(f"`joint_pos` is empty in motion file: {motion_file}")

        frame_idx = min(startup_cfg.ref_frame_index, joint_pos.shape[0] - 1)
        q_ref0 = joint_pos[frame_idx]

        if q_safe.shape[0] != self.env.num_dofs:
            raise ValueError(
                f"q_safe dof mismatch: expected {self.env.num_dofs}, got {q_safe.shape[0]}"
            )
        if q_ref0.shape[0] != self.env.num_dofs:
            raise ValueError(
                f"q_ref0 dof mismatch: expected {self.env.num_dofs}, got {q_ref0.shape[0]}"
            )

        if joint_vel is not None and joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"`joint_vel` shape mismatch: expect {joint_pos.shape}, got {joint_vel.shape}"
            )

        self._motion_joint_pos = joint_pos
        self._motion_joint_vel = joint_vel
        self._motion_body_pos_w = body_pos_w
        self._motion_body_quat_w = body_quat_w
        self._motion_body_lin_vel_w = body_lin_vel_w
        self._motion_body_ang_vel_w = body_ang_vel_w

        self._q_safe = q_safe
        self._q_ref0 = q_ref0

        logger.info(
            "[BeyondAMP startup] Loaded targets: "
            f"q_safe_dim={self._q_safe.shape[0]}, q_ref0_dim={self._q_ref0.shape[0]}, "
            f"motion_file={motion_file}, frame={frame_idx}"
        )

    def _reset_startup_state(self):
        self._startup_enabled = bool(
            self.cfg.env.is_sim and self.cfg.startup.enable and (not self.cfg.startup.reset_to_ref_random)
        )
        if not self._startup_enabled:
            self._startup_stage = self.StartupStage.DONE
        else:
            if self.cfg.startup.use_safe_stage:
                self._startup_stage = self.StartupStage.XML_TO_SAFE
            else:
                self._startup_stage = self.StartupStage.SAFE_TO_REF
        self._startup_stage_step = 0

        # 记录当前状态作为插值起点（q_xml）。
        self.env.update()
        self._startup_stage_start = np.asarray(self.env.dof_pos, dtype=np.float32)

        if self._startup_enabled:
            if self._startup_stage == self.StartupStage.XML_TO_SAFE:
                logger.info(
                    "[BeyondAMP startup] Stage-1 begin: q_xml -> q_safe "
                    f"(steps={self.cfg.startup.safe_interp_steps})"
                )
            else:
                logger.info(
                    "[BeyondAMP startup] Single-stage begin: q_xml -> q_ref0 "
                    f"(steps={self.cfg.startup.ref_interp_steps})"
                )
        elif self.cfg.startup.reset_to_ref_random:
            logger.info("[BeyondAMP startup] reset_to_ref_random enabled, skip startup interpolation.")

    def _reset_env_to_random_motion_frame(self):
        if not self.cfg.startup.reset_to_ref_random:
            return

        if not hasattr(self.env, "data") or not hasattr(self.env, "model"):
            logger.warning("reset_to_ref_random only supports MujocoEnv currently.")
            return

        if self._motion_joint_pos.shape[0] <= 0:
            logger.warning("Motion data is empty, skip reset_to_ref_random.")
            return

        # 采样随机参考帧。
        frame_idx = int(np.random.randint(0, self._motion_joint_pos.shape[0]))
        q = self._motion_joint_pos[frame_idx].copy()
        qd = (
            self._motion_joint_vel[frame_idx].copy()
            if self._motion_joint_vel is not None
            else np.zeros_like(q, dtype=np.float32)
        )

        if q.shape[0] != self.env.num_dofs:
            logger.error(
                f"Random ref reset aborted: dof mismatch, expected {self.env.num_dofs}, got {q.shape[0]}"
            )
            return

        # 直接写入 MuJoCo 状态。
        import mujoco

        mujoco.mj_resetDataKeyframe(self.env.model, self.env.data, 0)

        if (
            self.cfg.startup.reset_with_root_pose
            and self._motion_body_pos_w is not None
            and self._motion_body_quat_w is not None
        ):
            body_idx = int(
                min(self.cfg.startup.root_body_index, self._motion_body_pos_w.shape[1] - 1)
            )
            root_pos = self._motion_body_pos_w[frame_idx, body_idx].copy()
            root_quat_wxyz = self._motion_body_quat_w[frame_idx, body_idx].copy()
            quat_norm = float(np.linalg.norm(root_quat_wxyz))
            if quat_norm > 1e-6:
                root_quat_wxyz = root_quat_wxyz / quat_norm
            else:
                root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.env.data.qpos[0:3] = root_pos
            self.env.data.qpos[3:7] = root_quat_wxyz

        self.env.data.qpos[-self.env.num_dofs :] = q

        self.env.data.qvel[:] = 0.0
        self.env.data.qvel[-self.env.num_dofs :] = qd
        if (
            self.cfg.startup.reset_with_root_vel
            and self._motion_body_lin_vel_w is not None
            and self._motion_body_ang_vel_w is not None
        ):
            body_idx = int(
                min(self.cfg.startup.root_body_index, self._motion_body_lin_vel_w.shape[1] - 1)
            )
            self.env.data.qvel[0:3] = self._motion_body_lin_vel_w[frame_idx, body_idx]
            self.env.data.qvel[3:6] = self._motion_body_ang_vel_w[frame_idx, body_idx]

        self.env.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.update()

        logger.info(
            "[BeyondAMP startup] Random ref reset applied: "
            f"frame={frame_idx}, root_body_index={self.cfg.startup.root_body_index}"
        )

    def _stage_total_steps(self) -> int:
        if self._startup_stage == self.StartupStage.XML_TO_SAFE:
            return int(self.cfg.startup.safe_interp_steps)
        if self._startup_stage == self.StartupStage.SAFE_TO_REF:
            return int(self.cfg.startup.ref_interp_steps)
        return 0

    def _stage_target(self) -> np.ndarray:
        if self._startup_stage == self.StartupStage.XML_TO_SAFE:
            return self._q_safe
        if self._startup_stage == self.StartupStage.SAFE_TO_REF:
            return self._q_ref0
        return self._q_ref0

    def _interp_alpha(self, step_idx: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 1.0

        # step_idx 从 0 开始，这里 +1 让最后一步正好达到目标姿态。
        ratio = min((step_idx + 1) / float(total_steps), 1.0)
        if self.cfg.startup.use_smoothstep:
            # smoothstep: 3r^2 - 2r^3，端点速度为 0，切换更柔和。
            return float(ratio * ratio * (3.0 - 2.0 * ratio))
        return float(ratio)

    def _compute_startup_pd_target(self) -> np.ndarray:
        target = self._stage_target()
        total_steps = self._stage_total_steps()
        alpha = self._interp_alpha(self._startup_stage_step, total_steps)
        return (1.0 - alpha) * self._startup_stage_start + alpha * target

    def _advance_startup_stage_if_needed(self):
        total_steps = self._stage_total_steps()
        if total_steps > 0 and self._startup_stage_step < total_steps:
            return

        # 阶段结束后，以下一阶段开始时的真实机器人姿态作为新的插值起点。
        self.env.update()
        self._startup_stage_start = np.asarray(self.env.dof_pos, dtype=np.float32)
        self._startup_stage_step = 0

        if self._startup_stage == self.StartupStage.XML_TO_SAFE:
            self._startup_stage = self.StartupStage.SAFE_TO_REF
            logger.info(
                "[BeyondAMP startup] Stage-2 begin: q_safe -> q_ref0 "
                f"(steps={self.cfg.startup.ref_interp_steps})"
            )
            return

        if self._startup_stage == self.StartupStage.SAFE_TO_REF:
            self._startup_stage = self.StartupStage.DONE
            self._startup_enabled = False

            # 交接给 actor 前清零 last_action，避免把过渡期历史动作带入 AMP 观测。
            inner_policy = self.policy.policy
            if hasattr(inner_policy, "last_action"):
                inner_policy.last_action = np.zeros_like(inner_policy.last_action)

            logger.info("[BeyondAMP startup] Transition done. Actor takeover.")

    def _startup_step(self, dry_run: bool = False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        # 可选：每步执行一次 policy 前向，输出不直接控制机器人，只用于“热启动”。
        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        extras = extras if isinstance(extras, dict) else {}
        if self.cfg.startup.dryrun_policy_each_step:
            inner_policy = self.policy.policy
            old_last_action = None
            if hasattr(inner_policy, "last_action"):
                old_last_action = np.asarray(inner_policy.last_action).copy()
            _ = self.policy.get_action(obs)
            if old_last_action is not None:
                inner_policy.last_action = old_last_action

        pd_target = self._compute_startup_pd_target()

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

            self._startup_stage_step += 1
            self._advance_startup_stage_if_needed()

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        super().post_step_callback(env_data, ctrl_data, extras, pd_target)
        commands = ctrl_data.get("COMMANDS", [])
        if self.cfg.startup.reset_to_ref_random and ("[SIM_REBORN]" in commands):
            # super 已执行过 env.reborn，这里二次覆盖为随机参考帧。
            self._reset_env_to_random_motion_frame()
            self.policy.reset()
            self._reset_startup_state()

    def step(self, dry_run: bool = False):
        if (not self._startup_runtime_ready) or dry_run:
            super().step(dry_run=dry_run)
            return

        if self._startup_enabled and self._startup_stage != self.StartupStage.DONE:
            self._startup_step(dry_run=False)
            return

        super().step(dry_run=False)
