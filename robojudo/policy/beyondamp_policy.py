import logging
import os

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import BeyondAMPPolicyCfg
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


@policy_registry.register
class BeyondAMPPolicy(Policy):
    """BeyondAMP actor-only ONNX policy for sim2sim."""

    cfg_policy: BeyondAMPPolicyCfg

    def __init__(self, cfg_policy: BeyondAMPPolicyCfg, device: str):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        # ONNXRuntime provider 选择：优先尊重用户设备，其次回退到 CPU。
        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt" and "TensorrtExecutionProvider" in available_providers:
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(cfg_policy.policy_file, providers=providers)

        session_inputs = self.session.get_inputs()
        if len(session_inputs) != 1:
            raise ValueError(
                f"BeyondAMP ONNX expects exactly 1 input, but got {len(session_inputs)}: "
                f"{[i.name for i in session_inputs]}"
            )

        session_outputs = self.session.get_outputs()
        if len(session_outputs) < 1:
            raise ValueError("BeyondAMP ONNX has no output.")

        self.obs_input_name = session_inputs[0].name
        self.output_names = [o.name for o in session_outputs]
        self.action_output_name = "actions" if "actions" in self.output_names else self.output_names[0]

        self.obs_dim = cfg_policy.obs_dim
        onnx_obs_dim = session_inputs[0].shape[-1]
        if isinstance(onnx_obs_dim, int) and onnx_obs_dim != self.obs_dim:
            raise ValueError(f"Obs dim mismatch: config={self.obs_dim}, onnx={onnx_obs_dim}")

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.action_scales = np.asarray(self.cfg_policy.action_scales, dtype=np.float32)
        if self.action_scales.shape[0] != self.num_actions:
            raise ValueError(f"action_scales len mismatch: {self.action_scales.shape[0]} != {self.num_actions}")

        self.reset()
        self._prepare_policy()

    def _prepare_policy(self):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        _ = self.get_action(obs)
        self.last_action[:] = 0.0

    def reset(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def post_step_callback(self, commands: list[str] | None = None):
        pass

    def get_observation(self, env_data, ctrl_data):
        # 注意：RoboJuDo 中 base_lin_vel/base_ang_vel 已经是机体系速度。
        projected_gravity = get_gravity_orientation(env_data.base_quat).astype(np.float32)

        base_lin_vel = env_data.base_lin_vel
        if base_lin_vel is None:
            base_lin_vel = np.zeros(3, dtype=np.float32)
        else:
            base_lin_vel = np.asarray(base_lin_vel, dtype=np.float32)

        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        dof_pos_rel = np.asarray(env_data.dof_pos, dtype=np.float32) - self.default_dof_pos.astype(np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        last_action = self.last_action.astype(np.float32)

        # 96 维严格对齐：proj_g + base_lin_vel + base_ang_vel + (q-q0) + qd + last_action
        obs = np.concatenate(
            [projected_gravity, base_lin_vel, base_ang_vel, dof_pos_rel, dof_vel, last_action], axis=0
        ).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"BeyondAMP observation dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}")

        return obs, {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 1:
            raise ValueError(f"BeyondAMP obs must be 1-D, got shape {obs.shape}")
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"BeyondAMP obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}")

        ort_inputs = {self.obs_input_name: np.expand_dims(obs, axis=0).astype(np.float32)}
        ort_outputs = self.session.run([self.action_output_name], ort_inputs)

        actions = np.asarray(ort_outputs[0], dtype=np.float32).reshape(-1)
        if actions.shape[0] != self.num_actions:
            raise ValueError(f"BeyondAMP action dim mismatch: got {actions.shape[0]}, expected {self.num_actions}")

        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)

        # 保存未缩放动作，下一步观测中的 last_action 需要使用它。
        self.last_action = actions.copy()
        return actions * self.action_scales
