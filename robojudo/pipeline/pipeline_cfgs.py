from typing import Any

from pydantic import model_validator

from robojudo.config import Config
from robojudo.controller import CtrlCfg
from robojudo.environment import EnvCfg
from robojudo.policy import PolicyCfg
from robojudo.tools.debug_log import DebugCfg


class PipelineCfg(Config):
    pipeline_type: str  # name of the pipeline class
    # ===== Pipeline Config =====
    device: str = "cpu"

    debug: DebugCfg = DebugCfg()

    run_fullspeed: bool = False
    """If True, run the pipeline at full speed, ignoring the desired frequency"""

    do_safety_check: bool = False
    """
    If True, perform safety check after each step.
    We recommend enabling this, however if motion is very aggressive, you may disable it.
    """


class RlPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []
    policy: PolicyCfg | Any


class BeyondAMPStartupCfg(Config):
    """BeyondAMP 在 sim2sim 启动阶段使用的两段过渡配置。"""

    enable: bool = True
    """是否启用启动过渡。"""

    use_safe_stage: bool = True
    """是否启用 q_xml -> q_safe 阶段；为 False 时直接 q_xml -> q_ref0。"""

    safe_dof_pos: list[float] | None = None
    """q_safe。若为 None，则回退到 env 配置中的默认关节姿态。"""

    ref_motion_file: str = ""
    """用于提供 q_ref0 的 motion npz 文件路径。"""

    ref_frame_index: int = 0
    """从 motion 中取哪一帧作为 q_ref0。"""

    safe_interp_steps: int = 120
    """q_xml -> q_safe 的插值步数。"""

    ref_interp_steps: int = 160
    """q_safe -> q_ref0 的插值步数。"""

    use_smoothstep: bool = True
    """插值曲线是否使用 smoothstep（更平滑）。"""

    dryrun_policy_each_step: bool = True
    """过渡期是否每步执行一次 policy 前向（输出不直接用于控制，仅做预热）。"""

    reset_to_ref_random: bool = False
    """是否在 reset/reborn 时直接随机采样参考帧写入仿真状态（无启动插值）。"""

    reset_with_root_pose: bool = True
    """随机帧重置时，是否同时写入 root 位姿。"""

    reset_with_root_vel: bool = True
    """随机帧重置时，是否同时写入 root 速度。"""

    root_body_index: int = 0
    """在 body_* 数据中用于 root 的 body 索引（默认 0）。"""

    @model_validator(mode="after")
    def check_values(self):
        if self.safe_interp_steps < 0:
            raise ValueError("safe_interp_steps must be >= 0")
        if self.ref_interp_steps < 0:
            raise ValueError("ref_interp_steps must be >= 0")
        if self.ref_frame_index < 0:
            raise ValueError("ref_frame_index must be >= 0")
        if self.root_body_index < 0:
            raise ValueError("root_body_index must be >= 0")
        return self


class RlBeyondAMPPipelineCfg(RlPipelineCfg):
    pipeline_type: str = "RlBeyondAMPPipeline"
    startup: BeyondAMPStartupCfg = BeyondAMPStartupCfg()


class RlMultiPolicyPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlMultiPolicyPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    policies: list[PolicyCfg | Any] = []
    """First policy as init, rest as extra policies, can be switched to"""


class RlLocoMimicPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlLocoMimicPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    loco_policy: PolicyCfg | Any
    """LocoMotion policy, as init"""
    mimic_policies: list[PolicyCfg | Any] = []
    """MotionMimic policies, can be switched to"""

    # ===== Upper body override Config =====
    upper_dof_num: int = 0
    upper_dof_pos_default: list[float] | None = []
    """Default positions of the upper body DOFs"""
    upper_dof_override_indices: list[int] | None = []
    """Indices of the upper body DOFs to be overridden"""

    @model_validator(mode="after")
    def check_upper_dof(self):
        if self.upper_dof_pos_default is not None:
            if len(self.upper_dof_pos_default) != self.upper_dof_num:
                raise ValueError(
                    f"Length of upper_dof_pos_default ({len(self.upper_dof_pos_default)}) "
                    f"must be equal to upper_dof_num ({self.upper_dof_num})"
                )
        if self.upper_dof_override_indices is not None:
            for idx in self.upper_dof_override_indices:
                if idx < -self.upper_dof_num or idx >= 0:
                    raise ValueError(
                        f"upper_dof_override_indices contains invalid index {idx}, "
                        f"must be in [-{self.upper_dof_num}, 0)"
                    )

        return self
