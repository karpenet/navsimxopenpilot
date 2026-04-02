"""
Mirror of NAVSIM install tree: copy into ``navsim/navsim/agents/openpilot/`` for imports.

NAVSIM ``AbstractAgent`` that can run openpilot vision+policy via ONNX (``use_stub_trajectory=False``)
or a constant-velocity straight line (default).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from openpilot.selfdrive.modeld.constants import ModelConstants

if TYPE_CHECKING:
    from navsim.agents.openpilot.onnx_inference import OpenpilotOnnxRunner


def _default_models_dir() -> Path:
    """Resolve openpilot ``modeld/models`` (sibling repo layout or env)."""
    import os
    env = os.environ.get("OPENPILOT_MODELD_MODELS")
    if env:
        return Path(env)
    # .../navsim/navsim/agents/openpilot/ -> repo root is parents[3]
    here = Path(__file__).resolve()
    root = here.parents[3]
    sibling = root.parent / "openpilot" / "openpilot" / "selfdrive" / "modeld" / "models"
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError(
        "Set openpilot_models_dir in Hydra or OPENPILOT_MODELD_MODELS; could not find sibling openpilot checkout."
    )


def _rgb_from_cam_f0(cam_f0) -> np.ndarray:
    """Return uint8 (H, W, 3) from NAVSIM ``Camera``."""
    im = cam_f0.image
    if im is None:
        raise ValueError("cam_f0.image is None; enable cam_f0 in SensorConfig")
    rgb = np.asarray(im)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


class OpenpilotNavsimAgent(AbstractAgent):
    """NAVSIM agent: ONNX openpilot stack or stub trajectory."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        openpilot_models_dir: Optional[str] = None,
        traffic_right_hand: bool = True,
        use_stub_trajectory: bool = True,
    ) -> None:
        super().__init__(trajectory_sampling, requires_scene=False)
        self._openpilot_models_dir: Optional[Path] = Path(openpilot_models_dir) if openpilot_models_dir else None
        self._traffic_right_hand = traffic_right_hand
        self._use_stub_trajectory = use_stub_trajectory
        self._initialized = False
        self._runner: OpenpilotOnnxRunner | None = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._use_stub_trajectory:
            self._runner = None
        else:
            from navsim.agents.openpilot.onnx_inference import OpenpilotOnnxRunner

            md = self._openpilot_models_dir or _default_models_dir()
            self._runner = OpenpilotOnnxRunner(md)
        self._initialized = True

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if not self._initialized:
            self.initialize()

        if self._use_stub_trajectory:
            return self._stub_constant_velocity_trajectory(agent_input, self._trajectory_sampling)

        assert self._runner is not None
        from navsim.agents.openpilot.onnx_inference import plan_to_poses_xy_heading

        self._runner.reset()
        desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        cmd = agent_input.ego_statuses[-1].driving_command
        if cmd is not None and len(cmd) > 0:
            mi = int(np.argmax(np.asarray(cmd)))
            if 0 <= mi < ModelConstants.DESIRE_LEN:
                desire[mi] = 1.0

        tc = np.zeros(2, dtype=np.float32)
        tc[1 if self._traffic_right_hand else 0] = 1.0

        last_out = None
        for cameras in agent_input.cameras:
            rgb = _rgb_from_cam_f0(cameras.cam_f0)
            last_out = self._runner.run_step(rgb, rgb, desire, tc)

        if last_out is None:
            return self._stub_constant_velocity_trajectory(agent_input, self._trajectory_sampling)

        poses = plan_to_poses_xy_heading(
            last_out["plan"],
            self._trajectory_sampling.num_poses,
            self._trajectory_sampling.interval_length,
        )
        return Trajectory(poses, self._trajectory_sampling)

    @staticmethod
    def _stub_constant_velocity_trajectory(agent_input: AgentInput, trajectory_sampling: TrajectorySampling) -> Trajectory:
        ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
        ego_speed = float((ego_velocity_2d**2).sum(-1) ** 0.5)
        num_poses, dt = trajectory_sampling.num_poses, trajectory_sampling.interval_length
        poses = np.array(
            [[(time_idx + 1) * dt * ego_speed, 0.0, 0.0] for time_idx in range(num_poses)],
            dtype=np.float32,
        )
        return Trajectory(poses, trajectory_sampling)
