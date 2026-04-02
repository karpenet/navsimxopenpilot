"""
Build comma/openpilot-style tensors from a NAVSIM ``Scene`` + ``AgentInput``.

Image packing matches ``navsim.agents.openpilot.onnx_inference`` (RGB → YUV layout
``(1, 12, 128, 256)`` uint8). Policy-side arrays are sized from ONNX/metadata
``input_shapes`` and filled with NAVSIM-derived values where applicable; the rest
are zeros (reasonable for a single-shot sim step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from navsim.common.dataclasses import AgentInput, Camera, Scene


def _policy_logical_key(onnx_name: str) -> Optional[str]:
    """Align with ``OpenpilotAgent._guess_logical_key`` for policy inputs."""
    n = onnx_name.lower()
    if "feature" in n and "buffer" in n:
        return "feature_buffer"
    if "traffic" in n:
        return "traffic_convention"
    if "prev" in n and "curv" in n:
        return "prev_desired_curvature"
    if "lateral" in n or ("steer" in n and "delay" in n) or n.endswith("_delay"):
        return "lateral_control_params"
    if "desire" in n:
        return "desire"
    if "state" in n or "hidden" in n or "gru" in n or "memory" in n or "initial" in n:
        return None
    return None


def pack_rgb_to_6ch_uint8(rgb_hwc: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> uint8 (6, 128, 256) model YUV layout."""
    im = Image.fromarray(np.ascontiguousarray(rgb_hwc))
    im = im.resize((512, 256), Image.Resampling.BILINEAR)
    px = np.asarray(im, dtype=np.float32)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255)
    u = (-0.14713 * r - 0.28886 * g + 0.436 * b + 128).clip(0, 255)
    v = (0.615 * r - 0.51499 * g - 0.10001 * b + 128).clip(0, 255)
    y, u, v = y.astype(np.uint8), u.astype(np.uint8), v.astype(np.uint8)
    h, w = y.shape
    y00, y01 = y[0::2, 0::2], y[0::2, 1::2]
    y10, y11 = y[1::2, 0::2], y[1::2, 1::2]
    u_down = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
    v_down = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
    return np.stack([y00, y01, y10, y11, u_down, v_down], axis=0)


def stack_two_frames_12ch(prev_6: Optional[np.ndarray], cur_6: np.ndarray) -> np.ndarray:
    """(6, 128, 256) * 2 -> (1, 12, 128, 256) uint8."""
    if prev_6 is None:
        prev_6 = cur_6
    stacked = np.concatenate([prev_6, cur_6], axis=0)
    return stacked.reshape(1, 12, 128, 256)


def _rgb_from_camera(cam: Camera, name: str) -> np.ndarray:
    if cam.image is None:
        raise ValueError(f"{name}.image is None; enable that camera in SensorConfig")
    rgb = np.asarray(cam.image)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb


def _build_image_streams(
    agent_input: AgentInput,
    road_camera: str,
    wide_camera: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cams = agent_input.cameras
    if len(cams) < 1:
        raise ValueError("AgentInput.cameras is empty")

    def pack_at(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        entry = cams[idx]
        road = _rgb_from_camera(getattr(entry, road_camera), road_camera)
        wide = _rgb_from_camera(getattr(entry, wide_camera), wide_camera)
        return pack_rgb_to_6ch_uint8(road), pack_rgb_to_6ch_uint8(wide)

    cur_r, cur_w = pack_at(-1)
    if len(cams) >= 2:
        prev_r, prev_w = pack_at(-2)
    else:
        prev_r, prev_w = cur_r, cur_w

    return stack_two_frames_12ch(prev_r, cur_r), stack_two_frames_12ch(prev_w, cur_w)


def _fill_policy_logical_tensors(
    agent_input: AgentInput,
    steering_delay_s: float,
    policy_input_shapes: Dict[str, Tuple[int, ...]],
    skip_names: Set[str],
) -> Dict[str, np.ndarray]:
    """One numpy array per *logical* key for policy inputs not fed from vision/recurrent state."""
    logical: Dict[str, np.ndarray] = {}
    for onnx_name, shape in policy_input_shapes.items():
        if onnx_name in skip_names:
            continue
        key = _policy_logical_key(onnx_name)
        if key is None:
            continue
        arr = np.zeros(shape, dtype=np.float32)
        lname = onnx_name.lower()
        if "traffic" in lname:
            flat = arr.reshape(-1)
            if flat.size >= 2:
                flat[0] = 0.0
                flat[1] = 1.0
            arr = flat.reshape(shape)
        elif "desire" in lname:
            flat = arr.reshape(-1)
            cmd = agent_input.ego_statuses[-1].driving_command
            if flat.size >= 8:
                flat[-8:] = 0.0
                if cmd is not None and len(cmd) > 0:
                    idx = int(np.argmax(np.asarray(cmd)))
                    if 0 <= idx < 8:
                        flat[-8 + idx] = 1.0
            arr = flat.reshape(shape)
        elif "lateral" in lname or ("steer" in lname and "delay" in lname) or lname.endswith("_delay"):
            flat = arr.reshape(-1)
            v = float(np.linalg.norm(agent_input.ego_statuses[-1].ego_velocity))
            if flat.size >= 1:
                flat[0] = v
            if flat.size >= 2:
                flat[1] = float(steering_delay_s)
            arr = flat.reshape(shape)
        elif "prev" in lname and "curv" in lname:
            pass
        logical[key] = arr
    return logical


@dataclass
class OpenpilotModelInputs:
    """Tensors for one ``OpenpilotAgent.compute_trajectory`` step."""

    image_stream: np.ndarray
    wide_image_stream: np.ndarray
    _policy_logical: Dict[str, np.ndarray]

    def as_dict(self, flatten_images: bool = False) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {
            "image_stream": self.image_stream,
            "wide_image_stream": self.wide_image_stream,
        }
        out.update(self._policy_logical)
        if flatten_images:
            for k in ("image_stream", "wide_image_stream"):
                a = np.asarray(out[k])
                if a.dtype == np.uint8 or np.issubdtype(a.dtype, np.unsignedinteger):
                    out[k] = (a.astype(np.float32).reshape(-1) / 255.0).astype(np.float32)
                else:
                    out[k] = np.asarray(a, dtype=np.float32).reshape(-1)
        return out


def build_openpilot_inputs_from_scene(
    scene: Scene,
    agent_input: AgentInput,
    road_camera: str = "cam_f0",
    wide_camera: str = "cam_f0",
    steering_delay_s: float = 0.0,
    policy_input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    policy_skip_input_names: Optional[Set[str]] = None,
) -> OpenpilotModelInputs:
    """
    :param scene: Currently unused; reserved for map-based features.
    :param policy_input_shapes: ONNX input name -> concrete shape (e.g. from metadata or static dims).
    :param policy_skip_input_names: Policy inputs filled from vision outputs or recurrent state.
        Other policy fields (e.g. ``prev_desired_curvature``) are filled with **zeros** when not skipped;
        lateral/desire/traffic are approximated from ``AgentInput`` (see ``_fill_policy_logical_tensors``).
    """
    del scene
    img_main, img_wide = _build_image_streams(agent_input, road_camera, wide_camera)
    skip = policy_skip_input_names or set()
    if policy_input_shapes:
        policy_logical = _fill_policy_logical_tensors(
            agent_input, steering_delay_s, policy_input_shapes, skip
        )
    else:
        policy_logical = {
            "desire": np.zeros(8, dtype=np.float32),
            "traffic_convention": np.array([0.0, 1.0], dtype=np.float32),
            "prev_desired_curvature": np.zeros(1, dtype=np.float32),
            "lateral_control_params": np.array(
                [
                    float(np.linalg.norm(agent_input.ego_statuses[-1].ego_velocity)),
                    float(steering_delay_s),
                ],
                dtype=np.float32,
            ),
        }
    return OpenpilotModelInputs(img_main, img_wide, policy_logical)
