"""
ONNX Runtime inference for openpilot ``driving_vision`` + ``driving_policy``.

Replays the logic in ``selfdrive/modeld/modeld.py`` (InputQueues, desire pulse,
slice + parse) without TinyGrad or VisionIPC. Images are approximated from RGB
using the same 6-channel YUV packing layout as ``modeld/models/README.md``.
"""

from __future__ import annotations

import codecs
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser


def _load_output_slices_from_onnx(onnx_path: Path) -> dict[str, slice]:
  import onnx

  model = onnx.load(str(onnx_path))
  raw = None
  for prop in model.metadata_props:
    if prop.key == "output_slices":
      raw = prop.value
      break
  if raw is None:
    raise ValueError(f"output_slices not found in ONNX metadata: {onnx_path}")
  return pickle.loads(codecs.decode(raw.encode(), "base64"))


def _slice_outputs(flat: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
  out: dict[str, np.ndarray] = {}
  for k, v in output_slices.items():
    if k == "pad":
      continue
    out[k] = flat[np.newaxis, v]
  return out


class InputQueues:
  """Copy of modeld ``InputQueues`` (no TinyGrad dependency)."""

  def __init__(self, model_fps: int, env_fps: int, n_frames_input: int):
    assert env_fps % model_fps == 0
    assert env_fps >= model_fps
    self.model_fps = model_fps
    self.env_fps = env_fps
    self.n_frames_input = n_frames_input
    self.dtypes: dict[str, np.dtype] = {}
    self.shapes: dict[str, tuple[int, ...]] = {}
    self.q: dict[str, np.ndarray] = {}

  def update_dtypes_and_shapes(self, input_dtypes: dict[str, np.dtype], input_shapes: dict[str, tuple[int, ...]]) -> None:
    self.dtypes.update(input_dtypes)
    if self.env_fps == self.model_fps:
      self.shapes.update(input_shapes)
    else:
      for k in input_shapes:
        shape = list(input_shapes[k])
        if "img" in k:
          n_channels = shape[1] // self.n_frames_input
          shape[1] = (self.env_fps // self.model_fps + (self.n_frames_input - 1)) * n_channels
        else:
          shape[1] = (self.env_fps // self.model_fps) * shape[1]
        self.shapes[k] = tuple(shape)

  def reset(self) -> None:
    self.q = {k: np.zeros(self.shapes[k], dtype=self.dtypes[k]) for k in self.dtypes}

  def enqueue(self, inputs: dict[str, np.ndarray]) -> None:
    for k in inputs:
      if inputs[k].dtype != self.dtypes[k]:
        raise ValueError(f"input {k} dtype {inputs[k].dtype} != {self.dtypes[k]}")
      input_shape = list(self.shapes[k])
      input_shape[1] = -1
      single_input = inputs[k].reshape(tuple(input_shape))
      sz = single_input.shape[1]
      self.q[k][:, :-sz] = self.q[k][:, sz:]
      self.q[k][:, -sz:] = single_input

  def get(self, *names: str) -> dict[str, np.ndarray]:
    if self.env_fps == self.model_fps:
      return {k: self.q[k] for k in names}
    out: dict[str, np.ndarray] = {}
    for k in names:
      shape = self.shapes[k]
      if "img" in k:
        n_channels = shape[1] // (self.env_fps // self.model_fps + (self.n_frames_input - 1))
        out[k] = np.concatenate(
          [self.q[k][:, s : s + n_channels] for s in np.linspace(0, shape[1] - n_channels, self.n_frames_input, dtype=int)],
          axis=1,
        )
      elif "pulse" in k:
        out[k] = self.q[k].reshape((shape[0], shape[1] * self.model_fps // self.env_fps, self.env_fps // self.model_fps, -1)).max(axis=2)
      else:
        idxs = np.arange(-1, -shape[1], -self.env_fps // self.model_fps)[::-1]
        out[k] = self.q[k][:, idxs]
    return out


def pack_rgb_to_6ch_uint8(rgb_hwc: np.ndarray) -> np.ndarray:
  """RGB uint8 (H, W, 3) -> uint8 (6, 128, 256) model YUV layout."""
  try:
    from PIL import Image
  except ImportError as e:
    raise ImportError("Pillow is required for ONNX image packing") from e

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
  """(6,128,256) * 2 -> (1, 12, 128, 256)."""
  if prev_6 is None:
    prev_6 = cur_6
  stacked = np.concatenate([prev_6, cur_6], axis=0)
  return stacked.reshape(1, 12, 128, 256)


def plan_to_poses_xy_heading(
  plan: np.ndarray,
  trajectory_num_poses: int,
  trajectory_dt: float,
) -> np.ndarray:
  """Interpolate parsed policy plan to [N, 3] = x, y, heading in model frame."""
  row = plan[0] if plan.ndim == 3 else plan
  t_m = np.array(ModelConstants.T_IDXS, dtype=np.float64)
  pos = row[:, Plan.POSITION]
  vel = row[:, Plan.VELOCITY]
  x, y = pos[:, 0].astype(np.float64), pos[:, 1].astype(np.float64)
  vx, vy = vel[:, 0].astype(np.float64), vel[:, 1].astype(np.float64)
  heading = np.arctan2(vy, vx)
  t_out = trajectory_dt * np.arange(1, trajectory_num_poses + 1, dtype=np.float64)
  xi = np.interp(t_out, t_m, x)
  yi = np.interp(t_out, t_m, y)
  hi = np.interp(t_out, t_m, heading)
  return np.stack([xi, yi, hi], axis=1).astype(np.float32)


class OpenpilotOnnxRunner:
  """Vision + policy ONNX, stateful temporal queues (matches modeld)."""

  def __init__(self, models_dir: Path, providers: Optional[list[str]] = None) -> None:
    try:
      import onnxruntime as ort
    except ImportError as e:
      raise ImportError("Install onnxruntime: pip install onnxruntime") from e

    self.models_dir = Path(models_dir)
    vision_onnx = self.models_dir / "driving_vision.onnx"
    policy_onnx = self.models_dir / "driving_policy.onnx"
    if not vision_onnx.is_file() or not policy_onnx.is_file():
      raise FileNotFoundError(f"Need {vision_onnx.name} and {policy_onnx.name} under {self.models_dir}")

    opts = ort.SessionOptions()
    prov = providers if providers is not None else ort.get_available_providers()
    self._vision = ort.InferenceSession(str(vision_onnx), sess_options=opts, providers=prov)
    self._policy = ort.InferenceSession(str(policy_onnx), sess_options=opts, providers=prov)

    self.vision_output_slices = _load_output_slices_from_onnx(vision_onnx)
    self.policy_output_slices = _load_output_slices_from_onnx(policy_onnx)

    self.parser = Parser()
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    pinfos = {i.name: i for i in self._policy.get_inputs()}
    self._policy_shapes = {n: tuple(int(d) if isinstance(d, int) else 1 for d in pinfos[n].shape) for n in pinfos}
    self._policy_fp16 = {n: "float16" in pinfos[n].type for n in pinfos}
    self.numpy_inputs = {k: np.zeros(self._policy_shapes[k], dtype=np.float32) for k in self._policy_shapes}

    self.full_input_queues = InputQueues(ModelConstants.MODEL_CONTEXT_FREQ, ModelConstants.MODEL_RUN_FREQ, ModelConstants.N_FRAMES)
    for k in ("desire_pulse", "features_buffer"):
      self.full_input_queues.update_dtypes_and_shapes(
        {k: self.numpy_inputs[k].dtype},
        {k: self.numpy_inputs[k].shape},
      )
    self.full_input_queues.reset()

    self._prev_pack_main: Optional[np.ndarray] = None
    self._prev_pack_big: Optional[np.ndarray] = None

    self._vision_in_names = [i.name for i in self._vision.get_inputs()]
    self._policy_in_names = [i.name for i in self._policy.get_inputs()]
    vinfo = {i.name: i for i in self._vision.get_inputs()}
    self._vision_uint8 = {n: "uint8" in vinfo[n].type for n in vinfo}

  def reset(self) -> None:
    self.prev_desire[:] = 0
    self.full_input_queues.reset()
    self._prev_pack_main = None
    self._prev_pack_big = None

  def run_step(
    self,
    rgb_main: np.ndarray,
    rgb_big: np.ndarray,
    desire_onehot: np.ndarray,
    traffic_convention: np.ndarray,
  ) -> dict[str, np.ndarray]:
    p_main = pack_rgb_to_6ch_uint8(rgb_main)
    p_big = pack_rgb_to_6ch_uint8(rgb_big)
    img = stack_two_frames_12ch(self._prev_pack_main, p_main)
    big_img = stack_two_frames_12ch(self._prev_pack_big, p_big)
    self._prev_pack_main = p_main
    self._prev_pack_big = p_big

    vfeeds = {}
    for n in self._vision_in_names:
      vfeeds[n] = img if n == "img" else big_img
      if self._vision_uint8.get(n):
        vfeeds[n] = np.asarray(vfeeds[n], dtype=np.uint8)

    vout = self._vision.run(None, vfeeds)[0]
    vflat = np.asarray(vout, dtype=np.float32).reshape(-1)
    vision_out = self.parser.parse_vision_outputs(_slice_outputs(vflat, self.vision_output_slices))

    pulse = np.asarray(desire_onehot, dtype=np.float32).reshape(-1).copy()
    assert pulse.shape[0] == ModelConstants.DESIRE_LEN
    pulse[0] = 0.0
    new_desire = np.where(pulse - self.prev_desire > 0.99, pulse, 0.0)
    self.prev_desire[:] = pulse

    hb = np.asarray(vision_out["hidden_state"], dtype=np.float32).reshape(1, -1)
    self.full_input_queues.enqueue({"features_buffer": hb, "desire_pulse": new_desire.reshape(1, -1)})
    for k in ("desire_pulse", "features_buffer"):
      self.numpy_inputs[k][:] = self.full_input_queues.get(k)[k]

    self.numpy_inputs["traffic_convention"][:] = np.asarray(traffic_convention, dtype=np.float32).reshape(
      self._policy_shapes["traffic_convention"]
    )

    pfeeds = {}
    for n in self._policy_in_names:
      arr = np.asarray(self.numpy_inputs[n])
      if self._policy_fp16.get(n):
        arr = arr.astype(np.float16)
      pfeeds[n] = arr

    pout = self._policy.run(None, pfeeds)[0]
    pflat = np.asarray(pout, dtype=np.float32).reshape(-1)
    policy_out = self.parser.parse_policy_outputs(_slice_outputs(pflat, self.policy_output_slices))
    return {**vision_out, **policy_out}
